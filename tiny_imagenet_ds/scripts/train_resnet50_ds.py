import argparse,os

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer


def add_argument():
    parser = argparse.ArgumentParser(description="Tinyimagenet")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        help="number of dataloader cpus (default: 4)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    print(args)
    return args


def main(args):
    # Initialize DeepSpeed distributed backend.
    deepspeed.init_distributed()
     ########################################################################
    # Step 2. Define the network with DeepSpeed.
    #
    # First, we define a Convolution Neural Network.
    # Then, we define the DeepSpeed configuration dictionary and use it to
    # initialize the DeepSpeed engine.
    ########################################################################
    net = models.resnet50()

    # Get list of parameters that require gradients.
    #parameters = filter(lambda p: p.requires_grad, net.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=net.parameters(),
    )
    micro_batch_size=int(model_engine.train_micro_batch_size_per_gpu())
    global_batch_size=int(model_engine.train_batch_size())

   ########################################################################
    # Step1. Data Preparation.
    #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    #
    # Note:
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    ########################################################################


    # Load or download cifar data.
    trainset = datasets.ImageFolder("/ibex/reference/CV/tinyimagenet/train",
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                           batch_size=micro_batch_size,
                                           sampler=trainsampler,
                                           shuffle=(trainsampler is None),
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
        # Load or download cifar data.
    valset = datasets.ImageFolder("/ibex/reference/CV/tinyimagenet/train",
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
    valsampler = torch.utils.data.distributed.DistributedSampler(valset,shuffle=False)
    valloader = torch.utils.data.DataLoader(trainset, 
                                           batch_size=micro_batch_size,
                                           sampler=valsampler,
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    global_rank= model_engine.global_rank

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Define the Classification Cross-Entropy loss function.
    criterion = nn.CrossEntropyLoss()

    ########################################################################
    # Step 3. Train the network.
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize. (DeepSpeed handles the distributed details for us!)
    ########################################################################
    print(f'rank[{global_rank},{local_rank}]:: Batch size={micro_batch_size} , Global batch size={global_batch_size}')
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model_engine.train()
        for i, data in enumerate(trainloader):
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # Try to convert to target_dtype if needed.
            if target_dtype != None:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()
            # Print statistics
            running_loss += loss.item()
            if global_rank == 0 and i % args.log_interval == (
                args.log_interval - 1
            ):  # Print every log_interval mini-batches.
                print(
                    f"[Epoch: {epoch+1:5d}, batch:{i+1:5d}] Training loss: {running_loss / args.log_interval : .3f}"
                )
                running_loss = 0.0

        # Validation 
        running_loss_v  = 0.0
        model_engine.eval()
        with torch.no_grad():
            for ii, data_v in enumerate(valloader):
                inputs_v,labels_v = data_v[0].to(local_device), data_v[1].to(local_device)
                # Try to convert to target_dtype if needed
                if target_dtype != None:
                    inputs_v = inputs_v.to(target_dtype)
                outputs_v = model_engine(inputs_v)
                loss_v  = criterion(outputs_v, labels_v)
                running_loss_v  += loss_v.item()
        if global_rank == 0:
            print(
                f"[Epoch: {epoch + 1 :5d}]       Validation loss: {running_loss_v / ii  : .3f}"
            )

    print("Finished Training")

    ########################################################################
    # Step 4. Test the network on the test data.
    ########################################################################
    #test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    args = add_argument()
    main(args)
