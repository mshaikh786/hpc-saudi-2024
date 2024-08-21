import argparse

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
        "-b",
        "--batch-size",
        default=256,
        type=int,
        help="mini-batch size per GPU",
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
                                           batch_size=args.batch_size,
                                           sampler=trainsampler,drop_last=True,
                                           num_workers=6,
                                           pin_memory=True)
    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

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

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
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
            if local_rank == 0 and i % args.log_interval == (
                args.log_interval - 1
            ):  # Print every log_interval mini-batches.
                print(
                    f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
                )
                running_loss = 0.0
    print("Finished Training")

    ########################################################################
    # Step 4. Test the network on the test data.
    ########################################################################
    #test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    args = add_argument()
    main(args)
