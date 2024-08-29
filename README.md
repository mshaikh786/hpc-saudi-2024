# Scaling DL models on HPC resources

 
## Software installation

MS DeepSpeed can be installed as ```conda``` environment. The following command shows how to install such an envrionment on an x86 machine running Linux opreating system:

```
conda env create -f conda/deepspeed.yaml 
```

After installation, a couple of things may need to change:

1. If you are running on an HPC cluster which shares the compute nodes between jobs by multiple users, it is advisable to refactor the ```deepspeed``` setup to use a TCP port prescribed by the user for use in multi-node training.
Please edit the file ```${CONDA_PREFIX}/lib/python3.10/site-packages/deepspeed/constant.py``` and add the following code in-place where ```TORCH_DISTRIBUTED_DEFAULT_PORT``` is defined.

```
import socket
s=socket.socket()
s.bind(("", 0))

if ('MASTER_PORT' in os.environ):
    TORCH_DISTRIBUTED_DEFAULT_PORT = os.environ['MASTER_PORT']
else:
    port = s.getsockname()[1]
    TORCH_DISTRIBUTED_DEFAULT_PORT = port
s.close()
```

With the above change, the environment variable ```MASTER_PORT```, if set in the shell, will be honored and used. It should be passed as a value of the as a command line arguement option ```--master-port``` when invoking the python training script with ```deepspeed``` integtaion.

2. Fused optimized kernels are compiled in-situ. The following is necessary for the required headers and libraries to be found. Create symbolic links manually in the paths know to the compilers. Assuming the ```conda``` environment is active:

```
ln -s ${CONDA_PREFIX}/targets/x86_64-linux/include/* ${CONDA_PREFIX}/include
ln -s ${CONDA_PREFIX}/targets/x86_64-linux/lib/* ${CONDA_PREFIX}/lib

```
Set the following variable in the runner script helps DeepSpeed to compile a kernel multiple GPU microarchitectures. E.g. the following enables compilation for NVIDIA Volta, Ampere architetures.

``` 
export TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0'
```
