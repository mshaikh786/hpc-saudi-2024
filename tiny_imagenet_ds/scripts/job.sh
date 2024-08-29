#!/bin/bash -l

#SBATCH -J bench-tinyIM-ds
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH -c 6
#SBATCH --mem=50G
#SBATCH -t 01:0:0
#SBATCH -A c2227
#SBATCH -C rtx2080ti

scontrol show job ${SLURM_JOBID}
#rm -rf /home/shaima0d/.cache/torch_extensions/*
export CUDA_HOME=${CONDA_PREFIX}
source /ibex/user/shaima0d/miniconda3/bin/activate /ibex/ai/home/shaima0d/KSL_Trainings/hpc-saudi-2024/ds-env
export OMP_NUM_THREADS=1
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"


# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
export master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${MASTER_PORT}"

workers=${SLURM_CPUS_PER_TASK}

echo "Hostname: $(/bin/hostname)"
echo "CPU workers: $workers"

start=$(date +%s)
for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
    srun --cpu-bind=cores -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
    python -m torch.distributed.launch --use_env --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
    --master_addr=${master_ip} --master_port=${MASTER_PORT}  ../scripts/train_resnet50_ds.py  --epochs 30 --num-workers=${SLURM_CPUS_PER_TASK}\
    --deepspeed --deepspeed_config ../scripts/ds_config.json \
    --log-interval 100 &
done
wait
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
