#!/bin/bash
export LAUNCHER="accelerate launch \
      --num_processes  ${SLURM_GPUS}  \
      --num_machines ${SLURM_NNODES} \
      --rdzv_backend static \
      --machine_rank ${SLURM_NODEID} \
      --main_process_ip ${master_ip} \
      --main_process_port ${MASTER_PORT} \
      --same_network \
      "
export SCRIPT="./finetune.py"

export SCRIPT_ARGS=" \
      --mixed_precision no
      "

echo $LAUNCHER $SCRIPT $SCRIPT_ARGS
$LAUNCHER $SCRIPT $SCRIPT_ARGS
