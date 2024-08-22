#!/bin/bash

for g in 1 2 4 8 16
do	echo $g
	rundir=run_${g}
	mkdir -p ${rundir}
	cd ${rundir}
	cp ../scripts/job.sh .
	trainscript=../scripts/train_resnet50_ds.py
	if [ ${g} -gt 8 ]
	then
		sbatch --gpus=${g} --gpus-per-node=8 -n 2 --tasks-per-node=1 -c 10 job.sh
	else
		sbatch --gpus=${g} --gpus-per-node=${g} -n 1 --tasks-per-node=1 -c 10 job.sh
	fi
	cd ..
done


