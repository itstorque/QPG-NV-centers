#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tareq/public_html/logs/%A_%a.out
#SBATCH --error=/home/tareq/public_html/logs/%A_%a.err
#S BATCH --mail-type=ALL
#S BATCH --mail-user=tareq@mit.edu
#S BATCH --constraint=centos7
#SBATCH -x node339,node119,node129

module load engaging/python/3.6.0
# module load python/3.6.0

echo "Slurm batch_run starting in " `pwd`

echo $SLURM_JOB_NODELIST

if [ -z "$task_file" ]
then
  task_file="tasks.txt"
fi

command=`sed -n "$SLURM_ARRAY_TASK_ID p" $task_file`

echo "Task taken from file $task_file line $SLURM_ARRAY_TASK_ID"
echo "Command is \`$command\`"

echo "Starting job\n\n"

# cd /home/tareq/QPG-ChipDesign/parallel

time_format_string="\nWall clock: %E\nPage faults: %F\nAvg mem use: %Kkb\nMax res mem use:%Mkb\nCPU usage: %P\nExit: %x\n"
/usr/bin/time -f"$time_format_string" bash -c "$command"

echo "\n\nSlurm batch_run done"
