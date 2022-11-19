sbatch --array=1-*** --partition=sched_any --cpus-per-task=1 --mem=5000  --time=00:05:00 --export=task_file=tasks.txt batch_run.sh
