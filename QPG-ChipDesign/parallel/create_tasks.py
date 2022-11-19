import json
from pathlib import Path

filename = "main.py"
preamble = ""
output_file = "~/public_html/units"

### params ###

# add 1- trace plot

params = {

    ### PROBLEM SETUP ###

    'qubits':           [1],
    'target_qubit':     [1],

    'T':                [1],
    'd':                [1],

    ### PARAMETIZATION ###

    'steps':            [1],

    'FFT':              [False],
    'fft_window':       [None], # None, hann 

    'REG_PULSE':        [False], # not implemented yet for fft
    'MAX_SLOPE':        [0.05], #this is not used as of now

    ### INITIAL GUESS ###

    'GUESS_OPT_STEPS':  [None],
    'GUESS_SIZE':       [None],

    ### COLLAPSE ###

    'T2':               [0.],#[1.0, 2.0, 5.0, 20.0, float('inf')],

    ### OPTIMIZATION ###

    'METHOD':           ['S'], # 'S' -> Schrodinger Solver, 'L' -> Lindbladian Master Eq.
    'diff_factor':      [1], #
    'diff_target':      [None],

    'LR':               [0.001], #[0.1], #[0.01], #[8.], # LR for FFT w. hann 50, 100 steps -> 2000.
    'OPT_LIM':          [1e-8],
    'OPT_STEPS':        [1000],

    ### RESULT FORMAT ###

    'plot_res':         [50],

}

##############

commands = ""

if '/' == output_file[-1]:
    output_file = output_file[:-1]

runs = [{}]

for param, space in params.items():

    temp = []

    for run in runs:
        for val in space:

            run[param] = val
            temp.append(dict(run))

    runs = temp

k = 0
for i in runs:
    k += 1
    i = json.dumps(i)
    commands += preamble + f"python {filename} '{i}' {output_file} {k};\n"

f = open("tasks.txt", "w")
f.write(commands)
f.close()

output_folder = Path(output_file).expanduser()

# existence of parents probably suggests mistyping the directory name
output_folder.mkdir(parents=False, exist_ok=True)

with open(output_folder / "_sim_data.json", 'w+') as sim_data:
    sim_data.write(json.dumps( {i+1: v for i, v in enumerate( runs )} ))

space_len = len(runs);

print(f'run the command:\n sbatch --array=1-{space_len}  --partition=sched_any --cpus-per-task=1 --mem=5000  --time=00:15:00 --export=task_file=tasks.txt batch_run.sh')
