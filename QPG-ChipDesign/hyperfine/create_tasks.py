import json
from pathlib import Path
import numpy as np

filename = "main.py"
preamble = ""
output_file = "~/public_html/aug_11_F_plot_longer"
cpus_per_task = 3

### params ###

# add 1- trace plot

twopi = 2*3.1415926535

params = {

    ### PROBLEM SETUP ###

    'qubits':           [6],
    'target_qubits':    [[1]],#[4, 5, 6]],#[[4, 5, 6]],#[{4, 5, 6}],

    # 'T':      #[0.5,],#[0.5, 0.25],
    # # [0.05 , 0.1  , 0.15 , 0.2  , 0.25 , 0.3  , 0.35 , 0.4  , 0.45 ,
    # #    0.5  , 0.55 , 0.6  , 0.65 , 0.7  , 0.75 , 0.8  , 0.85 , 0.9  ,
    # #    0.95 , 1.   , 1.05 , 1.1  , 1.15 , 1.2  , 1.25 , 1.3  , 1.35 ,
    # #    1.4  , 1.45 , 1.5, 1.55, 1.6 , 1.65 ,
    # #    1.7  , 1.75, 1.8 , 1.85 ,
    # #    1.9 , 1.95, 2.   ] ,
    # np.linspace(0.05, 6.0, 120),     

    'd':                [1],

    ### PARAMETIZATION ###

    'steps':            [1, 100],#[100, 500, 1000],#[200],

    'FFT':              [False],
    'fft_window':       [None], # None, hann 

    'REG_PULSE':        [True], # not implemented yet for fft
    'MAX_SLOPE':        [0.05], # this is not used as of now

    ### INITIAL GUESS ###

    'GUESS_OPT_STEPS':  [None],
    'GUESS_SIZE':       [None],

    ### COLLAPSE ###

    'T2':               [1.8],#1.8],

    ### OPTIMIZATION ###

    'METHOD':           ['S'], # 'S' -> Schrodinger Solver, 'L' -> Lindbladian Master Eq.
    'diff_factor':      [1], #
    'diff_target':      [None],

    # newLR = 0.01 | 10000.
    'LR':               [0.1, 1],#[0.1, 1., 5., 10.],#[0.01],#{'T': np.linspace(0.01, 0.0001, 50)}, #[0.01], #[0.1], #[0.01], #[8.], # LR for FFT w. hann 50, 100 steps -> 2000.
    'OPT_LIM':          [1e-4], #[1e-8],
    'OPT_STEPS':        [1000],#[1000],#[2000],

    ### RESULT FORMAT ###

    'plot_res':         [None], # hmmm this might have a bug?? oof...

    ### CUSTOM VARS ###
    'CUSTOM_splitting': [0.0],#np.linspace(0.1, 7.0, 70)*twopi,
    'CUSTOM_fine_splitting': [1.0],#[2.18*2*3.1415926535], #[2.2,]
    'CUSTOM_ground_splitting': [0.0], #[2.2,]
    'omega_drive': [0],# + 2.2
    'CUSTOM_n': [1],#, 3, 5]
    'CUSTOM_pulse_reg': [0.001],

    'CUSTOM_B': [1.],

    'T': np.linspace(0.05, 10.0, 200),

}
# params = {

#     ### PROBLEM SETUP ###

#     'qubits':           [2],
#     'target_qubits':    [[1]],#[{4, 5, 6}],

#     'T':      [1],
#     # [0.05 , 0.1  , 0.15 , 0.2  , 0.25 , 0.3  , 0.35 , 0.4  , 0.45 ,
#     #    0.5  , 0.55 , 0.6  , 0.65 , 0.7  , 0.75 , 0.8  , 0.85 , 0.9  ,
#     #    0.95 , 1.   , 1.05 , 1.1  , 1.15 , 1.2  , 1.25 , 1.3  , 1.35 ,
#     #    1.4  , 1.45 , 1.5, 1.55, 1.6 , 1.65 ,
#     #    1.7  , 1.75, 1.8 , 1.85 ,
#     #    1.9 , 1.95, 2.   ] ,     
#     'd':                [1],

#     ### PARAMETIZATION ###

#     'steps':            [1, 200],

#     'FFT':              [False],
#     'fft_window':       [None], # None, hann 

#     'REG_PULSE':        [True], # not implemented yet for fft
#     'MAX_SLOPE':        [0.05], # this is not used as of now

#     ### INITIAL GUESS ###

#     'GUESS_OPT_STEPS':  [None],
#     'GUESS_SIZE':       [None],

#     ### COLLAPSE ###

#     'T2':               [None],#1.8],

#     ### OPTIMIZATION ###

#     'METHOD':           ['S'], # 'S' -> Schrodinger Solver, 'L' -> Lindbladian Master Eq.
#     'diff_factor':      [1], #
#     'diff_target':      [None],

#     # newLR = 0.01 | 10000.
#     'LR':               [0.01, 0.1, 1., 100.],#[0.01],#{'T': np.linspace(0.01, 0.0001, 50)}, #[0.01], #[0.1], #[0.01], #[8.], # LR for FFT w. hann 50, 100 steps -> 2000.
#     'OPT_LIM':          [1e-4], #[1e-8],
#     'OPT_STEPS':        [1000],#[2000],

#     ### RESULT FORMAT ###

#     'plot_res':         [200],

#     ### CUSTOM VARS ###
#     'CUSTOM_splitting': [2.5],#[2.5, 5.0, 7.5, 10.0, 25.0], #[2.2, 4.4, 6.6, 8.8, 11.0],
#     'CUSTOM_fine_splitting': [5.0], #[2.2,]
#     'CUSTOM_ground_splitting': [2.0], #[2.2,]
#     'omega_drive': [2.0],# + 2.2
#     'CUSTOM_n': [1]

# }

##############

commands = ""

if '/' == output_file[-1]:
    output_file = output_file[:-1]

runs = [{}]

special_params = []

for param, space in params.items():

    temp = []

    for run in runs:
        if type(space) != dict:

            for val in space:
                run[param] = val
                temp.append(dict(run))

        else:
            special_params.append((param, space))

    if type(space) != dict: runs = temp

for run in runs:
    for param, space in special_params:

        r = run[list(space.keys())[0]]

        index = params[list(space.keys())[0]].index( r )

        run[param] = space[list(space.keys())[0]][index]

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

print(f'run the command:\n sbatch --array=1-{space_len} --partition=newnodes --cpus-per-task={cpus_per_task} --mem=5000  --time=11:59:59 --export=ALL batch_run.sh --job-name='+output_file.split("/")[-1]+';')
