import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scipy.linalg as la

### to silence lack of display if not connected to xquartz
import matplotlib as mpl
mpl.use('Agg')
###

import matplotlib.pyplot as plt
from datetime import datetime
import sys
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import scipy
import matplotlib.colors as mpc
import seaborn as sb
import matplotlib.patches as patches
import json

def s_print(*args):
  print(*args)
  sys.stdout.flush()

add_init_states = lambda x: np.concatenate([[0], x])
pop_labels = lambda i: 'pops'+str(i) if i != 1 else 'pops'
purity_labels = lambda i: 'purity'+str(i) if i != 1 else 'purity'

def pltf(T, *args, **kwargs):
    if "alpha" in kwargs.keys():
        return plt.step(np.linspace(0, T, len(args[0])), *args, **kwargs)
    else:
        return plt.step(np.linspace(0, T, len(args[0])), *args, **kwargs, alpha=0.7)

def pltP(T, *args, **kwargs):

    # plt.yscale("log")

    rm_lbl_kwargs = kwargs.copy()
    rm_lbl_kwargs.pop('label', None)

    # plt.plot(np.linspace(0, T, len(args[0])), *args, **rm_lbl_kwargs, linewidth=0.5, alpha=0.1)

    # return plt.scatter(np.linspace(0, T, len(args[0])), *args, **kwargs, s=2, alpha=1)

    plt.plot(np.linspace(0, T, len(args[0])), *args, **kwargs, linewidth=1, alpha=1)

def conversion_function(x):

    x = x.strip('][').replace("\n", "")

    while " +" in x:
      x = x.replace(" +", "+")

    return np.array( [float(i) if 'j' not in i else complex(i) for i in x.split(' ') if i != '' and '...' not in i] )

converters = {t: (lambda x: conversion_function(x)) for t in ['pops', 'pops2', 'pops3', 'pops4', 'pops5', 'pops6', 'I', 'Q', 'purity', 'purity2', 'purity3', 'purity4', 'purity5', 'purity6', 'fft:I', 'fft:Q', 'guess:I', 'guess:Q']}

if __name__ == "__main__":

    pd.set_option('display.html.table_schema', True) # to can see the dataframe/table as a html
    pd.set_option('display.precision', 5)

    FILTER_LOSS = True

    sim_name = '~/public_html/paper_sim_basic/_all.csv'
    
    df = pd.read_csv(sim_name, converters=converters)
    df = df.loc[ (df['data:steps'] == 200) & (df['data:LR'] == 0.1) ]

    print(df[['data:steps', 'data:LR']])

    qubits = 2

    rows, cols = 2, 1

    for idx, sim in df.iterrows():

        plt.figure(figsize=(7, 5))

        plt.gcf().subplots_adjust(left=0.15)

        plt.subplot(rows, cols, 1)
        plt.ylabel('State Evolution')

        plt.title('Optimizing Pulses for 1 target qubit and 1 coupled qubit')

        for i in range(1, qubits+1):
          pltP(sim['T'], add_init_states(list(sim[ pop_labels(i) ])), label="Qubit "+str(i))

        plt.legend()

        # plt.subplot(rows, cols, 4)
        # plt.ylabel('Purity')
        # plt.xlabel('time [$\mu$s]')

        # for i in range(1, qubits+1):
        #   # if sim[ purity_labels(i) ]: sim[ purity_labels(i) ] = []
        #   pltP(sim['T'], [1] + list(sim[ purity_labels(i) ]), label="Qubit "+str(i))

        # plt.legend()

        plt.subplot(rows, cols, 2)
        # plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))
        plt.ylabel('Magnitude [MHz]')

        pltf(sim['T'], add_init_states(list(sim["I"])), 'g', label="I", alpha=1)
        pltf(sim['T'], add_init_states(list(sim["Q"])), 'orange', label="Q", alpha=1)
        plt.legend()

        # plt.subplot(rows, cols, 5)

        plt.xlabel('time [$\mu$s]')
        # pltf(sim['T'], add_init_states( np.sqrt(np.array(sim["I"])**2 + np.array(sim["Q"])**2 )), 'black', label="magnitude", alpha=1)
        # pltf(sim['T'], add_init_states( np.arctan(np.array(sim["Q"]) / np.array(sim["I"]) ) ), 'blue', label="phase", alpha=1)
        # plt.legend()

        plt.savefig('/home/tareq/public_html/paper/_paper_1.png')
        plt.close()

    sim_name = '~/public_html/mar17_type3_24/_all.csv'
    
    df = pd.read_csv(sim_name, converters=converters)
    df = df.loc[ (df['data:CUSTOM_splitting'] == 5) & (df['data:T'] == 2.0) ]

    print(df[['data:T', 'data:CUSTOM_splitting']])

    qubits = 6

    rows, cols = 2, 1

    for idx, sim in df.iterrows():

        plt.figure(figsize=(7, 5))

        plt.gcf().subplots_adjust(left=0.15)

        plt.subplot(rows, cols, 1)
        plt.ylabel('State Evolution')

        # plt.title('Optimizing Pulses for 1 spin with and 1 coupled qubit')

        for i in range(1, qubits+1):
          pltP(sim['T'], add_init_states(list(sim[ pop_labels(i) ])), label='$'+[
            '\\mathrm{spin } 1, m_n = 0', '\\mathrm{spin } 1, m_n = 1', '\\mathrm{spin } 1, m_n = -1',
            '\\mathrm{spin } 2, m_n = 0', '\\mathrm{spin } 2, m_n = 1', '\\mathrm{spin } 2, m_n = -1',][i-1] + '$', 
            color=[
            'green', 'deepskyblue', 'slateblue', 
            'orangered', 'crimson', 'orange'][i-1] )

        plt.legend()

        # plt.subplot(rows, cols, 4)
        # plt.ylabel('Purity')
        # plt.xlabel('time [$\mu$s]')

        # for i in range(1, qubits+1):
        #   # if sim[ purity_labels(i) ]: sim[ purity_labels(i) ] = []
        #   pltP(sim['T'], [1] + list(sim[ purity_labels(i) ]), label="Qubit "+str(i))

        # plt.legend()

        plt.subplot(rows, cols, 2)
        # plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))
        plt.ylabel('Magnitude [MHz]')

        pltf(sim['T'], add_init_states(list(sim["I"])), 'g', label="I", alpha=1)
        pltf(sim['T'], add_init_states(list(sim["Q"])), 'orange', label="Q", alpha=1)
        plt.legend()

        # plt.subplot(rows, cols, 5)

        plt.xlabel('time [$\mu$s]')
        # pltf(sim['T'], add_init_states( np.sqrt(np.array(sim["I"])**2 + np.array(sim["Q"])**2 )), 'black', label="magnitude", alpha=1)
        # pltf(sim['T'], add_init_states( np.arctan(np.array(sim["Q"]) / np.array(sim["I"]) ) ), 'blue', label="phase", alpha=1)
        # plt.legend()

        plt.savefig('/home/tareq/public_html/paper/_paper_2.png')
        plt.close()


        ######

        # plt.figure(figsize=(20, 5))

        # plt.subplot(rows, cols, 1)
        # plt.ylabel('State Evolution')

        # for i in range(1, qubits+1):
        #   pltP(sim['T'], add_init_states(list(sim[ pop_labels(i) ])), label="Qubit "+str(i))

        # plt.legend()

        # plt.subplot(rows, cols, 4)
        # plt.ylabel('Purity')
        # plt.xlabel('time [$\mu$s]')

        # for i in range(1, qubits+1):
        #   pltP(sim['T'], [1] + list(sim[ purity_labels(i) ]), label="Qubit "+str(i))

        # plt.legend()

        # if fft and plot_evos:

        #   plt.subplot(rows, cols, 2)
        #   plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))# plt.title(str(sim['STEPS'])+" STEPS, distance = " + str(sim['distance']) + ", time = " + str(sim['T']))
        #   plt.ylabel('Control Pulses in Time Domain [MHz]')

        #   pltf(sim['T'], add_init_states(list(sim["I"])), 'g', label="I", alpha=1)
        #   pltf(sim['T'], add_init_states(list(sim["Q"])), 'orange', label="Q", alpha=1)
        #   plt.legend()

        #   plt.subplot(rows, cols, 5)

        #   plt.xlabel('time [$\mu$s]')

        #   pltf(sim['T'], add_init_states( np.sqrt(np.array(sim["I"])**2 + np.array(sim["Q"])**2 )), 'black', label="magnitude", alpha=1)
        #   pltf(sim['T'], add_init_states( np.arctan(np.array(sim["Q"]) / np.array(sim["I"]) ) ), 'blue', label="phase", alpha=1)
        #   plt.legend()

        #   plt.subplot(rows, cols, 3)

        #   x_axis_freq = np.fft.fftfreq(sim['data:steps'], d=float(sim['T'])/sim['data:steps'])

        #   plt.xlabel('frequency [MHz] using '+ str(sim['data:steps']) + ' bins out of ' + str(sim['data:steps']*sim['data:diff_factor']) + ', 1 bin = ' + str(round(x_axis_freq[1], 3-int(np.floor(np.log10(np.abs(x_axis_freq[1]))))-1)) + " MHz")

        #   plt.stem(x_axis_freq, sim['fft:I'], 'g', markerfmt='go', label="I freq")
        #   plt.stem(x_axis_freq, sim['fft:Q'], 'y', markerfmt='yo', label="Q freq")
        #   plt.legend()

        # elif plot_evos:

        #   plt.subplot(rows, cols, 2)
        #   plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))
        #   plt.ylabel('Magnitude [MHz]')

        #   pltf(sim['T'], add_init_states(list(sim["I"])), 'g', label="I", alpha=1)
        #   pltf(sim['T'], add_init_states(list(sim["Q"])), 'orange', label="Q", alpha=1)
        #   plt.legend()

        #   plt.subplot(rows, cols, 5)

        #   plt.xlabel('time [$\mu$s]')
        #   pltf(sim['T'], add_init_states( np.sqrt(np.array(sim["I"])**2 + np.array(sim["Q"])**2 )), 'black', label="magnitude", alpha=1)
        #   pltf(sim['T'], add_init_states( np.arctan(np.array(sim["Q"]) / np.array(sim["I"]) ) ), 'blue', label="phase", alpha=1)
        #   plt.legend()

        #   if did_guess:

        #     plt.subplot(rows, cols, 6)

        #     plt.xlabel('time [$\mu$s]')
        #     plt.ylabel('init guess w/ loss: '+ str(sim['guess:loss']).replace("[", "").replace("]", "")[0:4])

        #     pltf(sim['T'], sim['guess:I'].flatten(), 'g', label="initial I", alpha=1)
        #     pltf(sim['T'], sim['guess:Q'].flatten(), 'orange', label="initial Q", alpha=1)
        #     plt.legend()

        # plt.savefig('~/paper/_paper_1.png')
        # plt.close()

    s_print("done w/ all _evo")
