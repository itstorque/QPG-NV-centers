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

if __name__ == "__main__":

    FILTER_LOSS = True

    folder_name = sys.argv[1]
    files = [folder_name+"/"+i for i in listdir(folder_name) if ('.csv' in i) and (i[0] != "_")]

    s_print("begin", folder_name)

    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    try:
      df = pd.read_csv(files[0])
    except:
      s_print("No simulation files")
      raise FileNotFoundError

    for i, x in json.loads(df['data'].iloc[0]).items():
      df['data:' + i] = [x]

    for file in files[1:]:

        temp = pd.read_csv(file)

        for i, x in json.loads(temp['data'].iloc[0]).items():
          temp['data:' + i] = [x]

        df = pd.concat([df, temp], ignore_index=True)

    df.index += 1

    df.to_csv(folder_name+'/_all.csv')

    def conversion_function(x):

        x = x.strip('][').replace("\n", "")

        while " +" in x:
          x = x.replace(" +", "+")

        return np.array( [float(i) if 'j' not in i else complex(i) for i in x.split(' ') if i != '' and '...' not in i] )

    converters = {t: (lambda x: conversion_function(x)) for t in ['pops', 'pops2', 'pops3', 'pops4', 'I', 'Q', 'purity', 'purity2', 'purity3', 'purity4', 'fft:I', 'fft:Q', 'guess:I', 'guess:Q']}

    # reload using the converters to parse numpy arrays and tensors
    df = pd.read_csv(folder_name+"/_all.csv", converters=converters)

    x_is_time = sys.argv[2]=="t"

    # fft = 'fft' in sys.argv
    fft = 'fft:I' in df.columns
    fft = fft and df['fft:I'].iloc[0]

    varx, varxname = ('T', 'time') if x_is_time else ('distance', 'distance')
    vary, varyname = 'STEPS', 'steps'

    for i in sys.argv:
      if 'varx=' == i[0:5]: varx, varxname = 'data:'+i[5:], i[5:]
      if 'vary=' == i[0:5]: vary, varyname = 'data:'+i[5:], i[5:]

    inversex, inversey = 'invx' in sys.argv, 'invy' in sys.argv

    if inversex: varxname = "1/"+varxname
    if inversey: varyname = "1/"+varyname

    if fft and varyname=="steps": varyname = 'bins'

    did_guess = 'guess:loss' in df.columns
    did_reg = df['data:REG_PULSE'].iloc[0]==True
    s_print(did_reg, df['data:REG_PULSE'])

    group_LR = len(df.groupby(['LR'])) > 1

    cmap = plt.cm.plasma
    cmaplog = plt.cm.Spectral

    df.replace("Infinity", float('inf'))

    if inversex: df["1/"+varx.replace("data:", "")] = 1/df[varx]; varx = "1/"+varx.replace("data:", "")
    if inversey: df["1/"+vary.replace("data:", "")] = 1/df[vary]; vary = "1/"+vary.replace("data:", "")

    for lr_aid, df_filtered in df.groupby(['LR']):

      try: ## LOSS CONTOUR ##

        x = df_filtered[varx].values
        y = df_filtered[vary].values
        z = df_filtered['loss'].values

        step_space = np.sort(df_filtered[vary].unique())

        s_print(df_filtered[[varx, vary, 'loss']])

        df_table = open(folder_name+"/_df_data.html", "w")
        df_table.write( df_filtered[[varx, vary, 'loss']].to_html(index=True, classes='dataframe ui green small sortable table', justify="left", border=0, bold_rows=False) )
        df_table.close()

        s_print("saved _df_data")

        X = np.unique(x)
        Y = np.unique(y)
        Z = np.zeros((len(Y), len(X)))

        for i in range(len(z)):
          xi = np.where(X == x[i])[0][0]
          yi = np.where(Y == y[i])[0][0]
          Z[yi, xi] = z[i]

        fig, ax = plt.subplots(figsize=(30, 8))

        CS = ax.contourf(X,Y,Z, np.arange(0, 1.00001, .05), cmap=cmap)
        CSlog = ax.contourf(X,Y,Z, np.linspace(np.log(1-.00001), 0, 10), cmap=cmaplog)

        delta_spacing = lambda x: (max(x)-min(x))/len(np.unique(x))
        extent = lambda x, y: [min(x)-delta_spacing(x), max(x)+delta_spacing(x), min(y)-delta_spacing(y), max(y)+delta_spacing(y)]
        ratio = lambda x, y: 8*(max(x)-min(x))/(12*(max(y)-min(y)))

        ax = plt.subplot(121)
        plt.imshow(Z, extent=extent(x, y), origin='lower', cmap=cmap, aspect=ratio(x, y))
        ax.set_yticks(step_space)
        c = plt.colorbar(CS)
        c.set_label('$\mathrm{LOSS}$', rotation=270)
        c.ax.get_yaxis().labelpad = 15
        plt.clim(0,1)

        ax.set_xlabel(varxname)#"distance [$\mu$m]" if not x_is_time else 'time')
        ax.set_ylabel(varyname)#"STEPS within T")

        ax = plt.subplot(122)
        im = plt.imshow(Z, extent=extent(x, y), origin='lower',
                   cmap=cmaplog, aspect=ratio(x, y), norm=mpc.LogNorm(vmin=0.0001, vmax=1))
        ax.set_yticks(step_space)
        c = plt.colorbar(im, ticks=[0.0001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], format='$%.2f$')
        c.set_label('$\log{\mathrm{LOSS}}$', rotation=270)
        c.ax.get_yaxis().labelpad = 15
        ax.set_xlabel(varxname)
        ax.set_ylabel(varyname)

        if group_LR: plt.savefig(folder_name+'/_LR='+str(lr_aid)+'_colormap.png')
        else: plt.savefig(folder_name+'/_colormap.png')
        s_print("saved _colormap")

      except Exception as e: s_print("No _colormap generated\n\n", e, "\n\n")

      try: ## LOSS PLOTS ##

        fg = sb.FacetGrid(df, hue=vary, col="STEPS")
        fg.map(plt.scatter, varx, "loss", alpha=0.5)
        fg.add_legend()

        if group_LR: plt.savefig(folder_name+'/_LR='+str(lr_aid)+'_loss.png')
        else: plt.savefig(folder_name+'/_loss.png')

        s_print("saved _loss")

      except Exception as e: s_print("No _loss generated\n\n", e, "\n\n")

      ## LOSS HIST ##

      groups = int(df[[vary]].nunique())

      fig, axs = plt.subplots(3 if did_reg else 1, groups, figsize=(groups*10, 3.5*3 if did_reg else 3.5))

      i=-1

      try:
        axs[0]
        axs = axs.flatten()
      except: axs = [axs]

      try:
        axs[0].set_ylabel('total loss')
        axs[groups].set_ylabel('pulse reg loss')
        axs[2*groups].set_ylabel('final state loss')
      except Exception as e: s_print("  ", e)

      for aid, grp in df.groupby([vary]):
          i+=1
          axs[i].set_title(varyname+' = '+str(aid))
          for aid2, grp in grp.groupby([varx]):
              for row in grp['loss_hist'].values:
                  row = [float(i.split('[')[-1].split(']')[0]) for i in row[1:-1].split(',')]
                  if FILTER_LOSS or row[-1]>0.1:
                      axs[i].plot(row,label=varxname+" = "+str(aid2), alpha=0.5)
              if did_reg:
                try:
                  for row in grp['pulse_loss_hist'].values:
                      if row != None:
                        row = [float(i.split('[')[-1].split(']')[0]) for i in row[1:-1].split(',')]
                        if FILTER_LOSS or row[-1]>0.1:
                            axs[i+groups].plot(row,label=varxname+" = "+str(aid2), alpha=0.5)
                  for row in grp['target_loss_hist'].values:
                      if row != None:
                        row = [float(i.split('[')[-1].split(']')[0]) for i in row[1:-1].split(',')]
                        if FILTER_LOSS or row[-1]>0.1:
                            axs[i+2*groups].plot(row,label=varxname+" = "+str(aid2), alpha=0.5)
                except: pass

      for ax in axs:
        ax.legend()
      plt.savefig(folder_name+'/_loss_hist.png')
      if group_LR: plt.savefig(folder_name+'/_LR='+str(lr_aid)+'_loss_hist.png')
      else: plt.savefig(folder_name+'/_loss_hist.png')

      s_print("saved _loss_hist")

      qubits = len([i for i in df.columns if 'pops' in i])
      pop_labels = lambda i: 'pops'+str(i) if i != 1 else 'pops'
      purity_labels = lambda i: 'purity'+str(i) if i != 1 else 'purity'

      rows, cols = 2, 3

      add_init_states = lambda x: np.concatenate([[0], x])

      def pltf(T, *args, **kwargs):
          if "alpha" in kwargs.keys():
              return plt.step(np.linspace(0, T, len(args[0])), *args, **kwargs)
          else:
              return plt.step(np.linspace(0, T, len(args[0])), *args, **kwargs, alpha=0.7)

      def pltP(T, *args, **kwargs):

          # plt.yscale("log")

          rm_lbl_kwargs = kwargs.copy()
          rm_lbl_kwargs.pop('label', None)

          plt.plot(np.linspace(0, T, len(args[0])), *args, **rm_lbl_kwargs, linewidth=0.5, alpha=0.1)

          return plt.scatter(np.linspace(0, T, len(args[0])), *args, **kwargs, s=2, alpha=1)

      for idx, sim in df.iterrows():

          plt.figure(figsize=(20, 5))

          plt.subplot(rows, cols, 1)
          plt.ylabel('State Evolution')

          for i in range(1, qubits+1):
            pltP(sim['T'], add_init_states(list(sim[ pop_labels(i) ])), label="Qubit "+str(i))

          plt.legend()

          plt.subplot(rows, cols, 4)
          plt.ylabel('Purity')
          plt.xlabel('time [$\mu$s]')

          for i in range(1, qubits+1):
            pltP(sim['T'], [1] + list(sim[ purity_labels(i) ]), label="Qubit "+str(i))

          plt.legend()

          if fft:

            plt.subplot(rows, cols, 2)
            plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))# plt.title(str(sim['STEPS'])+" STEPS, distance = " + str(sim['distance']) + ", time = " + str(sim['T']))
            plt.ylabel('Control Pulses in Time Domain [mT]')

            pltf(sim['T'], add_init_states(list(sim["I"])), 'g', label="I", alpha=1)
            pltf(sim['T'], add_init_states(list(sim["Q"])), 'orange', label="Q", alpha=1)
            plt.legend()

            plt.subplot(rows, cols, 5)

            plt.xlabel('time [$\mu$s]')

            pltf(sim['T'], add_init_states( np.sqrt(np.array(sim["I"])**2 + np.array(sim["Q"])**2 )), 'black', label="magnitude", alpha=1)
            pltf(sim['T'], add_init_states( np.arctan(np.array(sim["Q"]) / np.array(sim["I"]) ) ), 'blue', label="phase", alpha=1)
            plt.legend()

            plt.subplot(rows, cols, 3)

            x_axis_freq = np.fft.fftfreq(sim['data:steps'], d=float(sim['T'])/sim['data:steps'])

            plt.xlabel('frequency [MHz] using '+ str(sim['data:steps']) + ' bins out of ' + str(sim['data:steps']*sim['data:diff_factor']) + ', 1 bin = ' + str(round(x_axis_freq[1], 3-int(np.floor(np.log10(np.abs(x_axis_freq[1]))))-1)) + " MHz")

            plt.stem(x_axis_freq, sim['fft:I'], 'g', markerfmt='go', label="I freq")
            plt.stem(x_axis_freq, sim['fft:Q'], 'y', markerfmt='yo', label="Q freq")
            plt.legend()

          else:

            plt.subplot(rows, cols, 2)
            plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))
            plt.ylabel('Magnitude [mT]')

            pltf(sim['T'], add_init_states(list(sim["I"])), 'g', label="I", alpha=1)
            pltf(sim['T'], add_init_states(list(sim["Q"])), 'orange', label="Q", alpha=1)
            plt.legend()

            plt.subplot(rows, cols, 5)

            plt.xlabel('time [$\mu$s]')
            pltf(sim['T'], add_init_states( np.sqrt(np.array(sim["I"])**2 + np.array(sim["Q"])**2 )), 'black', label="magnitude", alpha=1)
            pltf(sim['T'], add_init_states( np.arctan(np.array(sim["Q"]) / np.array(sim["I"]) ) ), 'blue', label="phase", alpha=1)
            plt.legend()

            if did_guess:

              plt.subplot(rows, cols, 6)

              plt.xlabel('time [$\mu$s]')
              plt.ylabel('init guess w/ loss: '+ str(sim['guess:loss']).replace("[", "").replace("]", "")[0:4])

              pltf(sim['T'], sim['guess:I'].flatten(), 'g', label="initial I", alpha=1)
              pltf(sim['T'], sim['guess:Q'].flatten(), 'orange', label="initial Q", alpha=1)
              plt.legend()

          plt.savefig(folder_name+'/_evo_'+str(idx+1)+'.png')

      s_print("done w/ all _evo")

      for idx, sim in df.iterrows():

          fig, ax = plt.subplots(1, figsize=(10, 2.5))

          norm = mpc.LogNorm(vmin=0.0001, vmax=1)

          val = sim['loss']

          plt.ylabel('State Evolution')
          plt.title(varxname+" = "+str(sim[varx])+", "+varyname+" = "+str(sim[vary]))#str(sim['STEPS'])+" STEPS, distance = " + str(sim['distance']) + ", time = " + str(sim['T']))
          p = patches.Rectangle((0,0.9),sim['T']/10,0.1,linewidth=1,edgecolor='none',facecolor=cmaplog(norm(val+0.0001)))
          ax.add_patch(p)

          for i in range(1, qubits+1):
            pltP(sim['T'], add_init_states(list(sim[ pop_labels(i) ])))

          plt.savefig(folder_name+'/_pops_'+str(idx+1)+'.png')

      s_print("done w/ all _pops")
