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

def conversion_function(x):

  x = x.strip('][').replace("\n", "")

  while " +" in x:
    x = x.replace(" +", "+")

  return np.array( [float(i) if 'j' not in i else complex(i) for i in x.split(' ') if i != '' and '...' not in i] )

if __name__ == "__main__":

  pd.set_option('display.html.table_schema', True) # to can see the dataframe/table as a html
  pd.set_option('display.precision', 5)

  # FILTER_LOSS = True

  k = 'jul30_2qb_small_splitting_w_decay'

  count2qbit = False
  splitting_title = 'data:CUSTOM_splitting'
  div_splitting_norm = True # to account for bug where we had time /2pi all splitting is *2pi so need to divide by 2pi to get nice values again
  steps_instead_of_method = False

  # if k=='mar12':   d = {'mar12_type1_alt': 0, 'mar12_type2_alt': 0, 'mar12_type3_alt': 0, 'mar12_type4_bigLR': 0}
  # elif k=='mar17': d = {'mar17_type1_24': 0, 
  #                       'mar17_type1_24_2osc': 0, 
  #                       'mar17_type1_24_3osc': 0, 
  #                       'mar17_type2_24_1osc': 0, 
  #                       'mar17_type2_24_2osc': 0, 
  #                       'mar17_type2_24_3osc': 0, 
  #                       'mar17_type3_24': 0}
  # elif k=='mar17special': d = {'mar17_specialtype1_24': 0, 
  #                       'mar17_specialtype1_24_2osc': 0, 
  #                       'mar17_specialtype1_24_3osc': 0, 
  #                       'mar17_specialtype2_24': 0, 
  #                       'mar17_specialtype2_24_2osc': 0, 
  #                       'mar17_specialtype2_24_3osc': 0, 
  #                       'mar17_specialtype3_24': 0, 
  #                       'mar17_specialtype5_24': 0}
  # elif k=='mar31': d = {'mar31_type_3': 0, 
  #                       'mar31_type_2': 0,
  #                       'mar31_type_2_2osc': 0,
  #                        'mar31_type_2_3osc': '2 (3)',
  #                        'mar31_type_1': '1 (1)',
  #                        'mar31_type_1_2osc': '1 (2)',
                         # 'mar31_type_1_3osc': '1 (3)',}

  if k=='mar12':   lbls = {'mar12_type1_alt': 'Optimize 1 state w/ Square', 'mar12_type2_alt': 'Optimize both hyperfines w/ Square', 'mar12_type3_alt': 'Optimize all w/ regularization', 'mar12_type4_bigLR': 'Optimize all w/ FFT+hann'}
  elif k=='mar17': lbls = {'mar17_type1_24': 'Optimize 1 state w/ Square (pi/2)', 
                           'mar17_type1_24_2osc': 'Optimize 1 state w/ Square (3pi/2)',
                           'mar17_type1_24_3osc': 'Optimize 1 state w/ Square (5pi/2)',
                           'mar17_type2_24_1osc': 'Optimize both hyperfines w/ Square (pi/2)',
                           'mar17_type2_24_2osc': 'Optimize both hyperfines w/ Square (3pi/2)',
                           'mar17_type2_24_3osc': 'Optimize both hyperfines w/ Square (5pi/2)',
                           'mar17_type3_24': 'Optimize all w/ regularization (pi/2)',
                           }
  elif k=='mar17special': lbls = {'mar17_specialtype1_24': 'Optimize 1 state w/ Square (pi/2)', 
                                 'mar17_specialtype1_24_2osc': 'Optimize 1 state w/ Square (3pi/2)',
                                 'mar17_specialtype1_24_3osc': 'Optimize 1 state w/ Square (5pi/2)',
                                 'mar17_specialtype2_24': 'Optimize both hyperfines w/ Square (pi/2)',
                                 'mar17_specialtype2_24_2osc': 'Optimize both hyperfines w/ Square (3pi/2)',
                                 'mar17_specialtype2_24_3osc': 'Optimize both hyperfines w/ Square (5pi/2)',
                                 'mar17_specialtype3_24': 'Optimize all w/ regularization (pi/2)',
                                 'mar17_specialtype5_24': 'Optimize all w/ no regularization (pi/2)',
                                 }
  elif k=='mar31': lbls = {
                           'mar31_type_1': '1 (1)',
                           'mar31_type_1_2osc': '1 (2)',
                           'mar31_type_1_3osc': '1 (3)',
                           'mar31_type_2': '2 (1)',
                           'mar31_type_2_2osc': '2 (2)',
                           'mar31_type_2_3osc': '2 (3)',
                           'redo2_mar31_type_3': '3 (1)', 
                           # 'mar31_type_3_test': '3 (3)', 
                           }
  elif k=='jun10': lbls = {
                           'jun10_plots_1step_shro': '1 (1)',
                           'jun10_plots_1step_3pi_shro': '1 (2)',
                           'jun10_plots_1step_5pi_shro': '1 (3)',
                           # 'mar31_type_2': '2 (1)',
                           # 'mar31_type_2_2osc': '2 (2)',
                           # 'mar31_type_2_3osc': '2 (3)',
                           'jun10_plots_guess10_lr5_shro': '3 (1)', 
                           # 'mar31_type_3_test': '3 (3)', 
                           }
  elif k=='jun10_lindblad': lbls = {
                           'jun10_plots_1step': '1 (1)',
                           'jun10_plots_1step_3pi': '1 (2)',
                           'jun10_plots_1step_5pi': '1 (3)',
                           # 'mar31_type_2': '2 (1)',
                           # 'mar31_type_2_2osc': '2 (2)',
                           # 'mar31_type_2_3osc': '2 (3)',
                           'jun10_plots_guess10_lr5': '3 (1)', 
                           # 'mar31_type_3_test': '3 (3)', 
                           }
  elif k=='jun10_extended': lbls = {
                           'jun10_plots_1step_shro': 'Square pulse (1)',
                           'jun10_plots_1step_3pi_shro': 'Square pulse (2)',
                           'jun10_plots_1step_1.1': 'Square pulse (1)', 
                           'jun10_plots_1step_3.3': 'Square pulse (1)', 
                           'jun10_plots_1step_5pi_shro': 'Square pulse (3)',
                           # 'mar31_type_2': '2 (1)',
                           # 'mar31_type_2_2osc': '2 (2)',
                           # 'mar31_type_2_3osc': '2 (3)',
                           'jun10_plots_100step_1.1': 'Optimized with regularization (1)', 
                           'jun10_plots_100step_3.3': 'Optimized with regularization (1)', 
                           'jun10_plots_guess10_lr5_shro': 'Optimized with regularization (1)', 
                           # 'mar31_type_3_test': '3 (3)', 
                           }
  elif k=="jun15_simple": 
                          lbls = {
                           'jun15_plots_1_2qb': '1 (1)',
                           'jun15_plots_100_2qb': '3 (1)',

                           'jun15_plots_1_2qb_moredata': '1 (1)',
                           'jun15_plots_100_2qb_moredata': '3 (1)',
                           }

                          count2qbit = True
                          splitting_title = 'data:CUSTOM_fine_splitting'

  elif k=="jul30_2qb_small_splitting": 
                          lbls = {
                           'jul30_2qb_small_splitting': '1 (1)'
                           }

                          count2qbit = True
                          steps_instead_of_method = True
                          splitting_title = 'data:CUSTOM_fine_splitting'

  elif k=="jul30_2qb_small_splitting_w_decay": 
                          lbls = {
                           'jul30_2qb_small_splitting_w_decay': '1 (1)'
                           }

                          count2qbit = True
                          steps_instead_of_method = True
                          splitting_title = 'data:CUSTOM_fine_splitting'

  d = {i: 0 for i in lbls.keys()}

  df = None

  for key in d.keys():
    df_temp = pd.read_csv('~/public_html/'+key+'/_all.csv', converters={"I": conversion_function, "Q": conversion_function})
    df_temp['mar_sim_type'] = [key]*len(df_temp['data'])
    df_temp['Optimization Type'] = [lbls[key].split(' (')[0]]*len(df_temp['data'])
    df_temp['Initial Conditions'] = [lbls[key].split(' (')[1].split(')')[0]]*len(df_temp['data'])
    # df_temp['Number of Oscillations'] = [ int((int(i[0])+1)/2) if i[0]!='p' else 1 for i in df_temp['Initial Conditions'] ]
    df_temp['Number of Oscillations'] = [ int(i) if i[0]!='p' else 1 for i in df_temp['Initial Conditions'] ]
    if df is None:
    	df = df_temp
    else:
    	df = pd.concat([df, df_temp])

  # ptot = 3
  # p2, p3, p4, p5, p6 = 1, 1, 1, 1, 1

  # if count2qbit == True:
  #   ptot = 1
  #   p2, p3, p4, p5, p6 = -1, 0, 0, 0, 0

  if count2qbit == False:
    df['mar12_loss'] = 1 - ( + ( df["final_pops"]  + df["final_pops2"] + df["final_pops3"] ) - (df["final_pops4"] + df["final_pops5"] + df["final_pops6"]) ) /3
  else:
    df['mar12_loss'] = 1 - df["final_pops"] + df["final_pops2"]

  df.to_csv('~/QPG-ChipDesign/hyperfine/csv_join/'+k+'.csv')

  # for aid, grp in df.groupby(['mar_sim_type']):

  # 	print(grp[['T', splitting_title, 'mar12_loss']])

  if k in {'mar17', 'mar31', 'jun15_simple'}:
    df = df.loc[df["data:T"] >= 0.2]

  if div_splitting_norm: df[splitting_title] = round(df[splitting_title]/(2 * np.pi), 2)

  df['Spin Splitting $\\Delta$ (MHz)'] = df[splitting_title].apply(lambda x: str(x))

  df['Loss'] = df['mar12_loss']

  df['pulse length [us]'] = df['T']

  df['mag'] = df["I"].apply(lambda x: np.array(x)**2) + df["Q"].apply(lambda x: np.array(x)**2) #np.sqrt(np.array()**2 + np.array(df["Q"])**2 )
  df['Maximum Amplitude'] = df['mag'].apply(lambda x: np.sqrt(np.amax(x)))
  df['log(Maximum Amplitude)'] = df['Maximum Amplitude'].apply(lambda x: np.log(x))
  df['Inverse of Maximum Amplitude [1/MHz]'] = df['Maximum Amplitude'].apply(lambda x: 1/x)

  df['Average Amplitude'] = df['mag'].apply(lambda x: np.average(np.sqrt(x)))
  df['Inverse of Average Amplitude [1/MHz]'] = df['Average Amplitude'].apply(lambda x: 1/x)

  with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df[['mar_sim_type', 'STEPS', 'Initial Conditions', 'Number of Oscillations', 'mar12_loss']])#, 'data:T', 'Maximum Amplitude'

  col_order = [str(i) for i in sorted(df[splitting_title].unique())]

  sb.color_palette("tab10")
  sb.axes_style("whitegrid")

  sb.set(rc={'figure.figsize':(11.7,8.27)})

  if steps_instead_of_method:
    sb.relplot(
      data=df.loc[df["Number of Oscillations"] == 1],
      x="pulse length [us]", y="Loss", col="Spin Splitting $\\Delta$ (MHz)",
      # style="Number of Oscillations", 
      hue="STEPS",# size="size",
      # kind="line"
      col_order=col_order,
      style="STEPS", #style_order=['Square pulse', '100 step', '1000 step'],
      sizes=(2., 2.), size="Optimization Type",
      kind="line",# s=120
    )

    plt.gcf().subplots_adjust(bottom=0.15)
    sb.set(rc={'figure.figsize':(11.7,8.27)})
    plt.xlabel(r'pulse length [$\mu$s]')
    plt.gca().set(xlabel=r'pulse length [$\mu$s]')
    plt.ylim(-0.01, 1.01)

    plt.savefig('/home/tareq/public_html/manual/'+k+'_steps_instead_of_method.png')

    raise NotImplementedError

  # sb.relplot(
  #   data=df,
  #   x='pulse length [us]', y="Loss",#, col="time",
  #   style="Spin Splitting $\\Delta$ (MHz)", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=50,
  # )

  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  # plt.gcf().subplots_adjust(bottom=0.15)

  # t = sb.lineplot(
  #   data=df,
  #   x='pulse length [us]', y="Loss",#, col="time",
  #   style="Spin Splitting $\\Delta$ (MHz)", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   alpha=0.4
  #   )
  # t.get_legend().remove()
  # plt.savefig('/home/tareq/public_html/manual/'+k+'.png')

  # sb.relplot(
  #   data=df,
  #   x="pulse length [us]", y="Loss", col="Spin Splitting $\\Delta$ (MHz)",
  #   style="Number of Oscillations", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=120,
  # )

  # plt.gcf().subplots_adjust(bottom=0.15)
  # sb.set(rc={'figure.figsize':(11.7,8.27)})
  # plt.xlabel(r'pulse length [$\mu$s]')
  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')

  # plt.savefig('/home/tareq/public_html/manual/'+k+'_alt.png')

  # sb.relplot(
  #   data=df.loc[df["Number of Oscillations"] == 1],
  #   x="pulse length [us]", y="Loss", col="Spin Splitting $\\Delta$ (MHz)",
  #   # style="Number of Oscillations", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=120,
  # )

  # plt.gcf().subplots_adjust(bottom=0.15)
  # sb.set(rc={'figure.figsize':(11.7,8.27)})
  # plt.xlabel(r'pulse length [$\mu$s]')
  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')

  # plt.savefig('/home/tareq/public_html/manual/'+k+'_alt_lp.png')

  sb.relplot(
    data=df.loc[df["Number of Oscillations"] == 1],
    x="pulse length [us]", y="Loss", col="Spin Splitting $\\Delta$ (MHz)",
    # style="Number of Oscillations", 
    hue="Optimization Type",# size="size",
    # kind="line"
    col_order=col_order,
    style="Optimization Type", style_order=['Optimized with regularization', 'Square pulse'],
    sizes=(2., 2.), size="Optimization Type",
    kind="line",# s=120
  )

  plt.gcf().subplots_adjust(bottom=0.15)
  sb.set(rc={'figure.figsize':(11.7,8.27)})
  plt.xlabel(r'pulse length [$\mu$s]')
  plt.gca().set(xlabel=r'pulse length [$\mu$s]')

  plt.savefig('/home/tareq/public_html/manual/'+k+'_alt_lp_pub.png')

  # sb.relplot(
  #   data=df,
  #   x="pulse length [us]", y="Maximum Amplitude", col="Spin Splitting $\\Delta$ (MHz)",
  #   style="Number of Oscillations", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=120,
  # )

  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  # plt.gcf().subplots_adjust(bottom=0.15)

  # plt.savefig('/home/tareq/public_html/manual/'+k+'_power.png')

  # if k=='mar17':

  #   sb.relplot(
  #     data=df[df["Spin Splitting $\\Delta$ (MHz)"] == '5.0'],
  #     x="pulse length [us]", y="Inverse of Maximum Amplitude [1/MHz]", col="Spin Splitting $\\Delta$ (MHz)",
  #     style="Number of Oscillations", 
  #     hue="Optimization Type",# size="size",
  #     # kind="line"
  #     col_order=col_order,
  #     s=120,
  #   )

  #   plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  #   plt.gcf().subplots_adjust(bottom=0.15)

  #   plt.savefig('/home/tareq/public_html/manual/'+k+'_inv_power_5only.png')

  #   sb.relplot(
  #     data=df[df["Spin Splitting $\\Delta$ (MHz)"] == '5.0'],
  #     x="pulse length [us]", y="Inverse of Average Amplitude [1/MHz]", col="Spin Splitting $\\Delta$ (MHz)",
  #     style="Number of Oscillations", 
  #     hue="Optimization Type",# size="size",
  #     # kind="line"
  #     col_order=col_order,
  #     s=120,
  #   )

  #   plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  #   plt.gcf().subplots_adjust(bottom=0.15)

  #   plt.savefig('/home/tareq/public_html/manual/'+k+'_inv_avg_power_5only.png')

  # sb.relplot(
  #   data=df,
  #   x="pulse length [us]", y="Average Amplitude", col="Spin Splitting $\\Delta$ (MHz)",
  #   style="Number of Oscillations", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=120,
  # )

  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  # plt.gcf().subplots_adjust(bottom=0.15)

  # plt.savefig('/home/tareq/public_html/manual/'+k+'_avg_power.png')

  # sb.relplot(
  #   data=df,
  #   x="pulse length [us]", y="log(Maximum Amplitude)", col="Spin Splitting $\\Delta$ (MHz)",
  #   style="Number of Oscillations", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=120,
  # )

  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  # plt.gcf().subplots_adjust(bottom=0.15)

  # plt.savefig('/home/tareq/public_html/manual/'+k+'_log_power.png')

  # sb.relplot(
  #   data=df,
  #   x="pulse length [us]", y="Inverse of Maximum Amplitude [1/MHz]", col="Spin Splitting $\\Delta$ (MHz)",
  #   style="Number of Oscillations", 
  #   hue="Optimization Type",# size="size",
  #   # kind="line"
  #   col_order=col_order,
  #   s=120,
  # )

  # plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  # plt.gcf().subplots_adjust(bottom=0.15)

  # plt.savefig('/home/tareq/public_html/manual/'+k+'_inv_power.png')

  # # improvement plots

  # if k=='mar31':

  #   # - df.loc[df['Optimization Type'] == 'mar31_type_1' and df['Initial Conditions'] == row['Initial Conditions'] and df[splitting_title]==row[splitting_title] and df['data:T']==row['data:T'] ]

  #   df['Relative Loss'] = df.apply(lambda row: row['mar12_loss'] - df.loc[
  #               (df['Optimization Type'] == '1') & 
  #               (df['Number of Oscillations'] == '1') & 
  #               (df[splitting_title]==row[splitting_title]) & 
  #               (df['data:T']==row['data:T']) 
  #               ]['mar12_loss'].iloc[0], axis=1)

  #   sb.relplot(
  #     data=df,
  #     x="pulse length [us]", y="Relative Loss", col="Spin Splitting $\\Delta$ (MHz)",
  #     style="Number of Oscillations", 
  #     hue="Optimization Type",# size="size",
  #     # kind="line"
  #     col_order=col_order,
  #     s=120,
  #   )

  #   plt.gca().set(xlabel=r'pulse length [$\mu$s]')
  #   plt.gcf().subplots_adjust(bottom=0.15)

  #   plt.savefig('/home/tareq/public_html/manual/'+k+'_rel_loss.png')



