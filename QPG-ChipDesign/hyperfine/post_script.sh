#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tareq/public_html/logs/%A_%a.out
#SBATCH --error=/home/tareq/public_html/logs/%A_%a.err
#S BATCH --mail-type=ALL
#S BATCH --mail-user=tareq@mit.edu

module load engaging/python/3.6.0

python3 post.py ~/public_html/aug_11_F_plot_longer t varx=steps vary=CUSTOM_fine_splitting no_loss no_evo
# python3 post.py ~/public_html/aug_11_F_plot t varx=steps vary=CUSTOM_fine_splitting no_loss no_evo
# python3 post.py ~/public_html/aug_6_F_plot_otherqb t varx=steps vary=CUSTOM_fine_splitting no_loss no_evo
# python3 post.py ~/public_html/aug_6_F_plot t varx=steps vary=CUSTOM_fine_splitting no_loss no_evo
# python3 post.py ~/public_html/aug_5_F_plot t varx=steps vary=CUSTOM_fine_splitting no_loss no_evo
# python3 post.py ~/public_html/jul30_2qb_small_splitting_w_decay t varx=steps vary=CUSTOM_fine_splitting no_evo
# python3 post.py ~/public_html/jul30_2qb_small_splitting t varx=steps vary=CUSTOM_fine_splitting no_evo

# python3 post.py ~/public_html/jul8_sweep_long_omega_100_sq t varx=CUSTOM_splitting vary=CUSTOM_splitting no_evo
# python3 post.py ~/public_html/jul8_sweep_long_omega_100 t varx=CUSTOM_splitting vary=CUSTOM_splitting no_evo
# python3 post.py ~/public_html/jul8_sweep_omega_100 t varx=CUSTOM_splitting vary=CUSTOM_splitting no_evo
# python3 post.py ~/public_html/jul8_sweep_omega t varx=CUSTOM_splitting vary=CUSTOM_splitting no_evo

# python3 post.py ~/public_html/jun15_plots_100_2qb_moredata t varx=T vary=CUSTOM_splitting no_evo no_loss
# python3 post.py ~/public_html/jun15_plots_1_2qb_moredata t varx=T vary=CUSTOM_splitting no_evo no_loss

# python3 post.py ~/public_html/jun10_plots_100step_1.1 t varx=T vary=omega_drive
# python3 post.py ~/public_html/jun10_plots_1step_1.1 t varx=T vary=omega_drive
# python3 post.py ~/public_html/jun10_plots_100step_3.3 t varx=T vary=omega_drive
# python3 post.py ~/public_html/jun10_plots_1step_3.3 t varx=T vary=omega_drive

# python3 post.py ~/public_html/jun10_plots_1step t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_1step_3pi t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_1step_5pi t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_1step_shro t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_1step_3pi_shro t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_1step_5pi_shro t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_guess10_lr5_shro t varx=T vary=omega_drive
# python3 post.py ~/public_html/jun10_plots_guess10_lr5 t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/jun10_plots_guess0_lr5 t varx=T vary=omega_drive no_evo
# python3 post.py ~/public_html/new_n_step_odmr_test t varx=T vary=omega_drive
# python3 post.py ~/public_html/new_1_step_odmr_test t varx=T vary=omega_drive
# python3 post.py ~/public_html/new_1_step_odmr_non_analytical t varx=steps vary=LR
# python3 post.py ~/public_html/new_1_step_odmr t varx=steps vary=LR

# python3 post.py ~/public_html/tests_may30 t varx=steps vary=LR
# python3 post.py ~/public_html/test_old_pulse_odmr t varx=steps vary=T
# python3 post.py ~/public_html/tests_500step_may_27 t varx=steps vary=T
# python3 post.py ~/public_html/tests_1step_may_27 t varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/tests_may_27 t varx=CUSTOM_splitting vary=T

# python3 post.py ~/public_html/redo2_mar31_type_3 t varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/redo_mar31_type_3 t varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_3 t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_2 t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_2_2osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_2_3osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_1 t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_1_2osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_1_3osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar31_type_3_test t no_evo no_loss varx=CUSTOM_splitting vary=T

# python3 post.py ~/public_html/mar17_specialtype5_24 t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype3_24 t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype2_24 t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype2_24_2osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype2_24_3osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype1_24 t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype1_24_2osc t no_evo no_loss varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_specialtype1_24_3osc t no_evo no_loss varx=CUSTOM_splitting vary=T

# python3 post.py ~/public_html/mar17_type3_24 t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type2_24_3osc t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type2_24_2osc t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type2_24_1osc t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type1_24_3osc t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type1_24_2osc t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type1_24 t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/unit_tests t varx=post_data:I vary=T

# python3 post.py ~/public_html/mar17_type1_t2 t varx=CUSTOM_splitting vary=T

# python3 post.py ~/public_html/mar17_type2_alt t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type1_alt t no_evo varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/mar17_type3 t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar17_type2 t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar17_type1 t no_evo varx=CUSTOM_splitting vary=T

# python3 post.py ~/public_html/mar12_type2_alt t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type1_alt t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type3_alt t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type4_bigLR t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type1 t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type2 t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type3 t no_evo vary=CUSTOM_splitting varx=T
# python3 post.py ~/public_html/mar12_type4 t no_evo vary=CUSTOM_splitting varx=T

# python3 post.py ~/public_html/units t

# python3 post.py ~/public_html/changing_omega_drive t varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/high_res_changing_omega_drive t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/2_high_res_changing_omega_drive t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/3_high_res_changing_omega_drive t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/cnst_IQ_high_res_changing_omega_drive t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/2_cnst_IQ_high_res_changing_omega_drive t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/overkill_cnst_IQ_high_res_changing_omega_drive t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/single_shot_post_B t varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/post_B_vary_omega t no_evo varx=omega_drive vary=omega_drive
# python3 post.py ~/public_html/post_B_vary_omega_high_res t no_evo varx=omega_drive vary=omega_drive

# python3 post.py ~/public_html/00_hpf t varx=omega_drive
# python3 post.py ~/public_html/01_hpf t varx=omega_drive
# python3 post.py ~/public_html/02_hpf t varx=omega_drive
# python3 post.py ~/public_html/03_hpf t varx=omega_drive
# python3 post.py ~/public_html/04_hpf t varx=omega_drive
# python3 post.py ~/public_html/01_hpf_delta t varx=CUSTOM_splitting vary=T
# python3 post.py ~/public_html/01_hpf_delta_1step t varx=CUSTOM_splitting vary=T

# python3 post.py ~/public_html/test_alpha_smooth_vs_fft_window t varx=T2 invx fft
# python3 post.py ~/public_html/test_alpha_fft_window_2 t varx=T2 invx fft
# python3 post.py ~/public_html/test_alpha_fft_window t varx=T2 invx fft
# python3 post.py ~/public_html/test_alpha_plots_fft_2 t varx=T2 invx fft
# python3 post.py ~/public_html/test_alpha_plots_small_t t varx=T2 invx
# python3 post.py ~/public_html/test_alpha_plots_vanilla t varx=T2 invx
# python3 post.py ~/public_html/test_alpha_plots_fft t varx=T2 invx fft
# python3 post.py ~/public_html/test_alpha_plots_reg_alt t varx=T2 invx
# python3 post.py ~/public_html/test_alpha_plots_reg t varx=T2 invx
# python3 post.py ~/public_html/alpha_plots_reg t varx=T2 invx
# python3 post.py ~/public_html/alpha_plots_old t varx=T2 invx
# python3 post.py ~/public_html/alpha_plots_old_2 t varx=T2 invx

# python3 post.py ~/public_html/alpha_plots_old t
# python3 post.py ~/public_html/alpha_plots t
# python3 post.py ~/public_html/test_purity t
# python3 post.py ~/public_html/test_reg t
# python3 post.py ~/public_html/run_w_guess t
# python3 post.py ~/public_html/test_guess t
# python3 post.py ~/public_html/3qbit_alpha_fft t fft
# python3 post.py ~/public_html/3qbit_S_fft t  fft
# python3 post.py ~/public_html/3qbit_alpha t
# python3 post.py ~/public_html/big_alpha_sim t
# python3 post.py ~/public_html/alpha_sim t
