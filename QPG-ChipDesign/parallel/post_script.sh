#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tareq/public_html/logs/%A_%a.out
#SBATCH --error=/home/tareq/public_html/logs/%A_%a.err
#S BATCH --mail-type=ALL
#S BATCH --mail-user=tareq@mit.edu

module load engaging/python/3.6.0

python3 post.py ~/public_html/test_alpha_smooth_vs_fft_window t varx=T2 invx fft
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
