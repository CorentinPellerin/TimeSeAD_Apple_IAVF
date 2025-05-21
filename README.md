## Getting started

For installation and usage guides please refer to the [documentation](https://timesead.readthedocs.io/en/latest).

For example, assuming TimeSeAD is located directly under root:

    
    cd ~/TimeSeAD/
    conda env create --file setup/conda_env_cuda.yaml
    conda activate TimeSeAD
    pip install -e . 
    pip install -e .[experiments] 
    pip install sympy

## Running Experiments

Run:

    source $HOME/TimeSeAD/experiment_configs/apple/experiment_manager/prepare_tsad.sh
    export CUDA_VISIBLE_DEVICES=00
    bash $HOME/TimeSeAD/experiment_configs/apple/experiment_manager/list_runs.sh

This will echo a list of experiments with each line being a single python command. 
For example:

    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_eif.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_hbos.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_iforest.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_iqr_ad.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_kmeans.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_knn.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_kpca_ad.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_oos_ad.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_pca_ad.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_wmd_ad.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/baselines/train_wmd_fix_ad.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/gan/train_beatgan_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/gan/train_lstm_vae_gan_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/gan/train_madgan_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/gan/train_tadgan_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/other/train_lstm_ae_ocsvm_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/other/train_mtad_gat_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/other/train_ncad_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/other/train_thoc_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/prediction/train_gdn_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/prediction/train_lstm_prediction_filonov_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/prediction/train_lstm_prediction_malhotra_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/prediction/train_tcn_prediction_he_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/prediction/train_tcn_prediction_munir_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_anomtransf_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_autoformer.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_dense_ae_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_fedformer.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_genad_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_lstm_ae_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_lstm_max_ae_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_mscred_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_stgat_mad_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_tcn_ae_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_untrained_lstm_ae_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/reconstruction/train_usad_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/vae/train_donut_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/vae/train_gmm_vae_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/vae/train_lstm_vae_park_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/vae/train_lstm_vae_soelch_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/vae/train_omni_anomaly_on_apple.yml training_param_updates.training.device=cuda seed=123  \
    python $HOME/TimeSeAD/timesead_experiments/grid_search.py with $HOME/TimeSeAD/experiment_configs/apple/vae/train_sis_vae_on_apple.yml training_param_updates.training.device=cuda seed=123 


Simply execute those commands to run the corresponding experiments. 
Results are per default logged to `$HOME/TimeSeAD/log`.

There is a script 

    python $HOME/TimeSeAD/experiment_configs/apple/experiment_manager/csv_to_plot.py

that can be used to collect the logged results to a csv file.

Note that not all custom experiments are yet functional. 
