# KRNO
Code repository for the ICML 2025 paper 'Shifting time: time-series forecasting with Khatri-Rao neural operators'

- The scripts used for training and post-processing all the datasets are provided in the folder `code/scripts`.

## Scripts Details

### Spatial and Spatio-temporal problems
- `code/scripts/darcy-flow/run_all.sh` contains the commands for training and testing using KRNO on Darcy flow dataset.
- In the case of **Hyper-Elastic**, **Shallow water**, and **Climate modeling** problems, `train_script.py` in the corresponding folders is used to train KRNO. Testing is done using the script `post_process.py`.

### Temporal forecasting problems
-  Scripts `code/scripts/mujoco/run_all${i}.sh` are used for training KRNO on all MuJoCo datasets. We used the training pipeline and evaluation metric implemented by Oh, Y. et al. [1]  ([https://github.com/yongkyung-oh/Stable-Neural-SDEs](https://github.com/yongkyung-oh/Stable-Neural-SDEs)) 
- Scripts to train KRNO on MIMIC, USHCN, and Human Activity datasets are given in `code/scripts/irregular_time_series/krno`. We used the training pipeline and evaluation merics implemented  by Zhang, Weijia, et al. [2] ([https://github.com/usail-hkust/t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN)). Instructions to download these datasets are also provided in their github page.  
- Training scripts for spiral trajectory (short and long) are provided in `code/scripts/spiral`. 
- `code/scripts/darts/run_darts_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **Darts** datasets. Final testing results for each dataset is written to the test file 'outputs/KRNO/*<dataset*>/final_test_results.txt' along with intermediate results from hyperparameter studies. 
- `code/scripts/m4_crypto_traj/run_crypto_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **Crypto** test cases.
- `code/scripts/m4_crypto_traj/run_traj_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **Player Trajectory** datasets.  
- `code/scripts/m4_crypto_traj/run_m4_KRNO.sh` contains the commands for hyperparameter tuning and testing using KRNO on all the **M4** test cases.  

[1] Oh, Y., Lim, D., & Kim, S. (2024, May). Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data. In The Twelfth International Conference on Learning Representations

[2] Zhang, Weijia, et al. "Irregular multivariate time series forecasting: A transformable patching graph neural networks approach." ICML 2024.

## Datasets

- The datasets for the Darcy, Hyper-elasticity, and climate modeling are already included in the corresponding scripts folders. 
- Dataset for shallow water problem can be downloaded from [LOCA github page](https://github.com/PredictiveIntelligenceLab/LOCA?tab=readme-ov-file). Copy the downloaded npz files to the folder `code/scripts/shallow_water_<model>/loca-data/`.
- Darts datasets are included in the folder `code/dataset/darts`
- Instructions to download the datasets for M4, Crypto and Player Trajectory are given in `code/dataset/ReadMe.md`. 





## Installing dependencies

`conda create --name KRNO_env python=3.10`

`conda install numpy==1.26.2 scipy matplotlib seaborn h5py`

`conda install -c conda-forge pandas scikit-learn patool tqdm sktime wandb cartopy`

`conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia`

`pip install neuraloperator==0.3.0 torch-harmonics==0.6.5`


## Installing KRNO

`cd KRNO/code/kno`

`pip install .`





