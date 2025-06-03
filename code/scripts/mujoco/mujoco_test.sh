# nohup python -u mujoco-sde.py  --lr 0.001  --missing_rate 0.0 --time_seq 50 --y_seq 10 --epoch 200 --step_mode 'valloss' --model krno > nohup_missrate_0_elu.out &

# nohup python3 -u mujoco-sde.py  --lr 0.001  --missing_rate 0.3 --time_seq 50 --y_seq 10 --epoch 200 --step_mode 'valloss' --model krno > nohup_missrate_30_elu.out &

# nohup python3 -u mujoco-sde.py  --lr 0.001  --missing_rate 0.5 --time_seq 50 --y_seq 10 --epoch 200 --step_mode 'valloss' --model krno > nohup_missrate_50_elu.out &

# nohup python3 -u mujoco-sde.py  --lr 0.001  --missing_rate 0.7 --time_seq 50 --y_seq 10 --epoch 200 --step_mode 'valloss' --model krno > nohup_missrate_70_elu.out &



nohup python3 -u mujoco-sde.py  --lr 0.001  --missing_rate 0.0 --time_seq 50 --y_seq 10 --epoch 200 \
         --step_mode 'valloss' --model neurallsde \
         --layers 2 --h_channels 128 --hh_channels 128 > nohup_missrate_0_neurallsde_bs_128_test_time.out &