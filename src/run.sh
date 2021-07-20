python simulate.py  \
        --atom_num 79 \
        --ca_path /home/bokutotu/HDD/Lab/outputs/CA --c_path /home/bokutotu/HDD/Lab/outputs/C \
        --n_path /home/bokutotu/HDD/Lab/outputs/N --o_path /home/bokutotu/HDD/Lab/outputs/O \
        --chainlen_c 20 --floatlen_c 20 --chainlen_v 20 --floatlen_v 20 \
        --is_use_angles true --sim_len 250 \
        --prev_c_path ~/HDD/Lab/NPY/c_trj.npy \
        --prev_v_path ~/HDD/Lab/NPY/v_trj.npy \
        --prev_f_path ~/HDD/Lab/NPY/f_trj.npy \
        --norm 6000 --name res_lstm.npy
