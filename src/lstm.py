import numpy as np
import torch

import leapfrog
import preprocess


class LSTMLeapFrog(leapfrog.LeapFrog):

    def __init__(self, sim_len, atom_num, norm,
            chainlen_c, floatlen_c, chainlen_v, floatlen_v,
            coord, velocity, force,
            net_n, net_ca, net_c, net_o,
            feature_len, name, in_channels, in_channels_o):
        super().__init__(sim_len, atom_num, norm,
                coord, velocity, force,
                net_n, net_ca, net_c, net_o)
        self.feature_len = feature_len
        self.name = name

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.in_channels = in_channels
        self.in_channels_o = in_channels_o

        self.chainlen_c = chainlen_c
        self.floatlen_c = floatlen_c
        self.chainlen_v = chainlen_v
        self.floatlen_v = floatlen_v

    def before_sim(self):
        self.res_coord = np.zeros((self.feature_len+self.sim_len, self.atom_num, 3), dtype=np.float32)
        self.res_velocity = np.zeros((self.feature_len+self.sim_len, self.atom_num, 3), dtype=np.float32)
        self.res_force = np.zeros((self.feature_len+self.sim_len, self.atom_num, 3), dtype=np.float32)

        self.res_coord[0:self.feature_len] = self.coord
        self.res_velocity[0:self.feature_len] = self.velocity
        self.res_force[0:self.feature_len] = self.force

        self.features_ca = np.zeros(
                (self.sim_len + self.feature_len, self.atom_num//4+1, self.in_channels),
                dtype=np.float32)
        self.features_c = np.zeros(
                (self.sim_len + self.feature_len, self.atom_num//4+1, self.in_channels),
                dtype=np.float32)
        self.features_n = np.zeros(
                (self.sim_len + self.feature_len, self.atom_num//4+1, self.in_channels),
                dtype=np.float32)
        self.features_o = np.zeros(
                (self.sim_len + self.feature_len, self.atom_num//4, self.in_channels_o),
                dtype=np.float32)

        # 一回目のために0~feature_len - 1 までの特徴量を作成
        for time in range(self.feature_len-1):
            f_n, f_ca, f_c, f_o, b_n, b_ca, b_c, b_o = \
                    preprocess.make_single(
                        chainlen_c=self.chainlen_c, floatlen_c=self.floatlen_c,
                        chainlen_v=self.chainlen_v, floatlen_v=self.floatlen_v,
                        atom_num=self.atom_num,
                        c=self.res_coord[time], v=self.res_velocity[time],
                        is_use_angles=True
                    )
            self.features_n[time] =  f_n
            self.features_ca[time] = f_ca
            self.features_c[time] =  f_c
            self.features_o[time] =  f_o

    def simulation_step(self, time):
        # time - 1 の特徴量の計算と代入
        f_n, f_ca, f_c, f_o, b_n, b_ca, b_c, b_o = \
                preprocess.make_single(
                    chainlen_c=self.chainlen_c, floatlen_c=self.floatlen_c,
                    chainlen_v=self.chainlen_v, floatlen_v=self.floatlen_v,
                    atom_num=self.atom_num,
                    c=self.res_coord[time+self.feature_len-1],
                    v=self.res_velocity[time+self.feature_len-1],
                    is_use_angles=True
                )

        self.features_n[time+self.feature_len-1] =  f_n
        self.features_ca[time+self.feature_len-1] = f_ca
        self.features_c[time+self.feature_len-1] =  f_c
        self.features_o[time+self.feature_len-1] =  f_o

        # 入力に必要な特徴量の切り出し
        input_tensor_ca = torch.tensor(self.features_ca[time:time+self.feature_len]) \
            .to(self.device)
        input_tensor_c = torch.tensor(self.features_c[time:time+self.feature_len]) \
            .to(self.device)
        input_tensor_n = torch.tensor(self.features_n[time:time+self.feature_len]) \
            .to(self.device)
        input_tensor_o = torch.tensor(self.features_o[time:time+self.feature_len]) \
            .to(self.device)

        # ニューラルネットが学習する際の次元に変更する
        # (features_len, atom_num, in_channnels) -> (atom_num, features_len, in_channnels)
        input_tensor_c = input_tensor_c.transpose(0,1)
        input_tensor_ca = input_tensor_ca.transpose(0,1)
        input_tensor_n = input_tensor_n.transpose(0,1)
        input_tensor_o = input_tensor_o.transpose(0,1)

        # ニューラルネットで予測
        force_n, force_ca, force_c, force_o = \
                self.pred_nn(input_tensor_n, input_tensor_ca, input_tensor_c, input_tensor_o)

        # 使用するのは一番最後に予測されたものを使用する
        force_ca = force_ca[::, -1, ::]
        force_c = force_c[::, -1, ::]
        force_n = force_n[::, -1, ::]
        force_o = force_o[::, -1, ::]

        force = leapfrog.rotate_force(force_n,force_ca, force_c, force_o,
                b_n, b_ca, b_c, b_o, self.atom_num, self.norm)

        # 速度を計算する
        v_now = leapfrog.cal_v_2(self.res_velocity[time+self.feature_len -1], self.mass, force)
        self.res_velocity[time+self.feature_len] = v_now

        # 座標を計算
        c_now = leapfrog.cal_coord(self.res_coord[time, self.feature_len - 1], v_now)
        self.res_coord[time + self.feature_len] = c_now


    def save(self):
        np.save(self.name, self.res_coord[self.feature_len:-1:])
