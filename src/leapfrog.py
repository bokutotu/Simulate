import unittest

import numpy as np
import torch

from preprocess import basic, rotate_vec, rev_rotate_vec, make_single

MASS = {'CA': 12.01100, 'CB': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}
DT = 0.002


def cal_coord(now_coord, v_2, dt=0.002):
    """time step at nの座標を計算する
    Parameters
    ==========
    now_coord : numpy.array or torch.tensor
        現在の座標
    v_2 : numpy.array or torch.tensor
        time step at n + 1/2の速度
    step : int
        stepのサイズ
    """
    return now_coord + dt * v_2


def cal_v_2(v_2, m, f, dt=0.002):
    """time step at n + 1/2の速度を計算する
    Parameters
    ==========
    v_2 : numpy.array or torch.tensor
        time step at n - 1/2の速度
    step : int
        step サイズ
    m : float
        計算する原子の重さ
    f : numpy.array or torch.tesor
        原子にかかる力
    """
    return v_2 + f * dt / m


def cal_now_v(prev_v_2, forward_v_2):
    """time step at nの速度を求める
    Parameters
    =========
    prev_v_2 : numpy.array or torch.tensor
        time step at n - 1/2の速度
    forward_v_2 : numpy.array or torch.tensor
        time step at n + 1/2の速度
    """
    return (prev_v_2 + forward_v_2) / 2


def cal_init_v_2(v_0, step, m, f):
    """最初の速度を計算する
    Parameters
    =========
    v_0 : numpy.array or torch.tensor
        time step at 0
    step : int
        step size
    m : float
        計算する原子の重さ
    f : numpy.array or torch.tensor
        time step at 0の力
    """
    return v_0 - f * h / (2*m)


def rotate_force(force_n, force_ca, force_c, force_o, b_n, b_ca, b_c, b_o, atom_num, norm):
    """
    ニューラルネットワークで出力した力を記述子で回転する前の座標系に戻す

    Parameters
    ==============
    b_n: torch.tensor
        N原子に対する基底ベクトル
    b_ca: torch.tensor
        Cα原子に対する基底ベクトル
    b_c: torch.tensor
        C原子に対する基底ベクトル
    b_o: torch.tensor
        O原子に対する基底ベクトル
    force_n : torch.tensor
        N原子の力（記述子の座標軸のまま）
    force_ca : torch.tensor
        Cα原子の力（記述子の座標軸のまま）
    force_c : torch.tensor
        C原子の力（記述子の座標軸のまま）
    force_o : torch.tensor
        O原子の力（記述子の座標軸のまま）
    """
    for atom_idx in range(atom_num//4+1):
        if atom_idx < len(force_o):
            force_o[atom_idx] = rev_rotate_vec(
                    force_o[atom_idx].reshape(1,3),
                    np.array([0., 0., 0.]).astype(np.float32), b_o[atom_idx])

        force_ca[atom_idx] = rev_rotate_vec(
                force_ca[atom_idx].reshape(1,3),
                np.array([0., 0., 0.]).astype(np.float32), b_ca[atom_idx])

        force_c[atom_idx] = rev_rotate_vec(
                force_c[atom_idx].reshape(1,3),
                np.array([0., 0., 0.]).astype(np.float32), b_c[atom_idx])

        force_n[atom_idx] = rev_rotate_vec(
                force_n[atom_idx].reshape(1,3),
                np.array([0., 0., 0.]).astype(np.float32), b_n[atom_idx])

    # ノーマライズした値を元の値に戻す
    force = np.zeros((atom_num, 3))
    force[0::4] = force_n
    force[1::4] = force_ca
    force[2::4] = force_c
    force[3::4] = force_o
    force = force * norm
    return force

class LeapFrog:
    """LeapFrogを行う基底クラス

    """

    def __init__(self, sim_len, atom_num, norm,
            coord, velocity, force, 
            net_n, net_ca, net_c, net_o):

        """

        Parameters
        ===========
        sim_len: int
            シミュレーションを行うステップの数
        num_atom: int
            シミュレーションを行う系の原子の数
        coord: numpy.array
            シミュレーションを行う際に必要な初期構造
            (feature_len. atom_num, 3)
        norm: int or float
            力を計算する際のノーマライズする数
        velocity: numpy.array
            シミュレーションを行う際に必要な初期速度
            (feature_len. atom_num, 3)
        net_ca: torch.nn.Module
            Cα原子の力を予測するNN
        net_c: torch.nn.Module
            C原子の力を予測するNN
        net_n: torch.nn.Module
            N原子の力を予測するNN
        net_o: torch.nn.Module
            O原子の力を予測するNN
        floatlen_c: int
            座標に対する特徴量生成のためのfloat atomの数
        chainlen_c: int
            座標に他する特徴量生成のためのchain atom の数
        floatlen_v: int
            速度に対する特徴量生成のためのfloat atomの数
        chainlen_v: int
            速度に他する特徴量生成のためのchain atom の数
        """
        # 初期構造を定義する
        self.atom_num = atom_num
        self.sim_len = sim_len
        self.coord = coord
        self.velocity = velocity
        self.force = force

        # 力から速度を計算する際に必要な原子ごとの重さを表すarray
        # このモデルでは、C,Cα,N,Oのみを用いるためこの順番で原子の重さが並んだ
        # (atom_num, 3)のarrayとなる
        mass = []
        for idx in range(self.atom_num):
            if idx % 4 == 0:
                mass.append(MASS["N"])
            elif idx % 4 == 1 or idx % 4 == 2:
                mass.append(MASS["C"])
            else:
                mass.append(MASS["O"])
        mass = np.array(mass).reshape(-1,1)
        self.mass = np.concatenate([mass, mass, mass], axis=-1)

        self.norm = float(norm)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net_ca = net_ca.to(self.device)
        self.net_c =  net_c.to(self.device)
        self.net_n =  net_n.to(self.device)
        self.net_o =  net_o.to(self.device)

    def pred_nn(self, input_tensor_n, input_tensor_ca, input_tensor_c, input_tensor_o):
        input_tensor_n = input_tensor_n.to(self.device)
        input_tensor_ca = input_tensor_ca.to(self.device)
        input_tensor_c = input_tensor_c.to(self.device)
        input_tensor_o = input_tensor_o.to(self.device)

        force_ca = self.net_ca(input_tensor_ca).detach().to("cpu").numpy()
        force_c = self.net_c(input_tensor_c).detach().to("cpu").numpy()
        force_n = self.net_n(input_tensor_n).detach().to("cpu").numpy()
        force_o = self.net_o(input_tensor_o).detach().to("cpu").numpy()
        return force_n, force_ca, force_c, force_o

    def simulation_step(self, time):
        pass

    def save(self):
        pass

    def before_sim(self):
        pass

    def simulate(self):
        """シミュレーションを行う"""
        self.before_sim()
        for time in range(self.sim_len):
            print("time @ {} is start".format(time+1))
            self.simulation_step(time)

        self.save()
