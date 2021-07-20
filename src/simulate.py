import numpy as np
import torch

import argparse

from pes_train.load import load_from_checkpoint

from lstm import LSTMLeapFrog


def main(args):
    coord = np.load(args.prev_c_path)
    velocity = np.load(args.prev_v_path)
    force = np.load(args.prev_f_path)

    in_channels_o = ((args.chainlen_c * 2 + args.floatlen_c) + (args.chainlen_v * 2 + args.floatlen_v))*4
    in_channels = in_channels_o + 3 if args.is_use_angles else in_channels_o

    config, net_ca = load_from_checkpoint(in_channels, args.ca_path)
    config, net_c = load_from_checkpoint(in_channels, args.c_path)
    config, net_n = load_from_checkpoint(in_channels, args.n_path)
    config, net_o = load_from_checkpoint(in_channels_o, args.o_path)

    if config["model"]["mode"] == "LSTM":
        coord = coord[0:config["time"]]
        velocity = velocity[0:config["time"]]
        force = force[0:config["time"]]
        simulater = LSTMLeapFrog(args.sim_len, args.atom_num, args.norm, 
                args.chainlen_c, args.floatlen_c, args.chainlen_v, args.floatlen_v,
                coord, velocity, force,
                net_n, net_ca, net_c, net_o,
                config["time"], args.name, in_channels, in_channels_o
                )
        simulater.simulate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="md simulate using training forecaster")
    parser.add_argument("--atom_num", type=int, )

    parser.add_argument("--ca_path", type=str,)
    parser.add_argument("--c_path", type=str,)

    parser.add_argument("--n_path", type=str,)
    parser.add_argument("--o_path", type=str,)

    parser.add_argument("--chainlen_c", type=int,)
    parser.add_argument("--floatlen_c", type=int,)
    parser.add_argument("--chainlen_v", type=int,)
    parser.add_argument("--floatlen_v", type=int,)

    parser.add_argument("--is_use_angles", type=bool, default=True)

    parser.add_argument("--sim_len", type=int, default=10)

    parser.add_argument("--prev_c_path", type=str, )
    parser.add_argument("--prev_v_path", type=str, )
    parser.add_argument("--prev_f_path", type=str, )

    parser.add_argument("--norm", type=int, )
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    main(args)
