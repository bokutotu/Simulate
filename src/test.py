import leapfrog
import preprocess

import unittest

import numpy as np
MASS = {'CA': 12.01100, 'CB': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}


class TestCalMethod(unittest.TestCase):

    def test_cal_coord(self):
        """cal_coord関数のテストを行う"""
        c = np.load("../test/c_test.npy")
        v = np.load("../test/v_test.npy")

        atom_num = c.shape[1]

        cal_c_1 = leapfrog.cal_coord(c[0], v[0])
        diff = cal_c_1 - c[1]
        diff_abs = np.mean(np.abs(diff))
        self.assertTrue(diff_abs < 5e-4)

    def test_cal_v_2(self):
        """cal_v_2のテストを行う"""
        v = np.load("../test/v_test.npy")
        f = np.load("../test/f_test.npy")

        atom_num = v.shape[1]

        mass = []
        for idx in range(atom_num):
            if idx % 4 == 0:
                mass.append(MASS["N"])
            elif idx % 4 == 1 or idx % 4 == 2:
                mass.append(MASS["C"])
            else:
                mass.append(MASS["O"])
        mass = np.array(mass).reshape(-1,1)
        mass = np.concatenate([mass, mass, mass], axis=-1)

        cal_v_1 = leapfrog.cal_v_2(v[0], 0.002, mass, f[0])

        diff = cal_v_1 - v[1]
        diff_abs = np.mean(np.abs(diff))

        self.assertTrue(diff_abs < 1e-4)


if __name__ == "__main__":
    unittest.main()
