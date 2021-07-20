import numpy as np

MASS = {'CA': 12.01100, 'CB': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}


def cal_half_velocities(c_now, c_preb, dt=0.002):
    return (c_now - c_preb) / dt


def get_mass_list(atom_num):
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
    return mass


def cal_velocities(v, force, mass, dt=0.002):
    return v + dt / mass * force


def main():
    c = np.load("c_min_data")
    f = np.load("f_min_data")

    mass = get_mass_list(c.shape[1])

    v_1_2 = cal_half_velocities(c[1], c[0])
    v_3_2 = cal_half_velocities(c[2], c[1])

    cal_v_3_2 = cal_velocities(v_1_2, f[1], mass)

    print(np.mean(np.abs(v_3_2)))
    print( np.mean(np.abs(v_3_2 - cal_v_3_2)))


if __name__ == "__main__":
    main()
