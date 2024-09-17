import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys


def visualize_point(pc):
    ax1 = plt.subplot(projection='3d')
    ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='grey', s=8, alpha=0.2)
    ax1.scatter(pc[0, 0], pc[0, 1], pc[0, 2], c='r', s=12)
    ax1.scatter(pc[1, 0], pc[1, 1], pc[1, 2], c='b', s=12)
    ax1.scatter(pc[-1, 0], pc[-1, 1], pc[-1, 2], c='g', s=12)
    plt.show()


def sphere_normalization(pc):
    centroid = np.mean(pc, axis=0)
    pc_c = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_c ** 2, axis=1)))
    pc_normalized = pc_c / m
    return pc_normalized


def cube_normalization(pc):
    m = max(np.max(pc[:, 0]) - np.min(pc[:, 0]),
            np.max(pc[:, 1]) - np.min(pc[:, 1]),
            np.max(pc[:, 2]) - np.min(pc[:, 2]))
    return m


def write_group(upper_group, new_group):
    """
    Create a new group if the group does not exist, or else open the existed group.
    :param upper_group: the upper group of new group.
    :param new_group: the target group.
    :return: the target group.
    """
    if new_group in upper_group.keys():
        return upper_group[new_group]
    else:
        return upper_group.create_group(new_group)


if __name__ == "__main__":

    h5_file = h5py.File('column.hdf5', "a")

    for i1 in h5_file.keys():
        for i2 in h5_file[i1].keys():
            for i3 in h5_file[i1][i2].keys():
                for i4 in h5_file[i1][i2][i3].keys():
                    for i5 in h5_file[i1][i2][i3][i4].keys():
                        model = h5_file[i1][i2][i3][i4][i5]
                        print(model.name)

                        pc_lcl = model['Deformed_shape']["processed_center"][:]
                        indices = model['indices'][:]
                        d = model.attrs['d']
                        b_f = model.attrs['b_f']
                        t_f = model.attrs['t_f']
                        t_w = model.attrs['t_w']

                        # Multiple scaling factor normalization
                        m_scl = np.array([1 / b_f, 1 / (d - t_f), 1 / d]).reshape((1, 1, 1, 3))

                        # Plate buckling normalization
                        b_scl = np.array(1 / (d * np.sqrt(d / t_w) * np.sqrt(b_f / 2 * t_f))).reshape((1, 1, 1, 1))

                        # Sphere and cube normalization
                        s_scls = np.ones((len(indices), 2))
                        for i in range(len(indices)):
                            for j in [0, 1]:

                                c_pc = pc_lcl[i, j, ...]
                                m1_ = sphere_normalization(c_pc)
                                s_scl = 1 / m1_
                                s_scls[i, j] = s_scl

                        s_scls = s_scls.reshape((-1, 2, 1, 1))
                        group_scale = write_group(model, 'Norm_factor')

                        if "multiple" in group_scale.keys():
                            del group_scale["mnorm"]
                        if "sphere" in group_scale.keys():
                            del group_scale["snorm"]
                        if "buckling" in group_scale.keys():
                            del group_scale["bnorm"]

                        group_scale.create_dataset('mnorm', data=m_scl)
                        group_scale.create_dataset('snorm', data=s_scls)
                        group_scale.create_dataset('bnorm', data=b_scl)
