import numpy as np
import h5py


def sphere_normalization(pc):
    centroid = np.mean(pc, axis=0)
    pc_c = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_c ** 2, axis=1)))
    return m


def write_group(upper_group, new_group):
    if new_group in upper_group.keys():
        return upper_group[new_group]
    else:
        return upper_group.create_group(new_group)


if __name__ == "__main__":

    path = '/scratch/gu/MLDB/hdf5_folder/column.hdf5'
    h5_file = h5py.File(path, "a")

    for i1 in h5_file.keys():
        for i2 in h5_file[i1].keys():
            for i3 in h5_file[i1][i2].keys():
                for i4 in h5_file[i1][i2][i3].keys():
                    for i5 in h5_file[i1][i2][i3][i4].keys():
                        model = h5_file[i1][i2][i3][i4][i5]
                        print(model.name)

                        pc_lcl = model['Deformed_shape']['processed_center'][:]
                        indices = model['indices'][:]
                        d = model.attrs['d']
                        b_f = model.attrs['b_f']
                        t_f = model.attrs['t_f']
                        t_w = model.attrs['t_w']

                        # Multiple scaling factor normalization
                        m_scl = np.array([1 / b_f, 1 / (d - t_f), 1 / d]).reshape(1, 1, 1, 3)

                        # Plate buckling normalization
                        b_scl = np.array(1 / (d * np.sqrt(d / t_w) * np.sqrt(b_f / 2 * t_f))).reshape(1, 1, 1, 1)

                        # Sphere and cube normalization
                        s_scls = np.ones((len(indices), 2))
                        for i in range(len(indices)):
                            for j in [0, 1]:

                                c_pc = pc_lcl[i, j, ...]
                                # visualize_point(c_pc)

                                m1_ = sphere_normalization(c_pc)
                                s_scl = 1 / m1_
                                s_scls[i, j] = s_scl

                        s_scls = s_scls.reshape(-1, 2, 1, 1)

                        group_scale = write_group(model, 'Scale')

                        if "mnorm" in group_scale.keys():
                            del group_scale["mnorm"]
                        if "snorm" in group_scale.keys():
                            del group_scale["snorm"]
                        if "bnorm" in group_scale.keys():
                            del group_scale["bnorm"]

                        group_scale.create_dataset('mnorm', data=m_scl)
                        group_scale.create_dataset('snorm', data=s_scls)
                        group_scale.create_dataset('bnorm', data=b_scl)
