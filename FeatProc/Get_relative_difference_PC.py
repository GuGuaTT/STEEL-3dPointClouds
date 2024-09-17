import numpy as np
import h5py
import matplotlib.pyplot as plt


def calculate_area(vertex1, vertex2, vertex3):

    # Lengths of triangle mesh
    length1 = np.linalg.norm(vertex1 - vertex2)
    length2 = np.linalg.norm(vertex2 - vertex3)
    length3 = np.linalg.norm(vertex3 - vertex1)

    # Heron's formula
    p = (length1 + length2 + length3) / 2
    area_ = (p * (p - length1) * (p - length2) * (p - length3)) ** 0.5
    return area_


def write_group(upper_group, new_group):
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
                        model_ = h5_file[i1][i2][i3][i4][i5]
                        print(model_.name)

                        pts_all = model_['Deformed_shape']['processed_center'][:]
                        pts_list = []
                        for m in pts_all.shape[0]:
                            pts = pts_all[m, 0, ...].reshape(21, 21, 3)

                            fl0, fl1, wb = pts[:, 0:6, :],  pts[:, 15:, :], pts[:, 6:15, :]
                            bot_area = (np.linalg.norm(fl0[0, 0, :] - fl0[0, -1, :]) +
                                        np.linalg.norm(wb[0, 0, :] - wb[0, -1, :]) +
                                        np.linalg.norm(fl1[0, 0, :] - fl1[0, -1, :]))

                            surf_area = 0
                            for k in [fl0, fl1, wb]:
                                for i in range(k.shape[1] - 1):
                                    for j in range(k.shape[0] - 1):
                                        pt1, pt2, pt3, pt4 = k[j, i], k[j, i + 1], k[j + 1, i + 1], k[j + 1, i]
                                        area1, area2 = calculate_area(pt1, pt2, pt3), calculate_area(pt2, pt3, pt4)
                                        surf_area = surf_area + area1 + area2

                            total_height = surf_area / bot_area

                            pts_u = np.copy(pts)
                            pts_u[1:, :, :2] = pts[0, :, :2]
                            pts_u = pts_u.reshape(-1, 3)
                            heights = np.array([total_height / 20 * i for i in range(21)])
                            pts_u[:, 2] = np.concatenate([np.repeat(height, 21) for height in heights])

                            pts = pts.reshape(-1, 3)
                            pts_d = pts_u - pts
                            pts_list.append(pts_d)

                        pts_array = np.array(pts_list)
                        group_ds = write_group(model_, 'Deformed_shape')
                        if "relative_displacement" in group_ds.keys():
                            del model_["relative_displacement"]
                        group_ds.create_dataset('relative_displacement', data=pts_array)
