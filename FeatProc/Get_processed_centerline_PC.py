import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import sys


def write_group(upper_group, new_group):
    if new_group in upper_group.keys():
        return upper_group[new_group]
    else:
        return upper_group.create_group(new_group)


def buckle_check(bottom_pc, n_lf):
    bottom_pc = bottom_pc.reshape(-1, n_lf, 3)
    err_index = 0
    for ii_ in range(bottom_pc.shape[0]):
        for j in range(bottom_pc.shape[1] - 1):
            if bottom_pc[ii_, j, 2] >= bottom_pc[ii_, j + 1, 2]:
                err_index = 1
    return err_index


# z-node1 < z-interpolation < z-node2, linear lagrange interpolation
# Higher order interpolation was also tested, but they tend to have high interpolation error
def lagrange_interpolation(z, node1, node2):
    p1x, p2x = node1[0], node2[0]
    p1y, p2y = node1[1], node2[1]
    p1z, p2z = node1[2], node2[2]
    k2 = (z - p2z) / (p1z - p2z)
    k3 = (z - p1z) / (p2z - p1z)
    x = k2 * p1x + k3 * p2x
    y = k2 * p1y + k3 * p2y
    return [x, y]


# Extract feature points on each slice
def feature_point_extraction(node_array, interval_num):
    ls = [0]
    ls_acc = [0]
    for q in range(len(node_array) - 1):
        length = np.linalg.norm(node_array[q] - node_array[q + 1])
        ls.append(length)
        ls_acc.append(length + ls_acc[-1])
    total_length = ls_acc[-1]
    interval = total_length / interval_num
    ls_node = [[node_array[0, 0], node_array[0, 1]]]
    for j in range(1, interval_num):
        cint = interval * j
        for m in range(len(ls_acc) - 1):
            if ls_acc[m] <= cint < ls_acc[m + 1]:
                res = cint - ls_acc[m]
                ls_x = res / ls[m + 1] * (node_array[m + 1, 0] - node_array[m, 0]) + node_array[m, 0]
                ls_y = res / ls[m + 1] * (node_array[m + 1, 1] - node_array[m, 1]) + node_array[m, 1]
                ls_node.append([ls_x, ls_y])
                break
    ls_node.append([node_array[-1, 0], node_array[-1, 1]])
    return np.array(ls_node)


def floor(number, n):
    return math.floor(number * 10 ** n) / 10 ** n


# Get slice of center-line or profile node
def get_interpolation(pc_, L_s_):
    ls_ = []
    for i_ in range(len(pc_) - 1):
        if pc_[i_, 2] < L_s_ < pc_[i_ + 1, 2]:
            p = lagrange_interpolation(L_s_, pc_[i_], pc_[i_ + 1])
            ls_.append(p)
        elif L_s_ == pc_[i_, 2]:
            p = [pc_[i_, 0], pc_[i_, 1]]
            ls_.append(p)
    return ls_


def rodrigues_rotation(v1, v2):

    c = np.dot(v1, v2)
    if c == -1.:
        v2 = -v2
        c = np.dot(v1, v2)

    n_vector = np.cross(v1, v2)
    n_vector_invert = np.array((
        [0, -n_vector[2], n_vector[1]],
        [n_vector[2], 0, -n_vector[0]],
        [-n_vector[1], n_vector[0], 0]))
    R_w2c = np.eye(3) + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
    return R_w2c


def visualize_point(pc):
    ax1 = plt.subplot(projection='3d')
    ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='grey', s=8, alpha=0.2)
    ax1.scatter(pc[0, 0], pc[0, 1], pc[0, 2], c='r', s=12)
    ax1.scatter(pc[1, 0], pc[1, 1], pc[1, 2], c='b', s=12)
    ax1.scatter(pc[-1, 0], pc[-1, 1], pc[-1, 2], c='g', s=12)
    plt.show()


def get_slice_matrix(model, bt_index, c_frame, ni_web, ni_flange, ni_slice, id_):

    # Get bottom/top end member segment's profile and centerline nodes
    p_pc = model['Solid_node']['profile_node_coord'][c_frame, bt_index]
    c_pc = model['Solid_node']['center_node_coord'][c_frame, bt_index]
    n_l, n_w, n_h, n_tw, n_tf, n_k = (model.attrs["n_l"], model.attrs["n_w"], model.attrs["n_h"],
                                      model.attrs["n_tw"], model.attrs["n_tf"], model.attrs["n_k"])
    d, w, t_f, t_w, r = (model.attrs["d"], model.attrs["b_f"], model.attrs["t_f"],
                         model.attrs["t_w"], model.attrs["r"])

    # Number of elements on the along the web
    web_num = n_w * 2 + n_tw + n_k

    # If top solid element segment
    if bt_index == 1:

        # Select top end nodes and add a new dimension for constant value
        top_nodes = p_pc[n_l * 2:: n_l * 2 + 1]
        top_nodes = np.concatenate((top_nodes, np.ones((top_nodes.shape[0], 1))), axis=1)

        # Get correlation matrix and then calculate top end plane vector
        corr_mat = np.transpose(top_nodes) @ top_nodes
        _, _, vt = np.linalg.svd(corr_mat)
        plane_vec1 = vt[-1][:-1] / np.linalg.norm(vt[-1][:-1])

        # Calculate rotation matrix for setting a new coordinate system
        R = rodrigues_rotation(np.array([0, 0, 1]), plane_vec1)
        pn_index = 1
        R_K = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Transformation of all the points
        p_pc = p_pc @ R
        c_pc = c_pc @ R

        # Get mean top z-coordinate after transformation
        mean_top_z = np.mean((top_nodes[:, :-1] @ R)[:, 2])
        mean_top_y = np.mean((top_nodes[:, :-1] @ R)[:, 1])
        mean_top_x = np.mean((top_nodes[:, :-1] @ R)[:, 0])

        # Judgement of positive or negative rotation matrix， always put the undeformed section at the bottom
        if p_pc[0, -1] < p_pc[n_l * 2, -1]:
            pn_index = -1
            # p_pc = p_pc @ R_K
            c_pc = c_pc @ R_K
            mean_top_z = np.mean((top_nodes[:, :-1] @ R @ R_K)[:, 2])
            mean_top_y = np.mean((top_nodes[:, :-1] @ R @ R_K)[:, 1])
            mean_top_x = np.mean((top_nodes[:, :-1] @ R @ R_K)[:, 0])

        # Setting new coordinate and remove the decimals
        # p_pc[:, -1] = p_pc[:, -1] - mean_top_z
        # p_pc[n_l * 2:: n_l * 2 + 1, -1] = 0.
        c_pc[:, -1] = c_pc[:, -1] - mean_top_z
        c_pc[n_l:: n_l + 1, -1] = 0.

        # Move the point cloud in x- and y- directions
        # p_pc[:, 0] = p_pc[:, 0] - mean_top_x
        # p_pc[:, 1] = p_pc[:, 1] - mean_top_y
        c_pc[:, 0] = c_pc[:, 0] - mean_top_x
        c_pc[:, 1] = c_pc[:, 1] - mean_top_y

        # Change order
        # p_pc[:] = p_pc[::-1, ]
        c_pc[:] = c_pc[::-1, ]

        # ax1 = plt.subplot(projection='3d')
        # ax1.scatter(c_pc[:, 0], c_pc[:, 1], c_pc[:, 2], c='grey', s=8, alpha=0.2)
        # ax1.scatter(c_pc[0, 0], c_pc[0, 1], c_pc[0, 2], c='r', s=12)
        # ax1.scatter(c_pc[1, 0], c_pc[1, 1], c_pc[1, 2], c='b', s=12)
        # ax1.scatter(c_pc[-1, 0], c_pc[-1, 1], c_pc[-1, 2], c='g', s=12)
        # plt.show()

    # Significant buckling check
    if buckle_check(c_pc, n_l + 1) == 1:
        return None

    # Loop for feature points on each slice
    int_slice = id_ / ni_slice
    center_feature = []
    for i_slice in np.arange(0, id_ + int_slice, int_slice):

        # Get current z-coordinate of slice
        id_slice = i_slice * d

        # Get the current interpolation
        ls_c = get_interpolation(c_pc, id_slice)
        # ls_p1 = get_interpolation(p_pc, id_slice)

        # Modify the profile nodes on each slice (connect start to end)
        # ls_p1.append(ls_p1[0])
        # ls_p = np.array(ls_p1)

        # Modify the center-line nodes on each slice
        ls_c = np.array(ls_c)
        ls_c1 = ls_c[:web_num + 1]
        ls_c2 = ls_c[web_num + 1: -(web_num + 1)]
        ls_c3 = ls_c[-(web_num + 1):]

        hf = int(ls_c1.shape[0] / 2)
        pt1, pt2 = np.mean(ls_c1[hf - 1: hf + 1], axis=0), np.mean(ls_c3[hf - 1: hf + 1], axis=0)
        ls_c2 = np.concatenate([pt1.reshape(1, -1), ls_c2, pt2.reshape(1, -1)], axis=0)

        s1 = feature_point_extraction(ls_c1, ni_flange)
        s2 = feature_point_extraction(ls_c2, ni_web)
        s3 = feature_point_extraction(ls_c3, ni_flange)

        # Plot slice
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.plot(ls_p[:, 0], ls_p[:, 1], c='black')
        # ax1.plot(ls_c1[:, 0], ls_c1[:, 1], c='r')
        # ax1.plot(ls_c2[:, 0], ls_c2[:, 1], c='r')
        # ax1.plot(ls_c3[:, 0], ls_c3[:, 1], c='r')
        # ax1.scatter(s1[:, 0], s1[:, 1], c='b')
        # ax1.scatter(s2[:, 0], s2[:, 1], c='b')
        # ax1.scatter(s3[:, 0], s3[:, 1], c='b')
        # plt.show()

        # Save center-line mat
        s1_wz = np.concatenate((s1, np.ones((s1.shape[0], 1)) * id_slice), axis=1)
        s2_wz = np.concatenate((s2, np.ones((s2.shape[0], 1)) * id_slice), axis=1)
        s3_wz = np.concatenate((s3, np.ones((s3.shape[0], 1)) * id_slice), axis=1)
        s = np.concatenate((s1_wz, s2_wz, s3_wz), axis=0)
        center_feature.append(s)

    # Rotate back
    center_feature = np.array(center_feature)
    center_feature = center_feature.reshape((-1, 3))

    # if bt_index == 1:
    #     if pn_index == -1:
    #         center_feature[:, -1] = center_feature[:, -1] + mean_top_z
    #         center_feature = center_feature @ np.linalg.inv(R_K) @ np.linalg.inv(R)
    #     else:
    #         center_feature[:, -1] = center_feature[:, -1] + mean_top_z
    #         center_feature = center_feature @ np.linalg.inv(R)
    #
    #     center_feature = center_feature[::-1, ]

    # Plot 3D
    # ax1 = plt.subplot(projection='3d')
    # ax1.scatter(center_feature[:, 0], center_feature[:, 1], center_feature[:, 2], c='b')

    # Compare original
    # c_pc = model['Solid_node']['center_node_coord'][check, bt_index]
    # ax1.scatter(c_pc[:, 0], c_pc[:, 1], c_pc[:, 2], c='r')
    # plt.show()

    return center_feature


if __name__ == "__main__":

    h5_file = h5py.File('column.hdf5', "a")

    for i1 in h5_file.keys():
        for i2 in h5_file[i1].keys():
            for i3 in h5_file[i1][i2].keys():
                for i4 in h5_file[i1][i2][i3].keys():
                    for i5 in h5_file[i1][i2][i3][i4].keys():
                        model_ = h5_file[i1][i2][i3][i4][i5]
                        print(model_.name)

                        num_ind = len(model_['indices'][:])
                        fm = []
                        end_index = None

                        for ii in range(num_ind):

                            rc_i = model_['Reaction']['reserve_capacity'][model_['indices'][ii]]
                            print(ii, rc_i)
                            fm_i = []

                            for i in [0, 1]:
                                mat = get_slice_matrix(model=model_, bt_index=i, c_frame=ii, ni_web=8, ni_flange=5,
                                                       ni_slice=20, id_=1.0)

                                if mat is None or rc_i[0] < 0.4 or rc_i[1] < 0.4:
                                    end_index = ii
                                    break
                                else:
                                    fm_i.append(mat)

                            if end_index is None:
                                fm.append(fm_i)
                            else:
                                break

                        fm = np.array(fm)

                        # Modify model index length
                        indices = model_['indices'][:end_index]
                        beam_c = model_['Beam_node']['beam_node_coord'][:end_index, ...]
                        beam_r = model_['Beam_node']['beam_node_rotation'][:end_index, ...]
                        c_node = model_['Solid_node']['center_node_coord'][:end_index, ...]
                        p_node = model_['Solid_node']['profile_node_coord'][:end_index, ...]

                        del model_['indices']
                        del model_['Beam_node']['beam_node_coord']
                        del model_['Beam_node']['beam_node_rotation']
                        del model_['Solid_node']['center_node_coord']
                        del model_['Solid_node']['profile_node_coord']

                        model_.create_dataset('indices', data=indices)
                        model_['Beam_node'].create_dataset('beam_node_coord', data=beam_c)
                        model_['Beam_node'].create_dataset('beam_node_rotation', data=beam_r)
                        model_['Solid_node'].create_dataset('center_node_coord', data=c_node)
                        model_['Solid_node'].create_dataset('profile_node_coord', data=p_node)

                        # Standard local point cloud
                        group_ds = write_group(model_, 'Deformed_shape')
                        if "processed_center" in group_ds.keys():
                            del model_["processed_center"]
                        group_ds.create_dataset('processed_center', data=fm)


