import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from numba import jit


def buckle_check(bottom_pc, n_lf):
    bottom_pc = bottom_pc.reshape(-1, n_lf, 3)
    z_diff = np.diff(bottom_pc[:, :, 2], axis=1)
    return np.any(z_diff <= 0).astype(int)


# Extract feature points on each slice
def feature_point_extraction(node_array, interval_num):
    lengths = np.linalg.norm(np.diff(node_array, axis=0), axis=1)
    ls_acc = np.concatenate(([0], np.cumsum(lengths)))
    total_length = ls_acc[-1]
    positions = np.linspace(0, total_length, interval_num + 1)
    node_positions = np.interp(positions, ls_acc, node_array[:, 0]), np.interp(positions, ls_acc, node_array[:, 1])
    return np.vstack(node_positions).T


# z-node1 < z-interpolation < z-node2, linear lagrange interpolation
# Higher order interpolation was also tested, but they tend to have high interpolation error
@jit(nopython=True)
def lagrange_interpolation(z, node1, node2):
    p1x, p2x = node1[0], node2[0]
    p1y, p2y = node1[1], node2[1]
    p1z, p2z = node1[2], node2[2]
    k2 = (z - p2z) / (p1z - p2z)
    k3 = (z - p1z) / (p2z - p1z)
    x = k2 * p1x + k3 * p2x
    y = k2 * p1y + k3 * p2y
    return [x, y]


# Get slice of center-line or profile node
def get_interpolation(pc_, L_s_):

    mask_between = np.where((pc_[:-1, 2] < L_s_) & (pc_[1:, 2] > L_s_))[0]
    mask_equal = (pc_[:, 2] == L_s_)
    interpolations = np.array([lagrange_interpolation(L_s_, pc_[j], pc_[j + 1]) for j in mask_between])
    equal_points = pc_[mask_equal][:, :2].reshape(-1, 2)

    if len(interpolations) == 0:
        return equal_points
    elif len(interpolations) > 0 and len(equal_points) > 0:
        return np.vstack((interpolations, equal_points))
    else:
        return interpolations


@jit(nopython=True)
def rodrigues_rotation(v1, v2):
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)

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


@jit(nopython=True)
def invert_pc(p_pc, c_pc, n_l):

    p_pc = p_pc.astype(np.float64)
    c_pc = c_pc.astype(np.float64)

    # Select top end nodes and add a new dimension for constant value
    top_nodes = p_pc[n_l * 2:: n_l * 2 + 1]
    top_nodes = np.concatenate((top_nodes, np.ones((top_nodes.shape[0], 1))), axis=1).astype(np.float64)

    # Get correlation matrix and then calculate top end plane vector
    corr_mat = np.transpose(top_nodes) @ top_nodes
    _, _, vt = np.linalg.svd(corr_mat)
    plane_vec1 = vt[-1][:-1] / np.linalg.norm(vt[-1][:-1])

    # Calculate rotation matrix for setting a new coordinate system
    R = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), plane_vec1).astype(np.float64)
    R_K = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)

    # Transformation of all the points
    p_pc = p_pc @ R
    c_pc = c_pc @ R

    # Get mean top z-coordinate after transformation
    top_array = np.ascontiguousarray(top_nodes[:, :-1])
    inter_mat0 = top_array @ R
    mean_top_z = np.mean(inter_mat0[:, 2])
    mean_top_y = np.mean(inter_mat0[:, 1])
    mean_top_x = np.mean(inter_mat0[:, 0])

    # Judgement of positive or negative rotation matrix， always put the undeformed section at the bottom
    if p_pc[0, -1] < p_pc[n_l * 2, -1]:
        c_pc = c_pc @ R_K
        inter_mat1 = top_array @ R @ R_K
        mean_top_z = np.mean(inter_mat1[:, 2])
        mean_top_y = np.mean(inter_mat1[:, 1])
        mean_top_x = np.mean(inter_mat1[:, 0])

    # Setting new coordinate and remove the decimals
    c_pc[:, -1] = c_pc[:, -1] - mean_top_z
    c_pc[n_l:: n_l + 1, -1] = 0.

    # Move the point cloud in x- and y- directions
    c_pc[:, 0] = c_pc[:, 0] - mean_top_x
    c_pc[:, 1] = c_pc[:, 1] - mean_top_y

    # Change order
    c_pc[:] = c_pc[::-1]
    return c_pc


def assemble_features(ls_c, web_num, ni_flange, ni_web, id_slice):

    ls_c1 = ls_c[:web_num + 1]
    ls_c2 = ls_c[web_num + 1: -(web_num + 1)]
    ls_c3 = ls_c[-(web_num + 1):]

    hf = int(ls_c1.shape[0] / 2)
    pt1, pt2 = np.mean(ls_c1[hf - 1: hf + 1], axis=0), np.mean(ls_c3[hf - 1: hf + 1], axis=0)
    ls_c2 = np.concatenate([pt1.reshape(1, -1), ls_c2, pt2.reshape(1, -1)], axis=0)

    # plt.scatter(ls_c1[:, 0], ls_c1[:, 1], c='red', s=8, alpha=0.2)
    # plt.scatter(ls_c2[:, 0], ls_c2[:, 1], c='red', s=8, alpha=0.2)
    # plt.scatter(ls_c3[:, 0], ls_c3[:, 1], c='red', s=8, alpha=0.2)
    # plt.show()

    s1 = feature_point_extraction(ls_c1, ni_flange)
    s2 = feature_point_extraction(ls_c2, ni_web)
    s3 = feature_point_extraction(ls_c3, ni_flange)

    # Save center-line mat
    s1_wz = np.concatenate((s1, np.ones((s1.shape[0], 1)) * id_slice), axis=1)
    s2_wz = np.concatenate((s2, np.ones((s2.shape[0], 1)) * id_slice), axis=1)
    s3_wz = np.concatenate((s3, np.ones((s3.shape[0], 1)) * id_slice), axis=1)
    s = np.concatenate((s1_wz, s2_wz, s3_wz), axis=0)

    return s


def visualize_point(pc):
    ax = plt.subplot(projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='grey', s=8, alpha=0.2)
    ax.scatter(pc[0, 0], pc[0, 1], pc[0, 2], c='r', s=12)
    ax.scatter(pc[1, 0], pc[1, 1], pc[1, 2], c='b', s=12)
    ax.scatter(pc[-1, 0], pc[-1, 1], pc[-1, 2], c='g', s=12)
    plt.show()


def get_slice_matrix(model, bt_index, c_frame, ni_web, ni_flange, ni_slice, id_):

    # Get bottom/top end member segment's profile and centerline nodes
    p_pc = model['Solid_node']['profile_node_coord'][c_frame, bt_index]
    c_pc = model['Solid_node']['center_node_coord'][c_frame, bt_index]
    d, n_l = model.attrs['d'], model.attrs["n_l"]
    web_num = model.attrs["n_w"] * 2 + model.attrs["n_tw"] + model.attrs["n_k"]

    # ax = plt.subplot(projection='3d')
    # ax.scatter(c_pc[:, 0], c_pc[:, 1], c_pc[:, 2], c='red', s=8, alpha=0.2)
    # plt.show()

    # If top solid element segment
    if bt_index == 1:
        c_pc = invert_pc(p_pc, c_pc, n_l)

    # ax = plt.subplot(projection='3d')
    # ax.scatter(c_pc[:, 0], c_pc[:, 1], c_pc[:, 2], c='red', s=8, alpha=0.2)
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

        # # Get the current interpolation
        ls_c = get_interpolation(c_pc, id_slice)
        # plt.scatter(ls_c[:, 0], ls_c[:, 1], c='red', s=8, alpha=0.2)
        # plt.show()

        s = assemble_features(ls_c, web_num, ni_flange, ni_web, id_slice)
        center_feature.append(s)

    # Rotate back
    center_feature = np.array(center_feature)
    center_feature = center_feature.reshape((-1, 3))

    return center_feature


def write_group(upper_group, new_group):
    if new_group in upper_group.keys():
        return upper_group[new_group]
    else:
        return upper_group.create_group(new_group)


if __name__ == "__main__":

    path0 = '/scratch/gu/MLDB/hdf5_folder/column_f.hdf5'
    h5_file = h5py.File(path0, "a")

    for i1 in h5_file.keys():
        for i2 in h5_file[i1].keys():
            for i3 in h5_file[i1][i2].keys():
                for i4 in h5_file[i1][i2][i3].keys():
                    for i5 in h5_file[i1][i2][i3][i4].keys():
                        model_ = h5_file[i1][i2][i3][i4][i5]
                        print(model_.name, flush=True)

                        ind = model_['indices'][:]
                        fm = []
                        end_index = None

                        for ii in range(len(ind)):
                            rc_i = model_['Reaction']['reserve_capacity'][ind[ii]]
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
                        print(fm.shape, flush=True)

                        # Modify model index length
                        indices = model_['indices'][:end_index]
                        beam_c = model_['Beam_node']['beam_node_coord'][:end_index, ...]
                        beam_r = model_['Beam_node']['beam_node_rotation'][:end_index, ...]
                        c_node = model_['Solid_node']['center_node_coord'][:end_index, ...]
                        p_node = model_['Solid_node']['profile_node_coord'][:end_index, ...]
                        strs = model_['Solid_element']['ele_stress'][:end_index, ...]
                        strn = model_['Solid_element']['ele_strain'][:end_index, ...]
                        cord = model_['Solid_element']['ele_coord'][:end_index, ...]

                        del model_['indices']
                        del model_['Beam_node']['beam_node_coord']
                        del model_['Beam_node']['beam_node_rotation']
                        del model_['Solid_node']['center_node_coord']
                        del model_['Solid_node']['profile_node_coord']
                        del model_['Solid_element']['ele_stress']
                        del model_['Solid_element']['ele_strain']
                        del model_['Solid_element']['ele_coord']

                        model_.create_dataset('indices', data=indices)
                        model_['Beam_node'].create_dataset('beam_node_coord', data=beam_c)
                        model_['Beam_node'].create_dataset('beam_node_rotation', data=beam_r)
                        model_['Solid_node'].create_dataset('center_node_coord', data=c_node)
                        model_['Solid_node'].create_dataset('profile_node_coord', data=p_node)
                        model_['Solid_element'].create_dataset('ele_stress', data=strs)
                        model_['Solid_element'].create_dataset('ele_strain', data=strn)
                        model_['Solid_element'].create_dataset('ele_coord', data=cord)

                        # Standard local point cloud
                        group_ds = write_group(model_, 'Deformed_shape')
                        model_['Deformed_shape'].create_dataset('processed_center', data=fm)
