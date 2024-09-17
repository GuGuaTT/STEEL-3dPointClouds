import numpy as np
import h5py


def first_cycle_envelope_pos(x, y):

    first_cycles = np.array([2, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50])
    max_ind = np.argmax(y)
    cycle_array = np.array([])
    for i in range(1, len(x) - 1):
        if x[i - 1] < x[i] and x[i + 1] < x[i]:
            cycle_array = np.append(cycle_array, i)

    id_array = np.arange(cycle_array[0] + 1)
    for i in range(len(first_cycles)):
        if len(cycle_array) > first_cycles[i]:
            id_array = np.append(id_array, cycle_array[first_cycles[i]])
    id_array = np.append(id_array, max_ind)
    id_array = np.sort(np.array(list(set(id_array)))).astype(int)

    return x[id_array], y[id_array]


def first_cycle_envelope_neg(x, y):

    first_cycles = np.array([2, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50])
    max_ind = np.argmin(y)
    cycle_array = np.array([])
    for i in range(1, len(x) - 1):
        if x[i - 1] > x[i] and x[i + 1] > x[i]:
            cycle_array = np.append(cycle_array, i)

    id_array = np.array([0])
    for i in range(len(first_cycles)):
        if len(cycle_array) > first_cycles[i]:
            id_array = np.append(id_array, cycle_array[first_cycles[i]])

    id_array = np.append(id_array, max_ind)
    id_array = np.sort(np.array(list(set(id_array)))).astype(int)

    return x[id_array], y[id_array]


def calculate_cross_point(k1, b1, k2, x_m, y_m):
    """
    Get the inferred yield point in IMK model.
    :param k1: Effective modulus.
    :param b1: 0 in this case.
    :param k2: Slope of post-yielding stage.
    :param x_m: x-value of peak load.
    :param y_m: y-value of peak load.
    :return: x- and y-values of inferred yield point.
    """
    b2 = y_m - k2 * x_m
    x = (b2 - b1) / (k1 - k2)
    y = x * k1 + b1
    return x, y


def bisearch_area(test_x, test_y, xp, yp, eff, tol=1e-4, mx=int(3e5)):
    """
    Find the best value for slope of post-yielding stage by binary search.
    :param test_x: x-values of test curve.
    :param test_y: y-values of test curve.
    :param xp: x-value of peak load.
    :param yp: y-value of peak load.
    :param eff: Effective modulus.
    :param tol: tolerance.
    :param mx: max.
    :return: ultimate x- and y-values of yield point.
    """
    slope_ys_lower = 0
    slope_ys_upper = eff

    for i in range(mx):

        slope_ys_current = (slope_ys_lower + slope_ys_upper) / 2
        xc, yc = calculate_cross_point(eff, 0, slope_ys_current, xp, yp)

        slope_E_x = np.linspace(0, xc, 1000)
        slope_E_y = np.linspace(0, yc, 1000)
        slope_ys_x = np.linspace(xc, xp, 1000)
        slope_ys_y = np.linspace(yc, yp, 1000)

        bilinear_x = np.concatenate((slope_E_x, slope_ys_x))
        bilinear_y = np.concatenate((slope_E_y, slope_ys_y))

        diff_area = np.trapz(test_y, x=test_x) - np.trapz(bilinear_y, x=bilinear_x)

        if diff_area > 0:
            slope_ys_upper = slope_ys_current
        else:
            slope_ys_lower = slope_ys_current

        if abs(diff_area) < tol or i == mx - 1:
            # print("Slope at the post-yielding stage: %s" % slope_ys_current)
            return xc, yc


def calculate_imk_pos(model, length, axl, bc):

    rm0 = model['Reaction']['bot_RM1'][:] / 1e6
    tu0 = model['Displacement']['top_U2'][:] / length
    tu, rm = first_cycle_envelope_pos(tu0, rm0)

    d, bf, tf, tw, r = model.attrs['d'], model.attrs['b_f'], model.attrs['t_f'], model.attrs['t_w'], model.attrs['r']
    h = d - 2 * tf - 2 * r

    Iy = (bf ** 3 * tf * 2 + tw ** 3 * (d - 2 * tf)) / 12
    A = tf * bf * 2 + (d - 2 * tf) * tw
    ry = (Iy / A) ** 0.5

    if bc == 'Fixed':
        bc_ind = 0
    else:
        bc_ind = 1

    peak_ind = np.argmax(rm)
    pre_rm = rm[:peak_ind + 1]
    pst_rm = rm[peak_ind:]
    pre_tu = tu[:peak_ind + 1]
    pst_tu = tu[peak_ind:]
    tu_p, rm_p = tu[peak_ind], rm[peak_ind]

    if rm[-1] >= rm_p * 0.8:
        return None

    tu_M_n03 = np.interp(0.3 * rm_p, pre_rm, pre_tu)
    eff_E = 0.3 * rm_p / tu_M_n03
    tu_c, rm_c = bisearch_area(pre_tu, pre_rm, tu_p, rm_p, eff_E)

    if rm[-1] <= rm_p * 0.5:
        rm_u = rm_p * 0.5
        tu_u = np.interp(-rm_u, -pst_rm, pst_tu)
    else:
        rm_u = rm[-1]
        tu_u = tu[-1]

    k = (rm_p - rm_u) / (tu_p - tu_u)
    tu_ult = -rm_u / k + tu_u
    pp_x = np.linspace(tu_p, tu_ult, 1000)
    pp_y = np.linspace(rm_p, 0, 1000)
    rm_b = rm_p * 0.8
    tu_b = np.interp(-rm_b, -pp_y, pp_x)

    theta_p = tu_b - tu_c
    theta_pc = tu_ult - tu_b

    return np.array([theta_p, theta_pc]), np.array([h / tw, bf / tf / 2, length / d, length / ry, d, axl, bc_ind])


def calculate_imk_neg(model, length, axl, bc):

    rm0 = model['Reaction']['bot_RM1'][:] / 1e6
    tu0 = model['Displacement']['top_U2'][:] / length
    tu, rm = first_cycle_envelope_neg(tu0, rm0)

    d, bf, tf, tw, r = model.attrs['d'], model.attrs['b_f'], model.attrs['t_f'], model.attrs['t_w'], model.attrs['r']
    h = d - 2 * tf - 2 * r

    Iy = (bf ** 3 * tf * 2 + tw ** 3 * (d - 2 * tf)) / 12
    A = tf * bf * 2 + (d - 2 * tf) * tw
    ry = (Iy / A) ** 0.5

    if bc == 'Fixed':
        bc_ind = 0
    else:
        bc_ind = 1

    peak_ind = np.argmin(rm)
    pre_rm = rm[:peak_ind + 1]
    pst_rm = rm[peak_ind:]
    pre_tu = tu[:peak_ind + 1]
    pst_tu = tu[peak_ind:]
    tu_p, rm_p = tu[peak_ind], rm[peak_ind]

    if rm[-1] <= rm_p * 0.8:
        return None

    tu_M_n03 = np.interp(-0.3 * rm_p, -pre_rm, -pre_tu)
    eff_E = -0.3 * rm_p / tu_M_n03
    tu_c, rm_c = bisearch_area(-pre_tu, -pre_rm, -tu_p, -rm_p, eff_E)
    tu_c, rm_c = -tu_c, -rm_c

    if rm[-1] >= rm_p * 0.5:
        rm_u = rm_p * 0.5
        tu_u = np.interp(rm_u, pst_rm, pst_tu)
    else:
        rm_u = rm[-1]
        tu_u = tu[-1]

    k = (rm_p - rm_u) / (tu_p - tu_u)
    tu_ult = -rm_u / k + tu_u
    pp_x = np.linspace(tu_p, tu_ult, 1000)
    pp_y = np.linspace(rm_p, 0, 1000)
    rm_b = rm_p * 0.8
    tu_b = np.interp(rm_b, pp_y, pp_x)

    theta_p = tu_b - tu_c
    theta_pc = tu_ult - tu_b

    return np.array([-theta_p, -theta_pc]), np.array([h / tw, bf / tf / 2, length / d, length / ry, d, axl, bc_ind])


if __name__ == "__main__":

    root = 'data\\'
    h5_file = h5py.File('column.hdf5', "r")
    test_section = h5_file.attrs['test']
    validation_section = h5_file.attrs['validation']
    training_section = h5_file.attrs['training']

    ts_output, vl_output, tr_output = [], [], []
    ts_input, vl_input, tr_input = [], [], []
    ts_name, vl_name, tr_name = [], [], []

    for i1 in h5_file.keys():
        for i2 in h5_file[i1].keys():
            for i4 in h5_file[i1][i2]['Symmetric'].keys():
                for i5 in h5_file[i1][i2]['Symmetric'][i4].keys():
                    md = h5_file[i1][i2]['Symmetric'][i4][i5]

                    name = i1 + '-' + i2 + '-Symmetric-' + i4 + '-' + i5
                    print(name)
                    info1 = calculate_imk_pos(md, float(i2), float(i4), i5)
                    # info2 = calculate_imk_neg(md, float(i2), float(i4), i5)

                    # Categorize sections
                    if i1 in test_section:
                        ind_pts = 0
                    elif i1 in validation_section:
                        ind_pts = 1
                    elif i1 in training_section:
                        ind_pts = 2
                    else:
                        raise KeyError

                    if info1 is not None:
                        out, inp = info1

                        if ind_pts == 0:
                            ts_output.append(out)
                            ts_input.append(inp)
                            ts_name.append(name)

                        elif ind_pts == 1:
                            vl_output.append(out)
                            vl_input.append(inp)
                            vl_name.append(name)

                        else:
                            tr_output.append(out)
                            tr_input.append(inp)
                            tr_name.append(name)

                    # if info2 is not None:
                    #     out, inp = info2
                    #
                    #     if ind_pts == 0:
                    #         ts_output.append(out)
                    #         ts_input.append(inp)
                    #         ts_name.append(name)
                    #
                    #     elif ind_pts == 1:
                    #         vl_output.append(out)
                    #         vl_input.append(inp)
                    #         vl_name.append(name)
                    #
                    #     else:
                    #         tr_output.append(out)
                    #         tr_input.append(inp)
                    #         tr_name.append(name)

    np.save(root + 'ts_theta.npy', np.array(ts_output))
    np.save(root + 'vl_theta.npy', np.array(vl_output))
    np.save(root + 'tr_theta.npy', np.array(tr_output))

    np.save(root + 'ts_input.npy', np.array(ts_input))
    np.save(root + 'vl_input.npy', np.array(vl_input))
    np.save(root + 'tr_input.npy', np.array(tr_input))

    np.save(root + 'ts_name.npy', np.array(ts_name))
    np.save(root + 'vl_name.npy', np.array(vl_name))
    np.save(root + 'tr_name.npy', np.array(tr_name))
