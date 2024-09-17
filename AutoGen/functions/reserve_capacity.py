import numpy as np


def build_cumulative(x):
    x_ = [abs(x[0])]
    for i in range(len(x) - 1):
        x_.append(x_[-1] + abs(x[i + 1] - x[i]))

    return x_


def get_end(info_list, ind):
    for i in range(len(info_list) - 1):
        if info_list[i + 1] > ind > info_list[i]:
            break

    return info_list[i], info_list[i + 1]


def calculate_reserve_capacity(cycle_x, cycle_y):

    # Normalize moment
    cycle_x = cycle_x.tolist()
    cycle_y_pre = cycle_y.tolist()
    cycle_y = [i / max(map(abs, cycle_y_pre)) for i in cycle_y_pre]

    if len(cycle_x) != len(cycle_y):
        raise ValueError("Two vectors do not have the same dimension!")

    # Initiate parameters
    reversal_index = 0
    cycle_init_index = 0
    cycle_p_info = []
    cycle_n_info = []
    cycle_p_max = []
    cycle_n_min = []

    # Start the loop
    for q in range(1, len(cycle_x)):

        # If last increment step
        if q == len(cycle_x) - 1:
            cycle_y_list = cycle_y[cycle_init_index:]

            # Record info
            # If positive direction half cycle
            if reversal_index == 0:

                # If there is a descending stage in the last positive half cycle
                if max(cycle_y_list) > cycle_y_list[-1]:

                    # Obtain maximum moment's corresponding index in the half cycle
                    max_ind = cycle_y_list.index(max(cycle_y_list))

                    # Record [maximum point index in full cycle, None, number of maximum points before]
                    cycle_p_info.append([cycle_init_index + max_ind, None, len(cycle_p_max)])

                    # Update number of maximum points
                    cycle_p_max.append(cycle_init_index + max_ind)

                # If there is not a descending stage in the last positive half cycle
                else:

                    # Update number of maximum points
                    if cycle_y[q] > 0:
                        cycle_p_max.append(q)

            # If negative direction half cycle
            elif reversal_index == 1:

                # If there is a descending stage in the last negative half cycle
                if min(cycle_y_list) < cycle_y_list[-1]:

                    # Obtain minimum moment's corresponding index in the half cycle
                    min_ind = cycle_y_list.index(min(cycle_y_list))

                    # Record [minimum point index in full cycle, None, number of minimum points before]
                    cycle_n_info.append([cycle_init_index + min_ind, None, len(cycle_n_min)])

                    # Update number of minimum points
                    cycle_n_min.append(cycle_init_index + min_ind)

                # If there is not a descending stage in the last negative half cycle
                else:

                    # Update number of minimum points
                    if cycle_y[q] < 0:
                        cycle_n_min.append(q)

        # Positive half cycle and q - 1 is the index of the turning point
        elif reversal_index == 0 and cycle_x[q] < cycle_x[q - 1]:

            # Obtain the positive half cycle
            cycle_y_list = cycle_y[cycle_init_index: q]

            # If there is a descending stage in the positive half cycle
            if max(cycle_y_list) > cycle_y_list[-1]:

                # Obtain maximum moment's corresponding index in the half cycle
                max_ind = cycle_y_list.index(max(cycle_y_list))

                # Record [maximum point index in full cycle, end point of half cycle, number of maximum points before]
                cycle_p_info.append([cycle_init_index + max_ind, q - 1, len(cycle_p_max)])

                # Update number of maximum points
                cycle_p_max.append(cycle_init_index + max_ind)

            # If there is not a descending stage in the positive half cycle
            else:

                # Update number of maximum points
                cycle_p_max.append(q - 1)

            # Update parameters
            cycle_init_index = q - 1
            reversal_index = 1

        # Negative half cycle and q - 1 is the index of the turning point
        elif reversal_index == 1 and cycle_x[q] > cycle_x[q - 1]:

            # Obtain the negative half cycle
            cycle_y_list = cycle_y[cycle_init_index: q]

            # If there is a descending stage in the negative half cycle
            if min(cycle_y_list) < cycle_y_list[-1]:

                # Obtain minimum moment's corresponding index in the half cycle
                min_ind = cycle_y_list.index(min(cycle_y_list))

                # Record [minimum point index in full cycle, end point of half cycle, number of minimum points before]
                cycle_n_info.append([cycle_init_index + min_ind, q - 1, len(cycle_n_min)])

                # Update number of minimum points
                cycle_n_min.append(cycle_init_index + min_ind)

            # If there is not a descending stage in the negative half cycle
            else:

                # Update number of minimum points
                cycle_n_min.append(q - 1)

            # Update parameters
            cycle_init_index = q - 1
            reversal_index = 0

    # Build cumulative drift and reserve capacity vectors
    cycle_y = np.array(cycle_y)
    cum_x = build_cumulative(np.array(cycle_x))
    rm_p = np.ones_like(cycle_y)
    rm_n = np.ones_like(cycle_y)

    # Parameter loi denotes the index where degrading starts
    ones_indices = np.where(np.abs(cycle_y) > 0.99)[0]
    loi = np.max(ones_indices)

    # Only keep the info after degrading
    cycle_p_info = [cycle_p_info[i] for i in range(len(cycle_p_info)) if cycle_p_info[i][1] is None or cycle_p_info[i][1] >= loi]
    cycle_n_info = [cycle_n_info[i] for i in range(len(cycle_n_info)) if cycle_n_info[i][1] is None or cycle_n_info[i][1] >= loi]

    # Monotonic case, no need to modify
    if len(cycle_n_min) == 0 and len(cycle_p_max) == 1:
        rm_p[loi:] = cycle_y[loi:]
        rm_n[loi:] = cycle_y[loi:]
        return np.concatenate((rm_p.reshape((-1, 1)), rm_n.reshape((-1, 1))), axis=1)

    # If there is a valid descending stage after member degrading
    if cycle_p_info:

        # For each positive half cycle with descending stage
        for i in range(len(cycle_p_info)):

            # Maximum point index in this half cycle
            ds1 = cycle_p_info[i][0]

            # End point index in this half cycle
            hld = cycle_p_info[i][1]

            # If not the last descending stage
            if i != len(cycle_p_info) - 1:

                # Next maximum point index
                ds2 = cycle_p_info[i + 1][0]

                # Interpolate between end point and descending stage
                rm_p[hld:ds2 + 1] = np.interp(cum_x[hld:ds2 + 1], [cum_x[hld], cum_x[ds2]], [cycle_y[hld], cycle_y[ds2]])

            # If the first descending stage
            if i == 0:
                rm_p[loi:ds1 + 1] = np.interp(cum_x[loi:ds1 + 1], [cum_x[loi], cum_x[ds1]], [1., cycle_y[ds1]])

            # If there is a holding stage
            if hld is not None:
                rm_p[ds1:hld + 1] = cycle_y[ds1:hld + 1]

            # If there is not a holding stage
            else:
                rm_p[ds1:hld] = cycle_y[ds1:hld]

            # If the last descending stage
            if i == len(cycle_p_info) - 1:
                trailing_ones = np.where(rm_p != 1)[0]
                if trailing_ones[-1] != len(cycle_x):
                    rm_p[trailing_ones[-1] + 1:] = rm_p[trailing_ones[-1]]

    # If there is a valid descending stage after member degrading
    if cycle_n_info:

        # For each positive half cycle with descending stage
        for i in range(len(cycle_n_info)):

            # Maximum point index in this half cycle
            ds1 = cycle_n_info[i][0]

            # End point index in this half cycle
            hld = cycle_n_info[i][1]

            # If not the last descending stage
            if i != len(cycle_n_info) - 1:

                # Next maximum point index
                ds2 = cycle_n_info[i + 1][0]

                # Interpolate between end point and descending stage
                rm_n[hld:ds2 + 1] = np.interp(cum_x[hld:ds2 + 1], [cum_x[hld], cum_x[ds2]], [-cycle_y[hld], -cycle_y[ds2]])

            # If the first descending stage
            if i == 0:
                rm_n[loi:ds1 + 1] = np.interp(cum_x[loi:ds1 + 1], [cum_x[loi], cum_x[ds1]], [1., -cycle_y[ds1]])

            # If there is a holding stage
            if hld is not None:
                rm_n[ds1:hld + 1] = -cycle_y[ds1:hld + 1]

            # If there is not a holding stage
            else:
                rm_n[ds1:hld] = -cycle_y[ds1:hld]

            # If the last descending stage
            if i == len(cycle_n_info) - 1:
                trailing_ones = np.where(rm_n != 1)[0]
                if trailing_ones[-1] != len(cycle_x):
                    rm_n[trailing_ones[-1] + 1:] = rm_n[trailing_ones[-1]]

    # Final check out-of-envelope curves to expand reserve capacity
    n_mins = np.sort(np.array([cycle_n_info[i][0] for i in range(len(cycle_n_info))] +
                              [cycle_n_info[i][1] for i in range(len(cycle_n_info)) if cycle_n_info[i][1] is not None] + [loi]))
    p_maxs = np.sort(np.array([cycle_p_info[i][0] for i in range(len(cycle_p_info))] +
                              [cycle_p_info[i][1] for i in range(len(cycle_p_info)) if cycle_p_info[i][1] is not None] + [loi]))

    # Obtain the positive peak's corresponding x, y and rm values
    xm_p = np.array([cum_x[ind] for ind in cycle_p_max])
    ym_p = np.array([cycle_y[ind] for ind in cycle_p_max])
    rm_p_s = np.array([rm_p[ind] for ind in cycle_p_max])
    res_p = ym_p - rm_p_s

    while np.max(res_p) > 0:

        ind_max = np.argmax(res_p)

        xm_i = xm_p[ind_max]
        ym_i = ym_p[ind_max]
        ind_t = cycle_p_max[ind_max]

        ind_p, ind_a = get_end(p_maxs, ind_t)
        xm_ip = cum_x[ind_p]
        ym_ip = rm_p[ind_p]
        xm_ia = cum_x[ind_a]
        ym_ia = rm_p[ind_a]

        rm_p[ind_p: ind_t + 1] = np.interp(cum_x[ind_p: ind_t + 1], [xm_ip, xm_i], [ym_ip, ym_i])
        if ind_a != -1:
            rm_p[ind_t: ind_a + 1] = np.interp(cum_x[ind_t: ind_a + 1], [xm_i, xm_ia], [ym_i, ym_ia])
        else:
            rm_p[ind_t:] = np.interp(cum_x[ind_t:], [xm_i, xm_ia], [ym_i, rm_p[-1]])

        rm_p_s = np.array([rm_p[ind] for ind in cycle_p_max])
        res_p = ym_p - rm_p_s
        p_maxs = np.sort(np.append(p_maxs, ind_t))

    # Obtain the negative peak's corresponding x, y and rm values
    xm_n = np.array([cum_x[ind] for ind in cycle_n_min])
    ym_n = np.array([cycle_y[ind] for ind in cycle_n_min])
    rm_n_s = np.array([rm_n[ind] for ind in cycle_n_min])
    res_n = ym_n + rm_n_s

    while np.min(res_n) < 0:

        ind_min = np.argmin(res_n)

        xm_i = xm_n[ind_min]
        ym_i = ym_n[ind_min]
        ind_t = cycle_n_min[ind_min]

        ind_p, ind_a = get_end(n_mins, ind_t)
        xm_ip = cum_x[ind_p]
        ym_ip = -rm_n[ind_p]
        xm_ia = cum_x[ind_a]
        ym_ia = -rm_n[ind_a]

        rm_n[ind_p: ind_t + 1] = np.interp(cum_x[ind_p: ind_t + 1], [xm_ip, xm_i], [-ym_ip, -ym_i])
        if ind_a != -1:
            rm_n[ind_t: ind_a + 1] = np.interp(cum_x[ind_t: ind_a + 1], [xm_i, xm_ia], [-ym_i, -ym_ia])
        else:
            rm_n[ind_t:] = np.interp(cum_x[ind_t:], [xm_i, xm_ia], [-ym_i, rm_n[-1]])

        rm_n_s = np.array([rm_n[ind] for ind in cycle_n_min])
        res_n = ym_n + rm_n_s
        n_mins = np.sort(np.append(n_mins, ind_t))

    # Processing the unknown parts with lower bound
    cycle_n_min_selected = []

    # All ones at the negative side
    if (rm_n == np.ones_like(rm_n)).all():
        case_index = 2

        # Select all the negative peaks after the peak, as well as the peak
        for i in range(1, len(cycle_n_min) + 1):
            if cycle_n_min[-i] > loi:
                cycle_n_min_selected.append(cycle_n_min[-i])
        cycle_n_min_selected.append(loi)

    # Not all ones at the negative side
    else:
        case_index = 1

        for i in range(1, len(cycle_n_min) + 1):
            if i != len(cycle_n_min) and rm_n[cycle_n_min[-i]] == rm_n[cycle_n_min[-i-1]]:
                cycle_n_min_selected.append(cycle_n_min[-i])
            else:
                cycle_n_min_selected.append(cycle_n_min[-i])
                cycle_n_min_selected.append(np.where(rm_n != rm_n[-1])[0][-1] + 1)
                break

    cycle_n_min_selected = cycle_n_min_selected[::-1]
    ind0 = cycle_n_min_selected[0]

    if len(cycle_n_min_selected) > 2:

        # Here is to restrict cycle_n_min_selected to let its indices' y values strictly go down
        while True:

            slope_index = 0
            for i in range(case_index, len(cycle_n_min_selected)):
                if cycle_y[cycle_n_min_selected[i]] - cycle_y[cycle_n_min_selected[i-1]] < 0:
                    cycle_n_min_selected.pop(i-1)
                    slope_index = 1
                    break
            if slope_index == 0:
                break

        for i in range(1, len(cycle_n_min_selected)):
            ind1 = cycle_n_min_selected[i - 1]
            ind2 = cycle_n_min_selected[i]

            # loi to first negative peak after loi
            if case_index == 2 and i == 1:
                rm_n[ind1: ind2 + 1] = np.interp(cum_x[ind1: ind2 + 1], [cum_x[ind1], cum_x[ind2]], [1, -cycle_y[ind2]])

            else:
                rm_n[ind1: ind2 + 1] = np.interp(cum_x[ind1: ind2 + 1], [cum_x[ind1], cum_x[ind2]], [-cycle_y[ind1], -cycle_y[ind2]])

            if i == len(cycle_n_min_selected) - 1:
                rm_n[ind2:] = -cycle_y[ind2]

        for i in range(len(rm_n)):
            if rm_n[i] < rm_p[i]:
                rm_n[i] = rm_p[i]

    return np.concatenate((rm_p.reshape((-1, 1)), rm_n.reshape((-1, 1))), axis=1)
