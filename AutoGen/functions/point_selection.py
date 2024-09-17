
"""
** Function for increment point selection.
"""

import numpy as np


def point_selection(cols_x, cols_y, mem_len, cyclic_ini=0):
    """
    Function for increment point data selection.
    :param cols_x: top transverse displacement vector.
    :param cols_y: base reaction moment vector.
    :param mem_len: total/half member length.
    :param cyclic_ini: index indicating the initiation of cyclic loading step.
    :return: final selected indices in the cyclic loading step (not in the whole steps).
    """
    # Make list of the input arrays
    cols_x, cols_y = list(cols_x), list(cols_y)
    # Calculate maximum ranges for drift and moment
    range_x = max(cols_x) - min(cols_x)
    range_y = max(cols_y) - min(cols_y)
    # Get the cyclic step data
    cycle_x = cols_x[cyclic_ini:]
    cycle_y = cols_y[cyclic_ini:]
    cycle_i = list(range(len(cycle_x)))
    # Separate the half cycles
    reversal_index = 0
    cycle_init_index = 0
    cycles_x_list = []
    cycles_y_list = []
    cycles_i_list = []
    for q in range(1, len(cycle_x)):
        if q == len(cycle_x) - 1:
            cycle_x_list = list(cycle_x[cycle_init_index:])
            cycle_y_list = list(cycle_y[cycle_init_index:])
            cycle_i_list = list(cycle_i[cycle_init_index:])
            cycles_x_list.append(cycle_x_list)
            cycles_y_list.append(cycle_y_list)
            cycles_i_list.append(cycle_i_list)
        elif reversal_index == 0 and cycle_x[q] < cycle_x[q - 1]:
            cycle_x_list = list(cycle_x[cycle_init_index: q])
            cycle_y_list = list(cycle_y[cycle_init_index: q])
            cycle_i_list = list(cycle_i[cycle_init_index: q])
            cycle_init_index = q - 1
            reversal_index = 1
            cycles_x_list.append(cycle_x_list)
            cycles_y_list.append(cycle_y_list)
            cycles_i_list.append(cycle_i_list)
        elif reversal_index == 1 and cycle_x[q] > cycle_x[q - 1]:
            cycle_x_list = list(cycle_x[cycle_init_index: q])
            cycle_y_list = list(cycle_y[cycle_init_index: q])
            cycle_i_list = list(cycle_i[cycle_init_index: q])
            cycle_init_index = q - 1
            reversal_index = 0
            cycles_x_list.append(cycle_x_list)
            cycles_y_list.append(cycle_y_list)
            cycles_i_list.append(cycle_i_list)
    # Get arcs list
    slope_control = 2
    arcs_list = []
    plastic_list = []
    for i in range(len(cycles_x_list)):
        # Current cycle
        cycle_x_list = cycles_x_list[i]
        cycle_y_list = cycles_y_list[i]
        # Sub-lists in the format of [0, arc1, arc1 + arc2, ..., total arc length]
        arc_list = [0]
        # List in the format of [slope1, slope2, ..., slope n]
        slope_list = []
        arc = 0
        for j in range(1, len(cycle_x_list)):
            # calculate normalized arc list
            arc = ((cycle_x_list[j] / range_x - cycle_x_list[j - 1] / range_x) ** 2 +
                   (cycle_y_list[j] / range_y - cycle_y_list[j - 1] / range_y) ** 2) ** 0.5 + arc
            arc_list.append(arc)
            # calculate stiffness
            slope = (cycle_y_list[j] - cycle_y_list[j - 1]) / (cycle_x_list[j] - cycle_x_list[j - 1]) if (cycle_x_list[j] - cycle_x_list[j - 1]) != 0 else 0
            if slope != 0.:
                slope_list.append(slope)
        # Assemble sub-lists
        arcs_list.append(arc_list)
        if max(slope_list) / min(slope_list) > slope_control or max(slope_list) / min(slope_list) < 0:
            plastic_list.append(i)
    # Get plastic initiation cycle index
    # Plastic initiation starts when there are consecutively true_cycle_factor number of plastic half cycles
    cycle_factor = 4
    len_plastic = len(plastic_list)
    plastic_init = -1
    for i in range(len_plastic):
        if len_plastic - i > cycle_factor:
            true_cycle_factor = cycle_factor
        else:
            true_cycle_factor = len_plastic - i - 1
        for k in range(true_cycle_factor):
            if plastic_list[i + k + 1] != plastic_list[i] + k + 1:
                plastic_init = -1
                break
            else:
                plastic_init = i
        if plastic_init != -1:
            break
    # point_num is the number of points inside half cycle (two end points are not included)
    point_num = 3
    final_i_list = []
    reversal_index = 0
    for i in range(len(arcs_list)):
        # moment list in one half cycle
        cycle_i_list = cycles_i_list[i]
        cycle_y_list = cycles_y_list[i]
        # if elastic stage
        if plastic_list == [] or i < plastic_list[plastic_init]:
            final_i_list.append(cycle_i_list[0])
        # if plastic stage
        else:
            # arc array in one half cycle
            arc_list = np.array(arcs_list[i])
            # total arc length of the half cycle
            total_arc_length = arc_list[-1]
            # arc lengths that is to be recorded
            arc_length_list = []
            for k in range(point_num + 1):
                arc_length_list.append(total_arc_length / (point_num + 1) * k)
            # indices that is to be recorded
            index_list = []
            for k in range(point_num + 1):
                index = np.argmin(abs(arc_list - arc_length_list[k]))
                index_list.append(cycle_i_list[index])
            # Add the point with the largest moment value in each half cycle
            if reversal_index == 0:
                minmax_ind = np.argmax(np.array(cycle_y_list))
            else:
                minmax_ind = np.argmin(np.array(cycle_y_list))
            if cycle_i_list[minmax_ind] not in index_list:
                index_list.append(cycle_i_list[minmax_ind])
            for j in index_list:
                final_i_list.append(j)
        # update reversal index
        if reversal_index == 0:
            reversal_index = 1
        else:
            reversal_index = 0
    # Sort and remove duplicate
    final_i_list.append(cycle_i[-1])
    final_i_list = list(set(final_i_list))
    final_i_list.sort()
    # Determine if there are large drift gaps (default 0.04)
    cycle_x = [i / mem_len for i in cycle_x]
    while True:
        start_list = []
        for i in range(1, len(final_i_list)):
            if abs(cycle_x[final_i_list[i]] - cycle_x[final_i_list[i - 1]]) >= 0.04:
                start_list.append(i - 1)
        # Add some more points that shorten drift difference
        if len(start_list) != 0:
            for i in start_list:
                compare_list1 = []
                compare_list2 = []
                for j in range(final_i_list[i], final_i_list[i + 1]):
                    bound1 = abs(cycle_x[j] - cycle_x[final_i_list[i]])
                    bound2 = abs(cycle_x[j] - cycle_x[final_i_list[i + 1]])
                    compare_list1.append(bound1)
                    compare_list2.append(bound2)
                mid_ind = np.argmin(np.abs(np.array(compare_list1) - np.array(compare_list2)))
                final_i_list.append((final_i_list[i] + mid_ind))
        else:
            break
        # Final update
        final_i_list = list(set(final_i_list))
        final_i_list.sort()
    # Return modified final index list
    return final_i_list
