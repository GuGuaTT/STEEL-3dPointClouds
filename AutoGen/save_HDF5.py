import pickle
import os
import numpy as np
import h5py
from functions.reserve_capacity import calculate_reserve_capacity


def read_pickle(pkl_location, model_name, name):
    """
    Read data matrix from pkl file.
    :param pkl_location: folder for saving pkl files.
    :param model_name: model name.
    :param dataset name.
    :return data matrix.
    """
    with open(os.path.join(pkl_location, model_name + name + '.pkl'), "rb+") as f:
        mat = pickle.load(f, encoding='bytes')
    return np.array(mat)


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


def save_hdf5(model_set, h5_file, pkl_f, com_type):
    """
    Save hdf5 file from the data in pkl file.
    :param model_set: set of model names.
    :param h5_file: hdf5 filename.
    :param pkl_f: pkl folder.
    """
    for model_name in model_set:
        params = model_name.split('-')
        section_name = params[1]
        col_len = params[2]
        if params[3] == "C":
            loading_protocol = "Collapse_consistent"
        elif params[3] == "S":
            loading_protocol = "Symmetric"
        elif params[3] == "M":
            loading_protocol = "Monotonic"
        else:
            raise ValueError("Wrong loading protocol!")
        # Open folders
        group_section = write_group(h5_file, section_name)
        group_length = write_group(group_section, col_len)
        group_protocol = write_group(group_length, loading_protocol)
        if com_type == "Column":
            axial_load_ratio = params[4][0] + '.' + params[4][1]
            if params[5] == "Fe":
                boundary = "Flexible"
            elif params[5] == "Fx":
                boundary = "Fixed"
            else:
                raise ValueError("Wrong boundary")
            group_load_ratio = write_group(group_protocol, axial_load_ratio)
            group_bs = write_group(group_load_ratio, boundary)
        elif com_type == "Beam":
            support_ratio = params[5]
            group_bs = write_group(group_protocol, support_ratio)
        else:
            raise TypeError("Name should be only column or beam!")
        # Save time and indices
        time_array = read_pickle(pkl_f, model_name, '-time')
        group_bs.create_dataset('time', data=time_array)
        preserved_indices = read_pickle(pkl_f, model_name, '-preserved_indices')
        group_bs.create_dataset('indices', data=preserved_indices)
        # Save attributes
        params_array = read_pickle(pkl_f, model_name, '-params')
        group_bs.attrs['n_tf'] = int(params_array[0])
        group_bs.attrs['n_w'] = int(params_array[1])
        group_bs.attrs['n_k'] = int(params_array[2])
        group_bs.attrs['n_tw'] = int(params_array[3])
        group_bs.attrs['n_h'] = int(params_array[4])
        group_bs.attrs['n_l'] = int(params_array[5])
        group_bs.attrs['d'] = float(params_array[6])
        group_bs.attrs['b_f'] = float(params_array[7])
        group_bs.attrs['t_f'] = float(params_array[8])
        group_bs.attrs['t_w'] = float(params_array[9])
        group_bs.attrs['r'] = float(params_array[10])
        # Create groups
        group_reaction = write_group(group_bs, 'Reaction')
        group_displacement = write_group(group_bs, 'Displacement')
        group_solid_node = write_group(group_bs, 'Solid_node')
        group_solid_element = write_group(group_bs, 'Solid_element')
        group_beam_node = write_group(group_bs, 'Beam_node')
        # Save reactions
        trm1_array = read_pickle(pkl_f, model_name, '-bot_RF2')
        group_reaction.create_dataset('bot_RF2', data=trm1_array)
        brm1_array = read_pickle(pkl_f, model_name, '-bot_RM1')
        group_reaction.create_dataset('bot_RM1', data=brm1_array)
        # Save displacements
        tu2_array = read_pickle(pkl_f, model_name, '-top_U2')
        group_displacement.create_dataset('top_U2', data=tu2_array)
        tu3_array = read_pickle(pkl_f, model_name, '-top_U3')
        group_displacement.create_dataset('top_U3', data=tu3_array)
        tr1_array = read_pickle(pkl_f, model_name, '-top_R1')
        group_displacement.create_dataset('top_R1', data=tr1_array)
        # Save solid node coordinates
        cnode_coord_array = read_pickle(pkl_f, model_name, '-center_node')
        group_solid_node.create_dataset('center_node_coord', data=cnode_coord_array)
        pnode_coord_array = read_pickle(pkl_f, model_name, '-profile_node')
        group_solid_node.create_dataset('profile_node_coord', data=pnode_coord_array)
        # Save integration point information - STRESS AND STRAIN FIELDS OPTIONAL!
        ele_coord_array = read_pickle(pkl_f, model_name, '-ele_coord')
        group_solid_element.create_dataset('ele_coord', data=ele_coord_array)
        ele_stress_array = read_pickle(pkl_f, model_name, '-ele_stress')
        group_solid_element.create_dataset('ele_stress', data=ele_stress_array)
        ele_strain_array = read_pickle(pkl_f, model_name, '-ele_strain')
        group_solid_element.create_dataset('ele_strain', data=ele_strain_array)
        # Save beam node information
        beam_coord_array = read_pickle(pkl_f, model_name, '-beam_coord')
        group_beam_node.create_dataset('beam_node_coord', data=beam_coord_array)
        beam_rotation_array = read_pickle(pkl_f, model_name, '-beam_rotation')
        group_beam_node.create_dataset('beam_node_rotation', data=beam_rotation_array)
        # Save reserve capacity
        rc_array = calculate_reserve_capacity(tu2_array, brm1_array)
        group_reaction.create_dataset('reserve_capacity', data=rc_array)


def save_hdf5_column(model_set, pkl_f, file_name):
    # Open HDF5 file
    h5_file = h5py.File(file_name, "a")
    save_hdf5(model_set, h5_file, pkl_f, 'Column')


def save_hdf5_beam(model_set, pkl_f, file_name):
    # Open HDF5 file
    h5_file = h5py.File(file_name, "a")
    save_hdf5(model_set, h5_file, pkl_f, 'Beam')


if __name__ == "__main__":
    # Folder names
    main_folder = 'D:\\AutoGen'
    pkl_folder = os.path.join(main_folder, 'pkl_file')
    hdf5_folder = 'D:\\AutoGen\\HDF5_file'
    column_file_name = os.path.join(hdf5_folder, "column.hdf5")
    beam_file_name = os.path.join(hdf5_folder, "beam.hdf5")

    # Separate columns and beams
    column_set = []
    beam_set = []
    pkl_names = os.listdir(pkl_folder)
    for pkl_name in pkl_names:
        strip_name = pkl_name[:pkl_name.rindex('-')]
        if strip_name.split('-')[0] == "cl":
            column_set.append(strip_name)
        elif strip_name.split('-')[0] == "fb" or strip_name.split('-')[0] == "hb":
            beam_set.append(strip_name)
        else:
            raise NameError("Name problem!")

    # Remove duplicate
    column_set = list(set(column_set))
    beam_set = list(set(beam_set))

    # Save data
    save_hdf5_column(column_set, pkl_folder, column_file_name)
