import numpy as np
from operator import attrgetter
import pickle
import os
import gc
from mesh import *
from visualization import *
from odbAccess import *
from abaqus import *
from abaqusConstants import *
from caeModules import *
from functions.init_helper import section_reader, SYS_BEAM_SCALE, SYS_COLUMN_SCALE
from functions.point_selection import point_selection


def divide_groups(lc_array, d, tf, r):
    """
    Divide the end element object set into 3 sets: upper flange, web and lower flange sets.
    :param lc_array: dimension (number of elements in the set, 4), 4 -> 3 xyz coordinates representing centroid + 1 element label
    :param d: section depth.
    :param tf: flange thickness.
    :param r: fillet radius.
    :return: flange_lower, web, flange_upper sets.
    """
    flange_upper = []
    web = []
    flange_lower = []
    for item in lc_array:
        if item[1] > d / 2 - tf - (1 - 2 ** 0.5 / 2) * r:
            flange_upper.append(item)
        elif item[1] < -d / 2 + tf + (1 - 2 ** 0.5 / 2) * r:
            flange_lower.append(item)
        else:
            web.append(item)
    return np.array(flange_lower), np.array(web), np.array(flange_upper)


def calculate_nw(lc_array, tw, r, n_tf):
    """
    Returns half number of elements along length direction (x-) of flange.
    :param lc_array: Sorted component element list, dimension (number of elements in the set, 4).
    :param tw: thickness of web.
    :param r: fillet radius.
    :param n_tf: number of element layers at flange.
    :return: half number of elements along length direction (x-) of flange.
    """
    a = 0
    for item in lc_array:
        if item[0] > tw / 2 + r:
            a += 1
    return int(a / n_tf / 2)


def func1(mat):
    """
    Helper function.
    """
    mat = np.transpose(mat, (2, 1, 0, 3))
    label_map = np.array([])
    for x in range(mat.shape[2]):
        for y in range(mat.shape[1]):
            label_map = np.append(label_map, mat[:, y, x, 3])
    return label_map


def get_label_matrix(instance, all_nodes, bt_index, d, tf, tw, r, n_tf=3, n_tw=3):
    """
    Return the label map and mesh params.
    :param instance: instance object.
    :param all_nodes: all of the nodes in the model.
    :param bt_index: 0 denotes bottom segment and 1 denotes top segment.
    :param d: section depth.
    :param tf: flange thickness.
    :param tw: web thickness.
    :param r: fillet radius.
    :param n_tf: number of element layers at flange.
    :param n_tw: number of element layers at web.
    :return: label_maps: organized label matrix, dimension, (3, n_t x n_l x long_direction_number)
             n_l: number of element along length (z-) direction.
             n_h: number of elements at web.
             n_w: half number of elements along length direction (x-) of flange.
             n_k: number of elements at arc.
             n_tf: number of element layers at flange.
             n_tw: number of element layers at web.
    """
    if bt_index == 0:
        name_ = "S1"
        order_ind = 0
    else:
        name_ = "S2"
        order_ind = -1
    eleset_keys = instance.elementSets.keys()
    eleset_numbers = []
    n_l = len(instance.elementSets[name_ + 'E0'].elements)
    for eleset_name in eleset_keys:
        if eleset_name[0:2] == name_ and eleset_name[2] == "E":
            eleset_numbers.append(int(eleset_name[3:]))
    enum = len(eleset_numbers)
    end_face_ele = np.array(np.zeros((0, 4)))
    all_ele = np.array(np.zeros((0, n_l, 4)))
    for i in range(enum):
        centroid_list = np.array(np.zeros((0, 4)))
        for j in range(n_l):
            node_list = instance.elementSets[name_ + 'E%d' % i].elements[j].connectivity
            lb = instance.elementSets[name_ + 'E%d' % i].elements[j].label
            centroid = np.array([0., 0., 0.])
            for k in node_list:
                centroid += np.array(all_nodes[k - 1].coordinates)
            centroid = centroid / 20.
            centroid = np.append(centroid, lb).reshape((1, -1))
            centroid_list = np.concatenate((centroid_list, centroid), axis=0)
        centroid_list = np.array(sorted(centroid_list, key=lambda v: v[2]))
        end_face_ele = np.concatenate((end_face_ele, centroid_list[order_ind].reshape((1, -1))), axis=0)
        all_ele = np.concatenate((all_ele, centroid_list.reshape((1, n_l, 4))), axis=0)
    n_w = calculate_nw(end_face_ele, tw, r, n_tf)
    uf, mw, lf = divide_groups(end_face_ele, d, tf, r)
    n_k = len(uf) / n_tf - 2 * n_w - n_tw
    n_h = len(mw) / n_tw - n_k
    uf_ele = all_ele[:len(uf)].reshape((len(uf) / n_tf, n_tf, n_l, 4))
    mw_ele = all_ele[len(uf):len(uf)+len(mw)].reshape((len(mw) / n_tw, n_tw, n_l, 4))
    lf_ele = all_ele[len(uf)+len(mw):].reshape((len(lf) / n_tf, n_tf, n_l, 4))
    label_map1 = func1(uf_ele)
    label_map2 = func1(mw_ele)
    label_map3 = func1(lf_ele)
    label_map = [label_map1, label_map2, label_map3]
    return label_map, n_l, n_h, n_w, n_k, n_tf, n_tw


def get_integration_label_matrix(odb, initial_step, label_maps, len_beam_ele):
    """
    Return an integration point label matrix corresponding to the given label_maps.
    :param odb: odb file.
    :param initial_step: the name of initial step.
    :param label_maps: organized label matrix, dimension, (3, n_t x n_l x long_direction_number).
    :param len_beam_ele: number of beam elements.
    :return: organized integration point label matrix, dimension (3, number of elements, 8).
    """
    all_node_coord = odb.steps[initial_step].frames[0].fieldOutputs['COORD'].getSubset(position=INTEGRATION_POINT)
    # Dimension (number of elements x 8, 3)
    coord = np.array(map(attrgetter('data'), all_node_coord.values))[:-len_beam_ele]
    integration_label_maps = []
    # (3, n_t x n_l x long_direction_number) -> (n_t x n_l x long_direction_number, )
    for label_map in label_maps:
        # n x 8 matrix, n denotes number of elements,
        # [3, 7, 6, 2, 1, 5, 4, 0], denotes (4+1)th output of abaqus should be placed at 7th in the new system
        integration_label_map = []
        # Each element label
        for k in label_map:
            # Find the element's integration points' coordinates, dimension (8, 3)
            integration_coord = coord[(k - 1) * 8: k * 8]
            # Dimension (1, 3)
            median_coord = np.median(integration_coord, axis=0)
            # Dimension (8, 3)
            sign_matrix = (np.sign(integration_coord - median_coord)).tolist()
            # Put the integration points in the organized way
            integration_label = [sign_matrix.index([-1., -1., -1.]), sign_matrix.index([1., -1., -1.]),
                                 sign_matrix.index([1., 1., -1.]), sign_matrix.index([-1., 1., -1.]),
                                 sign_matrix.index([-1., -1., 1.]), sign_matrix.index([1., -1., 1.]),
                                 sign_matrix.index([1., 1., 1.]), sign_matrix.index([-1., 1., 1.])]
            # Dimension (number of elements, 8)
            integration_label_map.append(integration_label)
        # Dimension (3, number of elements, 8)
        integration_label_maps.append(integration_label_map)
    return integration_label_maps


def unwrap(map_array):
    """
    Turn the list into numpy array, and combine three components together.
    :param map_array: label map list (list of list or list of numpy array).
    :return: pure numpy array.
    """
    x = []
    for imap in map_array:
        x = x + list(imap)
    return np.array(x)


def assemble_mat(label_map, int_label_map):
    """
    Get flattened integration label map.
    :param label_map: top or bottom element label map, dimension, (total number of elements, ).
    :param int_label_map: the corresponding integration point label map, dimension, (total number of elements, 8).
    :return: flattened integration label map, dimension, (total number of elements x 8, ).
    """
    if len(label_map) != len(int_label_map):
        raise ValueError("Two dimensions are not equal!")
    new_array = np.array([])
    for label, int_label in zip(label_map, int_label_map):
        new_array = np.append(new_array, (label - 1) * 8 + int_label)
    return new_array.flatten().astype(int)


def pickle_write(data, pkl_folder, job_name, mat_name, decimal):
    """
    Save the given matrix into pickle files.
    :param data: the data matrix to be saved.
    :param pkl_folder: pickle file storage folder.
    :param job_name: job name.
    :param mat_name: matrix type name.
    :param decimal: decimal value.
    """
    with open(os.path.join(pkl_folder, job_name + mat_name + ".pkl"), "wb") as f:
        data = np.round(data, decimal)
        pickle.dump(data, f)


def get_node_label(instance, n_l, hf_index):
    """
    Obtain node label matrices.
    :param instance: the element instance.
    :param n_l: number of element along length (z-) direction.
    :param hf_index: indicator of full- or half- model, with 0 meaning full-model and 1 denoting half-model.
    :return: final_cnode_label: centerline node label matrix, dimension (number of centerline nodes, ).
             final_pnode_label: profile node label matrix, dimension (number of profile nodes, ).
    """
    nodeset_keys = instance.nodeSets.keys()
    cnodeset_numbers = []
    pnodeset_numbers = []
    for nodeset_name in nodeset_keys:
        if nodeset_name[0] == "S" and nodeset_name[2] == "C":
            cnodeset_numbers.append(int(nodeset_name[3:]))
        if nodeset_name[0] == "S" and nodeset_name[2] == "P":
            pnodeset_numbers.append(int(nodeset_name[3:]))
    # Get the maximum node set numbers and the capacity of each node set
    cmax = max(cnodeset_numbers)
    pmax = max(pnodeset_numbers)
    cnum = n_l + 1
    pnum = n_l * 2 + 1
    # member type 0 means full-model, while 1 means half-model
    if hf_index == 0:
        opt_list = [1, 2]
    elif hf_index == 1:
        opt_list = [1]
    else:
        raise ValueError
    # Get final centerline node label matrix
    final_cnode_label = np.zeros(0)
    for c in opt_list:
        for i in range(cmax + 1):
            nodeset = instance.nodeSets['S%dC%d' % (c, i)]
            z_coord_list = []
            for j in range(cnum):
                z_coord_list.append([nodeset.nodes[j].label, nodeset.nodes[j].coordinates[2]])
            z_coord_list.sort(key=lambda x: x[1], reverse=False)
            label_array = np.array(z_coord_list)[:, 0].astype(int)
            final_cnode_label = np.append(final_cnode_label, label_array)
    # Get final profile node label matrix
    final_pnode_label = np.zeros(0)
    for c in opt_list:
        for i in range(pmax + 1):
            nodeset = instance.nodeSets['S%dP%d' % (c, i)]
            z_coord_list = []
            for j in range(pnum):
                z_coord_list.append([nodeset.nodes[j].label, nodeset.nodes[j].coordinates[2]])
            z_coord_list.sort(key=lambda x: x[1], reverse=False)
            label_array = np.array(z_coord_list)[:, 0]
            final_pnode_label = np.append(final_pnode_label, label_array)
    return (final_cnode_label - 1).astype(int), (final_pnode_label - 1).astype(int)


def get_css_pc_main_matrices(odb, len_end_solid_elements, final_cnode_label, final_pnode_label, hf_index, bc_index, selected_frames, label_matrix, pkl_folder, name, scale):
    """
    Get and save node and element integration point coord, stress and strain matrices.
    :param odb: odb file object.
    :param len_end_solid_elements: number of solid elements on one end.
    :param final_cnode_label: centerline node label matrix, dimension (number of centerline nodes, ).
    :param final_pnode_label: profile node label matrix, dimension (number of profile nodes, ).
    :param hf_index: index indicating the model is full- or half- model, full- 0, half- 1.
    :param bc_index: index indicating the member is beam or column, column 0, beam 1.
    :param selected_frames: selected frame indices in the cyclic loading step.
    :param label_matrix: final label matrix.
    :param pkl_folder: pkl folder name.
    :param name: job name.
    :param scale: unit scale factor used.
    """
    # Column model
    if bc_index == 0:
        opt_list = [['Step-Axial', [0]], ['Step-Cyclic', selected_frames]]
        frame1 = len(odb.steps['Step-Axial'].frames)
        preserved_indices = [0]
        for i in selected_frames:
            preserved_indices.append(frame1 + i)
    # Beam model
    elif bc_index == 1:
        opt_list = [['Step-Cyclic', selected_frames]]
        preserved_indices = selected_frames
    else:
        raise ValueError("bc_index should be either 0 or 1!")
    # Full-model
    if hf_index == 0:
        ne = 2
        len_solid_elements = len_end_solid_elements * 2
    # Half-model
    elif hf_index == 1:
        ne = 1
        len_solid_elements = len_end_solid_elements
    else:
        raise ValueError("hf_index should be either 0 or 1!")
    # Start iteration
    num_int = len_solid_elements * 8
    final_integration_coord_matrix = np.zeros((num_int, 0))
    final_integration_stress_matrix = np.zeros((num_int, 0))
    final_integration_strain_matrix = np.zeros((num_int, 0))
    final_cnode_coord_matrix = np.zeros((len(final_cnode_label), 0))
    final_pnode_coord_matrix = np.zeros((len(final_pnode_label), 0))
    for step_name, index_list in opt_list:
        for nn in index_list:
            print(step_name, nn)
            # Get element and node coordinate, stress, strain data
            frame = odb.steps[step_name].frames[nn]
            integration_coord = frame.fieldOutputs['COORD'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            integration_s = frame.fieldOutputs['S'].getSubset(position=INTEGRATION_POINT).values
            integration_pe11 = frame.fieldOutputs['SDV2'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            integration_pe22 = frame.fieldOutputs['SDV3'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            integration_pe33 = frame.fieldOutputs['SDV4'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            integration_pe12 = frame.fieldOutputs['SDV5'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            integration_pe13 = frame.fieldOutputs['SDV6'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            integration_pe23 = frame.fieldOutputs['SDV7'].getSubset(position=INTEGRATION_POINT).bulkDataBlocks
            node_coord = frame.fieldOutputs['COORD'].getSubset(position=NODAL).bulkDataBlocks
            # Processing
            integration_coord_matrix = np.copy(integration_coord[0].data)
            integration_stress_matrix = np.array(map(attrgetter('maxPrincipal', 'midPrincipal', 'minPrincipal', 'mises'),
                                                     integration_s)[:num_int]).reshape((num_int, 4))
            integration_pe11_matrix = np.copy(integration_pe11[0].data)
            integration_pe22_matrix = np.copy(integration_pe22[0].data)
            integration_pe33_matrix = np.copy(integration_pe33[0].data)
            integration_pe12_matrix = np.copy(integration_pe12[0].data)
            integration_pe13_matrix = np.copy(integration_pe13[0].data)
            integration_pe23_matrix = np.copy(integration_pe23[0].data)
            node_coord_matrix = np.copy(node_coord[0].data)
            cnode_coord_matrix = node_coord_matrix[final_cnode_label]
            pnode_coord_matrix = node_coord_matrix[final_pnode_label]
            # Get integration coord, stress and strain matrices
            final_integration_coord_matrix = np.concatenate((final_integration_coord_matrix, integration_coord_matrix), axis=1)
            final_integration_stress_matrix = np.concatenate((final_integration_stress_matrix, integration_stress_matrix), axis=1)
            final_integration_strain_matrix = np.concatenate((final_integration_strain_matrix,
                                                              integration_pe11_matrix, integration_pe22_matrix,
                                                              integration_pe33_matrix, integration_pe12_matrix,
                                                              integration_pe13_matrix, integration_pe23_matrix), axis=1)
            # Get centerline and profile node matrices
            final_cnode_coord_matrix = np.concatenate((final_cnode_coord_matrix, cnode_coord_matrix), axis=1)
            final_pnode_coord_matrix = np.concatenate((final_pnode_coord_matrix, pnode_coord_matrix), axis=1)
    # Reshape
    final_integration_coord_matrix = final_integration_coord_matrix[label_matrix].reshape((ne, len_end_solid_elements, 8, len(preserved_indices), 3))
    final_integration_stress_matrix = final_integration_stress_matrix[label_matrix].reshape((ne, len_end_solid_elements, 8, len(preserved_indices), 4))
    final_integration_strain_matrix = final_integration_strain_matrix[label_matrix].reshape((ne, len_end_solid_elements, 8, len(preserved_indices), 6))
    # Transpose
    final_integration_coord_matrix = final_integration_coord_matrix.transpose((3, 0, 1, 2, 4))
    final_integration_stress_matrix = final_integration_stress_matrix.transpose((3, 0, 1, 2, 4))
    final_integration_strain_matrix = final_integration_strain_matrix.transpose((3, 0, 1, 2, 4))
    # Reshape
    final_cnode_coord_matrix = final_cnode_coord_matrix.reshape((ne, -1, len(preserved_indices), 3))
    final_pnode_coord_matrix = final_pnode_coord_matrix.reshape((ne, -1, len(preserved_indices), 3))
    # Transpose
    final_cnode_coord_matrix = final_cnode_coord_matrix.transpose((2, 0, 1, 3))
    final_pnode_coord_matrix = final_pnode_coord_matrix.transpose((2, 0, 1, 3))
    # Save matrices
    pickle_write(final_integration_coord_matrix * scale, pkl_folder, name, '-ele_coord', 3)
    del final_integration_coord_matrix
    gc.collect()
    pickle_write(final_integration_stress_matrix / scale ** 2, pkl_folder, name, '-ele_stress', 3)
    del final_integration_stress_matrix
    gc.collect()
    pickle_write(final_integration_strain_matrix, pkl_folder, name, '-ele_strain', 6)
    del final_integration_strain_matrix
    gc.collect()
    pickle_write(final_cnode_coord_matrix * scale, pkl_folder, name, '-center_node', 3)
    del final_cnode_coord_matrix
    gc.collect()
    pickle_write(final_pnode_coord_matrix * scale, pkl_folder, name, '-profile_node', 3)
    del final_pnode_coord_matrix
    gc.collect()
    pickle_write(np.array(preserved_indices), pkl_folder, name, '-preserved_indices', 0)


def save_beam_info(odb, len_beam_nodes, selected_frames, pkl_folder, name, bc_index, scale):
    """
    Get and save beam node information.
    :param odb: odb file.
    :param len_beam_nodes: number of beam nodes.
    :param selected_frames: selected frame indices in the cyclic loading step.
    :param pkl_folder: pkl folder name.
    :param name: job name.
    :param bc_index: index indicating the member is beam or column, column 0, beam 1.
    :param scale: unit scale factor used.
    """
    #
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    data = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL), ('UR', NODAL),), nodeSets=("PART-1-1.BEAM-ALL",))
    data_coord1 = np.array(data[len_beam_nodes: len_beam_nodes * 2])[:, :, 1].reshape((len_beam_nodes, -1, 1))
    data_coord2 = np.array(data[len_beam_nodes * 2: len_beam_nodes * 3])[:, :, 1].reshape((len_beam_nodes, -1, 1))
    data_coord3 = np.array(data[len_beam_nodes * 3: len_beam_nodes * 4])[:, :, 1].reshape((len_beam_nodes, -1, 1))
    data_rotation1 = np.array(data[len_beam_nodes * 5: len_beam_nodes * 6])[:, :, 1].reshape((len_beam_nodes, -1, 1))
    data_rotation2 = np.array(data[len_beam_nodes * 6: len_beam_nodes * 7])[:, :, 1].reshape((len_beam_nodes, -1, 1))
    data_rotation3 = np.array(data[len_beam_nodes * 7:])[:, :, 1].reshape((len_beam_nodes, -1, 1))
    coord_bnode_matrix = np.concatenate((data_coord1, data_coord2, data_coord3), axis=2)
    coord_rnode_matrix = np.concatenate((data_rotation1, data_rotation2, data_rotation3), axis=2)
    coord_rnode_matrix = coord_rnode_matrix[coord_bnode_matrix[:, 0, 2].argsort()]
    coord_bnode_matrix = coord_bnode_matrix[coord_bnode_matrix[:, 0, 2].argsort()]
    coord_bnode_matrix = coord_bnode_matrix.transpose((1, 0, 2))
    coord_rnode_matrix = coord_rnode_matrix.transpose((1, 0, 2))
    # If column member
    if bc_index == 0:
        frame1 = len(odb.steps['Step-Axial'].frames)
        preserved_indices = [0]
        for i in selected_frames:
            preserved_indices.append(frame1 + i)
        preserved_indices = np.array(preserved_indices).astype(int)
    # If beam member
    elif bc_index == 1:
        preserved_indices = np.array(selected_frames).astype(int)
    else:
        raise ValueError
    coord_bnode_matrix = coord_bnode_matrix[preserved_indices, :, :]
    coord_rnode_matrix = coord_rnode_matrix[preserved_indices, :, :]
    pickle_write(coord_bnode_matrix * scale, pkl_folder, name, '-beam_coord', 3)
    pickle_write(coord_rnode_matrix, pkl_folder, name, '-beam_rotation', 6)


def save_macro_data(odb, steps, pkl_folder, name, hf_index, scale):
    """
    Function for saving macro data such as displacement and moment, and return top transverse displacement and bottom moment.
    :param odb: odb file object.
    :param steps: all the step names in the model.
    :param pkl_folder: pkl folder name.
    :param name: job name.
    :param hf_index: index indicating the model is full- or half- model, full- 0, half- 1.
    :param scale: unit scale factor used.
    :return: top transverse displacement top_U2 and bottom reaction moment bot_RM1.
    """
    bot_set = "BOT"
    if hf_index == 0:
        top_set = "TOP"
    elif hf_index == 1:
        top_set = "BEAM-TOP"
    else:
        raise ValueError
    bot_node = odb.rootAssembly.instances["PART-1-1"].nodeSets[bot_set].nodes[0]
    top_node = odb.rootAssembly.instances["PART-1-1"].nodeSets[top_set].nodes[0]
    # Read history data
    top_U2 = session.XYDataFromHistory(name='TU2', odb=odb,
                                       outputVariableName='Spatial displacement: U2 at Node ' + str(
                                           top_node.label) + ' in NSET ' + top_set, steps=steps)
    top_U3 = session.XYDataFromHistory(name='TU3', odb=odb,
                                       outputVariableName='Spatial displacement: U3 at Node ' + str(
                                           top_node.label) + ' in NSET ' + top_set, steps=steps)
    top_R1 = session.XYDataFromHistory(name='TUR1', odb=odb,
                                       outputVariableName='Rotational displacement: UR1 at Node ' + str(
                                           top_node.label) + ' in NSET ' + top_set, steps=steps)
    bot_RF2 = session.XYDataFromHistory(name='BRF2', odb=odb,
                                        outputVariableName='Reaction force: RF2 at Node ' + str(
                                            bot_node.label) + ' in NSET ' + bot_set, steps=steps)
    bot_RM1 = session.XYDataFromHistory(name='BRM1', odb=odb,
                                        outputVariableName='Reaction moment: RM1 at Node ' + str(
                                            bot_node.label) + ' in NSET ' + bot_set, steps=steps)
    # Reshape data
    time = np.array(top_U2)[:, 0].reshape((-1, ))
    top_U2 = np.array(top_U2)[:, 1].reshape((-1, ))
    top_U3 = np.array(top_U3)[:, 1].reshape((-1, ))
    top_R1 = np.array(top_R1)[:, 1].reshape((-1, ))
    bot_RF2 = np.array(bot_RF2)[:, 1].reshape((-1, ))
    bot_RM1 = np.array(bot_RM1)[:, 1].reshape((-1, ))
    # Save pickle files
    pickle_write(time, pkl_folder, name, '-time', 5)
    pickle_write(top_U2 * scale, pkl_folder, name, '-top_U2', 3)
    pickle_write(top_U3 * scale, pkl_folder, name, '-top_U3', 3)
    pickle_write(top_R1, pkl_folder, name, '-top_R1', 6)
    pickle_write(bot_RF2, pkl_folder, name, '-bot_RF2', 2)
    pickle_write(bot_RM1 * scale, pkl_folder, name, '-bot_RM1', 2)
    # Return two vectors
    return top_U2 * scale, bot_RM1 * scale


def fb_model_result_reader(odb, main_folder, pkl_folder, name, section_name):
    """
    Data extraction from odb file (suitable for full-beam).
    :param odb: odb file.
    :param main_folder: main folder.
    :param pkl_folder: pkl file folder.
    :param name: job name.
    :param section_name: section name.
    """
    # Read geometrical params
    d, w, tf, tw, r, _ = section_reader(main_folder, section_name, SYS_BEAM_SCALE)
    # Get element and node objects
    instance = odb.rootAssembly.instances["PART-1-1"]
    all_elements = instance.elements
    all_nodes = instance.nodes
    beam_nodes = instance.nodeSets["BEAM-ALL"].nodes
    # Get element and node numbers
    len_beam_nodes = len(beam_nodes)
    len_beam_elements = len_beam_nodes - 1
    len_solid_elements = len(all_elements) - len_beam_elements
    # Get number of elements on top or bottom end
    boundary_top_bot = len_solid_elements / 2
    if type(boundary_top_bot) == float:
        raise TypeError('Problem with element numbers!')
    # Save macro data such as displacement, moment, force, return bottom moment and top displacement
    scaled_top_u2, scaled_bot_rm1 = save_macro_data(odb, ('Step-Cyclic',), pkl_folder, name, 0, SYS_BEAM_SCALE)
    selected_cyclic = point_selection(scaled_top_u2, scaled_bot_rm1, float(name.split('-')[2]), cyclic_ini=0)
    # Get element and integration point label maps
    label_maps_bot, n_l, n_h, n_w, n_k, n_tf, n_tw = get_label_matrix(instance, all_nodes, 0, d, tf, tw, r)
    integration_label_maps_bot = get_integration_label_matrix(odb, "Step-Cyclic", label_maps_bot, len_beam_elements)
    label_maps_top, _, _, _, _, _, _ = get_label_matrix(instance, all_nodes, 1, d, tf, tw, r)
    integration_label_maps_top = get_integration_label_matrix(odb, "Step-Cyclic", label_maps_top, len_beam_elements)
    # Unwrap the lists to numpy arrays
    label_maps_bot = unwrap(label_maps_bot)
    integration_label_maps_bot = unwrap(integration_label_maps_bot)
    label_maps_top = unwrap(label_maps_top)
    integration_label_maps_top = unwrap(integration_label_maps_top)
    # Get final integration point label maps
    final_label_mat_bot = assemble_mat(label_maps_bot, integration_label_maps_bot)
    final_label_mat_top = assemble_mat(label_maps_top, integration_label_maps_top)
    final_label_mat = np.concatenate((final_label_mat_bot, final_label_mat_top))
    # Get centerline and profile node matrices
    instance = odb.rootAssembly.instances["PART-1-1"]
    final_cnode_label, final_pnode_label = get_node_label(instance, n_l, 0)
    # Get and save element and node coord, stress and strain info
    get_css_pc_main_matrices(odb, boundary_top_bot, final_cnode_label, final_pnode_label, 0, 1, selected_cyclic, final_label_mat, pkl_folder, name, SYS_BEAM_SCALE)
    # Get and save beam node coord and rotation info
    save_beam_info(odb, len_beam_nodes, selected_cyclic, pkl_folder, name, 1, SYS_BEAM_SCALE)
    # Save attrs
    attrs = np.array([n_tf, n_w, n_k, n_tw, n_h, n_l, d * SYS_BEAM_SCALE, w * SYS_BEAM_SCALE, tf * SYS_BEAM_SCALE, tw * SYS_BEAM_SCALE, r * SYS_BEAM_SCALE]).reshape((11, ))
    pickle_write(attrs, pkl_folder, name, '-params', 3)


def hb_model_result_reader(odb, main_folder, pkl_folder, name, section_name):
    """
    Data extraction from odb file (suitable for half-beam).
    :param odb: odb file.
    :param main_folder: main folder.
    :param pkl_folder: pkl file folder.
    :param name: job name.
    :param section_name: section name.
    """
    # Read geometrical params
    d, w, tf, tw, r, _ = section_reader(main_folder, section_name, SYS_BEAM_SCALE)
    # Get element and node objects
    instance = odb.rootAssembly.instances["PART-1-1"]
    all_elements = instance.elements
    all_nodes = instance.nodes
    beam_nodes = instance.nodeSets["BEAM-ALL"].nodes
    # Get element and node numbers
    len_beam_nodes = len(beam_nodes)
    len_beam_elements = len_beam_nodes - 1
    len_solid_elements = len(all_elements) - len_beam_elements
    # Save macro data such as displacement, moment, force, return bottom moment and top displacement
    scaled_top_u2, scaled_bot_rm1 = save_macro_data(odb, ('Step-Cyclic',), pkl_folder, name, 1, SYS_BEAM_SCALE)
    selected_cyclic = point_selection(scaled_top_u2, scaled_bot_rm1, float(name.split('-')[2])/2, cyclic_ini=0)
    # Get element and integration point label maps
    label_maps, n_l, n_h, n_w, n_k, n_tf, n_tw = get_label_matrix(instance, all_nodes, 0, d, tf, tw, r)
    integration_label_maps = get_integration_label_matrix(odb, "Step-Cyclic", label_maps, len_beam_elements)
    # Unwrap the lists to numpy arrays
    label_maps = unwrap(label_maps)
    integration_label_maps = unwrap(integration_label_maps)
    # Get final integration point label maps
    final_label_mat = assemble_mat(label_maps, integration_label_maps)
    # Get centerline and profile node matrices
    instance = odb.rootAssembly.instances["PART-1-1"]
    final_cnode_label, final_pnode_label = get_node_label(instance, n_l, 1)
    # Get and save element and node coord, stress and strain info
    get_css_pc_main_matrices(odb, len_solid_elements, final_cnode_label, final_pnode_label, 1, 1, selected_cyclic, final_label_mat, pkl_folder, name, SYS_BEAM_SCALE)
    # Get and save beam node coord and rotation info
    save_beam_info(odb, len_beam_nodes, selected_cyclic, pkl_folder, name, 1, SYS_BEAM_SCALE)
    # Save attrs
    attrs = np.array([n_tf, n_w, n_k, n_tw, n_h, n_l, d * SYS_BEAM_SCALE, w * SYS_BEAM_SCALE, tf * SYS_BEAM_SCALE, tw * SYS_BEAM_SCALE, r * SYS_BEAM_SCALE]).reshape((11, ))
    pickle_write(attrs, pkl_folder, name, '-params', 3)


def cl_model_result_reader(odb, main_folder, pkl_folder, name, section_name):
    """
    Data extraction from odb file (suitable for column).
    :param odb: odb file.
    :param main_folder: main folder.
    :param pkl_folder: pkl file folder.
    :param name: job name.
    :param section_name: section name.
    """
    # Read geometrical params
    d, w, tf, tw, r, _ = section_reader(main_folder, section_name, SYS_COLUMN_SCALE)
    # Obtain frame numbers
    num_frames1 = len(odb.steps['Step-Axial'].frames)
    # Get element and node objects
    if name.split('-')[-1] == "Fe":
        all_elements = odb.rootAssembly.instances['PART-1-1'].elements[:-1]
    elif name.split('-')[-1] == "Fx":
        all_elements = odb.rootAssembly.instances['PART-1-1'].elements
    else:
        raise NameError
    instance = odb.rootAssembly.instances["PART-1-1"]
    all_nodes = instance.nodes
    beam_nodes = instance.nodeSets["BEAM-ALL"].nodes
    # Get element and node numbers
    len_beam_nodes = len(beam_nodes)
    len_beam_elements = len_beam_nodes - 1
    len_solid_elements = len(all_elements) - len_beam_elements
    # Get number of elements on top or bottom end
    boundary_top_bot = len_solid_elements / 2
    if type(boundary_top_bot) == float:
        raise TypeError('Problem with element numbers!')
    # Save macro data such as displacement, moment, force, return bottom moment and top displacement
    scaled_top_u2, scaled_bot_rm1 = save_macro_data(odb, ('Step-Axial', 'Step-Cyclic',), pkl_folder, name, 0, SYS_COLUMN_SCALE)
    selected_cyclic = point_selection(scaled_top_u2, scaled_bot_rm1, float(name.split('-')[2]), cyclic_ini=num_frames1)
    # Get element and integration point label maps
    label_maps_bot, n_l, n_h, n_w, n_k, n_tf, n_tw = get_label_matrix(instance, all_nodes, 0, d, tf, tw, r)
    integration_label_maps_bot = get_integration_label_matrix(odb, "Step-Axial", label_maps_bot, len_beam_elements)
    label_maps_top, _, _, _, _, _, _ = get_label_matrix(instance, all_nodes, 1, d, tf, tw, r)
    integration_label_maps_top = get_integration_label_matrix(odb, "Step-Axial", label_maps_top, len_beam_elements)
    # Unwrap the lists to numpy arrays
    label_maps_bot = unwrap(label_maps_bot)
    integration_label_maps_bot = unwrap(integration_label_maps_bot)
    label_maps_top = unwrap(label_maps_top)
    integration_label_maps_top = unwrap(integration_label_maps_top)
    # Get final integration point label maps
    final_label_mat_bot = assemble_mat(label_maps_bot, integration_label_maps_bot)
    final_label_mat_top = assemble_mat(label_maps_top, integration_label_maps_top)
    final_label_mat = np.concatenate((final_label_mat_bot, final_label_mat_top))
    # Get centerline and profile node matrices
    final_cnode_label, final_pnode_label = get_node_label(instance, n_l, 0)
    # Get and save element and node coord, stress and strain info
    get_css_pc_main_matrices(odb, boundary_top_bot, final_cnode_label, final_pnode_label, 0, 0, selected_cyclic, final_label_mat, pkl_folder, name, SYS_COLUMN_SCALE)
    # Get and save beam node coord and rotation info
    save_beam_info(odb, len_beam_nodes, selected_cyclic, pkl_folder, name, 0, SYS_COLUMN_SCALE)
    # Save attrs
    attrs = np.array([n_tf, n_w, n_k, n_tw, n_h, n_l, d * SYS_COLUMN_SCALE, w * SYS_COLUMN_SCALE, tf * SYS_COLUMN_SCALE, tw * SYS_COLUMN_SCALE, r * SYS_COLUMN_SCALE]).reshape((11, ))
    pickle_write(attrs, pkl_folder, name, '-params', 3)
