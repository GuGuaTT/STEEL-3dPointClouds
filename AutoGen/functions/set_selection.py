from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import *
from visualization import *
from odbAccess import *
from functions.init_helper import *
import numpy as np


Set_code1 = ["nodes1 = instance1.nodes",
             "nodes3 = instance3.nodes",
             "node_bot = nodes1.getByBoundingBox(xMax=tw/12, xMin=-tw/12, yMax=mesh_size_web/4, yMin=-mesh_size_web/4,"
             "zMax=mesh_size_solid/4, zMin=-mesh_size_solid/4)",
             "set1 = Assembly.Set(nodes=node_bot, name='BOT')",
             "set4 = Assembly.Set(nodes=instance1.nodes, name='SOLID-BOT-ALL')",
             "set6 = Assembly.Set(nodes=instance1.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=len_solid+mesh_size_solid/4, zMin=len_solid-mesh_size_solid/4), name='SOLID-BOT-TOP')",
             "set7 = Assembly.Set(nodes=instance3.nodes, name='BEAM-ALL')",
             "set2 = Assembly.Set(nodes=instance3.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=mem_len/2+mesh_size_beam/4, zMin=mem_len/2-mesh_size_beam/4), name='BEAM-TOP')",
             "set9 = Assembly.Set(nodes=instance3.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=len_solid+mesh_size_beam/4, zMin=len_solid-mesh_size_beam/4), name='BEAM-BOT')"]

Set_code2 = ["nodes1 = instance1.nodes",
             "nodes2 = instance2.nodes",
             "nodes3 = instance3.nodes",
             "node_bot = nodes1.getByBoundingBox(xMax=tw/12, xMin=-tw/12, yMax=mesh_size_web/4, yMin=-mesh_size_web/4,"
             "zMax=mesh_size_solid/4, zMin=-mesh_size_solid/4)",
             "node_top = nodes2.getByBoundingBox(xMax=tw/12, xMin=-tw/12, yMax=mesh_size_web/4, yMin=-mesh_size_web/4,"
             "zMax=mem_len+mesh_size_solid/4, zMin=mem_len-mesh_size_solid/4)",
             "set1 = Assembly.Set(nodes=node_bot, name='BOT')",
             "set2 = Assembly.Set(nodes=node_top, name='TOP')",
             "set3 = Assembly.Set(nodes=instance2.nodes, name='SOLID-TOP-ALL')",
             "set4 = Assembly.Set(nodes=instance1.nodes, name='SOLID-BOT-ALL')",
             "set5 = Assembly.Set(nodes=instance2.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=mem_len-len_solid+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4), name='SOLID-TOP-BOT')",
             "set6 = Assembly.Set(nodes=instance1.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=len_solid+mesh_size_solid/4, zMin=len_solid-mesh_size_solid/4), name='SOLID-BOT-TOP')",
             "set7 = Assembly.Set(nodes=instance3.nodes, name='BEAM-ALL')",
             "set8 = Assembly.Set(nodes=instance3.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=mem_len-len_solid+mesh_size_beam/4, zMin=mem_len-len_solid-mesh_size_beam/4), name='BEAM-TOP')",
             "set9 = Assembly.Set(nodes=instance3.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d, yMin=-d,\
             zMax=len_solid+mesh_size_beam/4, zMin=len_solid-mesh_size_beam/4), name='BEAM-BOT')"]


def set_selection_bot(Assembly, nodes1, instance1, w, d, tf, tw, r, len_solid, mesh_size_solid, mesh_size_flange, ef, ew, scale):
    """
    Function for establishing profile and centerline node sets and element sets in the bottom solid segment.
    :param Assembly: model assembly of instances.
    :param nodes1: nodes in the bottom solid segment.
    :param instance1: bottom solid segment instance.
    :param w: section width.
    :param d: section depth.
    :param tf: flange thickness.
    :param tw: web thickness.
    :param r: fillet radius.
    :param len_solid: length of solid element segment.
    :param mesh_size_solid: general mesh size of solid element segment.
    :param mesh_size_flange: mesh size of flange.
    :param ef: number of elements per thickness at flange, 3.
    :param ew: number of elements per thickness at web, 3.
    :param scale: unit scale.
    :return: organized element matrix.
    """
    tol = 1 / scale
    # create element sets
    s3_xy_coord_list = ele_set_define(instance1, Assembly, d, tf, r, ef, ew, 'S1E')
    # bot segments
    nodes1_p9 = instance1.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d/2+tf/12, yMin=d/2-tf/12, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p8 = instance1.nodes.getByBoundingBox(xMax=w/2+mesh_size_flange/4, xMin=w/2-mesh_size_flange/4, yMax=d/2+tf/12, yMin=d/2-r-tf*13./12, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p7 = instance1.nodes.getByBoundingBox(xMax=w, xMin=tw/2+r-tol, yMax=d/2-tf*11./12., yMin=d/2-tf*13./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p6 = instance1.nodes.getByBoundingCylinder(center1=(tw/2+r, d/2-tf-r, -mesh_size_solid/4), center2=(tw/2+r, d/2-tf-r, len_solid+mesh_size_solid/4), radius=r+tol)
    nodes1_p5 = instance1.nodes.getByBoundingBox(xMax=tw*7./12., xMin=tw*5./12., yMax=d/2-r-tf+tol, yMin=-d/2+r+tf-tol, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p4 = instance1.nodes.getByBoundingCylinder(center1=(tw/2+r, -d/2+tf+r, -mesh_size_solid/4), center2=(tw/2+r, -d/2+tf+r, len_solid+mesh_size_solid/4), radius=r+tol)
    nodes1_p3 = instance1.nodes.getByBoundingBox(xMax=w, xMin=tw/2+r-tol, yMax=-d/2+tf*13./12., yMin=-d/2+tf*11./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p2 = instance1.nodes.getByBoundingBox(xMax=w/2+mesh_size_flange/4, xMin=w/2-mesh_size_flange/4, yMax=-d/2+r+tf*13./12, yMin=-d/2-tf/12, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p1 = instance1.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=-d/2+tf/12, yMin=-d/2-tf/12, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p16 = instance1.nodes.getByBoundingBox(xMax=-w/2+mesh_size_flange/4, xMin=-w/2-mesh_size_flange/4, yMax=-d/2+r+tf*13./12, yMin=-d/2-tf/12, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p15 = instance1.nodes.getByBoundingBox(xMax=-tw/2-r+tol, xMin=-w, yMax=-d/2+tf*13./12., yMin=-d/2+tf*11./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p14 = instance1.nodes.getByBoundingCylinder(center1=(-tw/2-r, -d/2+tf+r, -mesh_size_solid/4), center2=(-tw/2-r, -d/2+tf+r, len_solid+mesh_size_solid/4), radius=r+tol)
    nodes1_p13 = instance1.nodes.getByBoundingBox(xMax=-tw*5./12., xMin=-tw*7./12., yMax=d/2-r-tf+tol, yMin=-d/2+r+tf-tol, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p12 = instance1.nodes.getByBoundingCylinder(center1=(-tw/2-r, d/2-tf-r, -mesh_size_solid/4), center2=(-tw/2-r, d/2-tf-r, len_solid+mesh_size_solid/4), radius=r+tol)
    nodes1_p11 = instance1.nodes.getByBoundingBox(xMax=-tw/2-r+tol, xMin=-w, yMax=d/2-tf*11./12., yMin=d/2-tf*13./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_p10 = instance1.nodes.getByBoundingBox(xMax=-w/2+mesh_size_flange/4, xMin=-w/2-mesh_size_flange/4, yMax=d/2+tf/12, yMin=d/2-r-tf*13./12, zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_c1 = instance1.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=-d/2+tf*7./12., yMin=-d/2+tf*5./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_c2 = instance1.nodes.getByBoundingBox(xMax=tw/12, xMin=-tw/12, yMax=d/2-tf*5./12., yMin=-d/2+tf*7./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    nodes1_c3 = instance1.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d/2-tf*5./12., yMin=d/2-tf*7./12., zMax=len_solid+mesh_size_solid/4, zMin=-mesh_size_solid/4)
    # Get final matrices
    final_pmat1, final_cmat1 = mat_assemble(nodes1_p1, nodes1_p2, nodes1_p3, nodes1_p4, nodes1_p5, nodes1_p6, nodes1_p7, nodes1_p8, nodes1_p9, nodes1_p10,
                                            nodes1_p11, nodes1_p12, nodes1_p13, nodes1_p14, nodes1_p15, nodes1_p16, nodes1_c1, nodes1_c2, nodes1_c3, scale)
    # Establish sets
    set_define(final_pmat1, nodes1, Assembly, 'S1P%d')
    set_define(final_cmat1, nodes1, Assembly, 'S1C%d')
    return s3_xy_coord_list


def set_selection_top(Assembly, nodes2, instance2, w, d, tf, tw, r, len_solid, mesh_size_solid, mesh_size_flange, mem_len, ef, ew, scale):
    """
    Function for establishing profile and centerline node sets and element sets in the bottom solid segment.
    :param Assembly: model assembly of instances.
    :param nodes2: nodes in the top solid segment.
    :param instance2: top solid segment instance.
    :param w: section width.
    :param d: section depth.
    :param tf: flange thickness.
    :param tw: web thickness.
    :param r: fillet radius.
    :param len_solid: length of solid element segment.
    :param mesh_size_solid: general mesh size of solid element segment.
    :param mesh_size_flange: mesh size of flange.
    :param mem_len: member length.
    :param ef: number of elements per thickness at flange, 3.
    :param ew: number of elements per thickness at web, 3.
    :param scale: unit scale.
    :return: organized element matrix.
    """
    tol = 1 / scale
    # create element sets
    s3_xy_coord_list = ele_set_define(instance2, Assembly, d, tf, r, ef, ew, 'S2E')
    # top segments
    nodes2_p9 = instance2.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d/2+tf/12, yMin=d/2-tf/12, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p8 = instance2.nodes.getByBoundingBox(xMax=w/2+mesh_size_flange/4, xMin=w/2-mesh_size_flange/4, yMax=d/2+tf/12, yMin=d/2-r-tf*13./12, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p7 = instance2.nodes.getByBoundingBox(xMax=w, xMin=tw/2+r-tol, yMax=d/2-tf*11./12., yMin=d/2-tf*13./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p6 = instance2.nodes.getByBoundingCylinder(center1=(tw/2+r, d/2-tf-r, mem_len+mesh_size_solid/4), center2=(tw/2+r, d/2-tf-r, mem_len-len_solid-mesh_size_solid/4), radius=r+tol)
    nodes2_p5 = instance2.nodes.getByBoundingBox(xMax=tw*7./12., xMin=tw*5./12., yMax=d/2-r-tf+tol, yMin=-d/2+r+tf-tol, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p4 = instance2.nodes.getByBoundingCylinder(center1=(tw/2+r, -d/2+tf+r, mem_len+mesh_size_solid/4), center2=(tw/2+r, -d/2+tf+r, mem_len-len_solid-mesh_size_solid/4), radius=r+tol)
    nodes2_p3 = instance2.nodes.getByBoundingBox(xMax=w, xMin=tw/2+r-tol, yMax=-d/2+tf*13./12., yMin=-d/2+tf*11./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p2 = instance2.nodes.getByBoundingBox(xMax=w/2+mesh_size_flange/4, xMin=w/2-mesh_size_flange/4, yMax=-d/2+r+tf*13./12, yMin=-d/2-tf/12, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p1 = instance2.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=-d/2+tf/12, yMin=-d/2-tf/12, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p16 = instance2.nodes.getByBoundingBox(xMax=-w/2+mesh_size_flange/4, xMin=-w/2-mesh_size_flange/4, yMax=-d/2+r+tf*13./12, yMin=-d/2-tf/12, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p15 = instance2.nodes.getByBoundingBox(xMax=-tw/2-r+tol, xMin=-w, yMax=-d/2+tf*13./12., yMin=-d/2+tf*11./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p14 = instance2.nodes.getByBoundingCylinder(center1=(-tw/2-r, -d/2+tf+r, mem_len+mesh_size_solid/4), center2=(-tw/2-r, -d/2+tf+r, mem_len-len_solid-mesh_size_solid/4), radius=r+tol)
    nodes2_p13 = instance2.nodes.getByBoundingBox(xMax=-tw*5./12., xMin=-tw*7./12., yMax=d/2-r-tf+tol, yMin=-d/2+r+tf-tol, zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p12 = instance2.nodes.getByBoundingCylinder(center1=(-tw/2-r, d/2-tf-r, mem_len+mesh_size_solid/4), center2=(-tw/2-r, d/2-tf-r, mem_len-len_solid-mesh_size_solid/4), radius=r+tol)
    nodes2_p11 = instance2.nodes.getByBoundingBox(xMax=-tw/2-r+tol, xMin=-w, yMax=d/2-tf*11./12., yMin=d/2-tf*13./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_p10 = instance2.nodes.getByBoundingBox(xMax=-w/2+mesh_size_flange/4, xMin=-w/2-mesh_size_flange/4, yMax=d/2+tf/12., yMin=d/2-r-tf*13./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_c1 = instance2.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=-d/2+tf*7./12., yMin=-d/2+tf*5./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_c2 = instance2.nodes.getByBoundingBox(xMax=tw/12, xMin=-tw/12, yMax=d/2-tf*5./12., yMin=-d/2+tf*7./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    nodes2_c3 = instance2.nodes.getByBoundingBox(xMax=w, xMin=-w, yMax=d/2-tf*5./12., yMin=d/2-tf*7./12., zMax=mem_len+mesh_size_solid/4, zMin=mem_len-len_solid-mesh_size_solid/4)
    # Get final matrices
    final_pmat2, final_cmat2 = mat_assemble(nodes2_p1, nodes2_p2, nodes2_p3, nodes2_p4, nodes2_p5, nodes2_p6, nodes2_p7, nodes2_p8, nodes2_p9, nodes2_p10,
                                            nodes2_p11, nodes2_p12, nodes2_p13, nodes2_p14, nodes2_p15, nodes2_p16, nodes2_c1, nodes2_c2, nodes2_c3, scale)
    # Establish sets
    set_define(final_pmat2, nodes2, Assembly, 'S2P%d')
    set_define(final_cmat2, nodes2, Assembly, 'S2C%d')
    return s3_xy_coord_list


def mat_assemble(node_p1, node_p2, node_p3, node_p4, node_p5, node_p6, node_p7, node_p8, node_p9, node_p10,
                 node_p11, node_p12, node_p13, node_p14, node_p15, node_p16, node_c1, node_c2, node_c3, scale):
    """
    Function for organizing profile and centerline nodes and indexing them.
    """
    mat_p1 = split_node_array(node_p1, 0, False, scale)
    mat_p2 = split_node_array(node_p2, 1, False, scale)
    mat_p3 = split_node_array(node_p3, 0, True, scale)
    mat_p4 = split_node_array(node_p4, 0, True, scale)
    mat_p5 = split_node_array(node_p5, 1, False, scale)
    mat_p6 = split_node_array(node_p6, 0, False, scale)
    mat_p7 = split_node_array(node_p7, 0, False, scale)
    mat_p8 = split_node_array(node_p8, 1, False, scale)
    mat_p9 = split_node_array(node_p9, 0, True, scale)
    mat_p10 = split_node_array(node_p10, 1, True, scale)
    mat_p11 = split_node_array(node_p11, 0, False, scale)
    mat_p12 = split_node_array(node_p12, 0, False, scale)
    mat_p13 = split_node_array(node_p13, 1, True, scale)
    mat_p14 = split_node_array(node_p14, 0, True, scale)
    mat_p15 = split_node_array(node_p15, 0, True, scale)
    mat_p16 = split_node_array(node_p16, 1, True, scale)
    mat_c1 = split_node_array_simple(node_c1, 0, False, scale)
    mat_c2 = split_node_array_simple(node_c2, 1, False, scale)
    mat_c3 = split_node_array_simple(node_c3, 0, False, scale)
    # concatenate the matrices
    final_pmat = np.concatenate((mat_p1[:-1], mat_p2[:-1], mat_p3[:-1], mat_p4[:-1], mat_p5[:-1], mat_p6[:-1], mat_p7[:-1], mat_p8[:-1], mat_p9[:-1],
                                 mat_p10[:-1], mat_p11[:-1], mat_p12[:-1], mat_p13[:-1], mat_p14[:-1], mat_p15[:-1], mat_p16[:-1]), axis=0)
    final_cmat = np.concatenate((mat_c1, mat_c2, mat_c3), axis=0)
    return final_pmat, final_cmat


def set_define(final_mat, nodes, Assembly, set_name):
    """
    Function for defining sets in ABAQUS according to node indices.
    :param final_mat: finalized node matrix.
    :param nodes: total node set.
    :param Assembly: instance assembly.
    :param set_name: set name format.
    """
    for i in range(final_mat.shape[0]):
        total_nodes = nodes[int(final_mat[i, 0])-1:int(final_mat[i, 0])]
        for j in range(1, final_mat.shape[1]):
            current_nodes = nodes[int(final_mat[i, j])-1:int(final_mat[i, j])]
            total_nodes = total_nodes + current_nodes
        Assembly.Set(nodes=total_nodes, name=set_name % i)


def ele_set_define(instance1, Assembly, d, tf, r, ef, ew, type_):
    """
    Function for defining element sets.
    :param instance1: solid segment instance.
    :param Assembly: model assembly of instances.
    :param d: section depth.
    :param tf: flange thickness.
    :param r: fillet radius.
    :param ef: number of elements per thickness at flange, 3.
    :param ew: number of elements per thickness at web, 3.
    :param type_: element set name format.
    :return: a list of list of node 1-D coordinates.
    """
    # define a center coordinate and index list for all the elements in the instance
    coord_list = []
    for i in range(len(instance1.elements)):
        con_list = instance1.elements[i].connectivity
        ele_coord_list = []
        for j in con_list:
            ele_coord_list.append(list(instance1.nodes[j].coordinates))
        ele_coord_list = np.mean(np.array(ele_coord_list), axis=0)
        coord_list.append(list(ele_coord_list))
    coord_list = np.array(coord_list)
    ind_list = np.array(range(len(instance1.elements)))
    # get the information of bottom surface elements
    bot_indices = np.where(coord_list[:, 2] == np.min(coord_list[:, 2]))[0]
    len_z = int(len(coord_list) / len(bot_indices))
    xy_coord = coord_list[bot_indices][:, :2]
    h_2 = d / 2 - tf - (1 - 2 ** 0.5 / 2) * r
    # split all elements into upper flange, lower flange and web sets
    fu_coord_list = coord_list[np.where(coord_list[:, 1] > h_2)]
    fu_ind_list = ind_list[np.where(coord_list[:, 1] > h_2)]
    fb_coord_list = coord_list[np.where(coord_list[:, 1] < -h_2)]
    fb_ind_list = ind_list[np.where(coord_list[:, 1] < -h_2)]
    wm_coord_list = coord_list[np.where(np.logical_and(-h_2 <= coord_list[:, 1], coord_list[:, 1] <= h_2))]
    wm_ind_list = ind_list[np.where(np.logical_and(-h_2 <= coord_list[:, 1], coord_list[:, 1] <= h_2))]
    # split bottom surface elements into upper flange, lower flange and web sets
    fu_x = int(len(xy_coord[np.where(xy_coord[:, 1] > h_2)]) / ef)
    fb_x = int(len(xy_coord[np.where(xy_coord[:, 1] < -h_2)]) / ef)
    wm_y = int(len(xy_coord[np.where(np.logical_and(-h_2 <= xy_coord[:, 1], xy_coord[:, 1] <= h_2))]) / ew)
    # Define sets in ABAQUS and get organized element set
    def get_lists(c_list, i_list, ele_num, ind1, ind2, n_):
        s1_args = np.argsort(c_list[:, 2])
        s1_coord_list = c_list[s1_args]
        s1_ind_list = i_list[s1_args]
        s1_coord_list = s1_coord_list.reshape((len_z, -1, 3))
        s1_ind_list = s1_ind_list.reshape((len_z, -1))
        s2_args = np.argsort(s1_coord_list[:, :, ind1], axis=1)
        s2_coord_list = np.zeros_like(s1_coord_list)
        s2_ind_list = np.zeros_like(s1_ind_list)
        for i in range(s1_coord_list.shape[0]):
            s2_coord_list[i] = s1_coord_list[i, s2_args[i]]
            s2_ind_list[i] = s1_ind_list[i, s2_args[i]]
        s2_coord_list = s2_coord_list.reshape((len_z, ele_num, -1, 3))
        s2_ind_list = s2_ind_list.reshape((len_z, ele_num, -1))
        s3_args = np.argsort(s2_coord_list[:, :, :, ind2], axis=2)
        s3_coord_list = np.zeros_like(s2_coord_list)
        s3_ind_list = np.zeros_like(s2_ind_list)
        for i in range(s2_coord_list.shape[0]):
            for j in range(s2_coord_list.shape[1]):
                s3_coord_list[i, j] = s2_coord_list[i, j, s3_args[i, j]]
                s3_ind_list[i, j] = s2_ind_list[i, j, s3_args[i, j]]
        s3_ind_list = np.transpose(s3_ind_list, (1, 2, 0))
        s3_coord_list = np.transpose(s3_coord_list, (1, 2, 0, 3))
        s3_xy_coord_list = s3_coord_list[:, :, 0, :2]
        # Define sets in ABAQUS
        for i in range(s3_ind_list.shape[0]):
            for j in range(s3_ind_list.shape[1]):
                set_list = s3_ind_list[i, j]
                total_elements = instance1.elements[set_list[0]: set_list[0] + 1]
                for k in range(1, len(set_list)):
                    current_elements = instance1.elements[set_list[k]: set_list[k] + 1]
                    total_elements = total_elements + current_elements
                Assembly.Set(nodes=total_elements, name=type_+'%d' % n_)
                n_ += 1
        return s3_xy_coord_list, n_
    # establish sets and return the list of list of node 1-D coordinates
    xy1, n = get_lists(fb_coord_list, fb_ind_list, fb_x, 0, 1, 0)
    xy2, n = get_lists(wm_coord_list, wm_ind_list, wm_y, 1, 0, n)
    xy3, _ = get_lists(fu_coord_list, fu_ind_list, fu_x, 0, 1, n)
    return [xy1[:, :, 0], xy2[:, :, 1], xy3[:, :, 0]]
