
"""
** Some helper functions for the main initial inp generation script.
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import *
from visualization import *
from odbAccess import *
import numpy as np
import math
import os


# Unit system scale factor
SYS_COLUMN_SCALE = 10.
SYS_BEAM_SCALE = 100.


# read section params in millimeter scale and save them in specified scale
def section_reader(main_folder, section_name, scale):
    """
    Function for reading section's geometrical parameters.
    :param main_folder: the main folder's absolute path.
    :param section_name: the name of section.
    :param scale: the scale factor used for import.
    :return: the section's depth d, width w, flange thickness t_f, web thickness t_w, fillet radius r and section strong-axis moment of inertia.
    """
    section_file = open(os.path.join(main_folder, "section_list.txt"))
    lines = section_file.readlines()
    section_file.close()
    for line in lines:
        line_info = line.strip().split(",")
        if line_info[0] == section_name:
            return float(line_info[2]) / scale, \
                   float(line_info[1]) / scale, \
                   float(line_info[4]) / scale, \
                   float(line_info[3]) / scale, \
                   float(line_info[5]) / scale, \
                   float(line_info[6]) * 1000000. / scale ** 4
    raise TypeError("No such section is found in the database.")


# output mesh size in different regions of member, and specify maximum mesh sizes
def mesh_size_output(main_folder, section_name, scale, min_web_number=5, min_flange_number=4):
    """
    Function for generating mesh size or element numbers in the numerical model.
    :param main_folder: the main folder's absolute path.
    :param section_name: the name of section.
    :param scale: the scale factor used for import.
    :param min_web_number: half of minimum element number along length direction of web.
    :param min_flange_number: half of minimum element number along length direction of flange.
    :return: mesh_size_solid: global mesh size for solid element segment.
             mesh_size_web: mesh size of web along length direction.
             mesh_size_flange: mesh size of flange along length direction.
             num_flange_up: number of elements at thickness of upper flange part.
             num_flange_low: number of elements at thickness of lower flange part.
             num_web: number of elements at thickness of web.
             num_mid: number of element at middle feature edge.
             num_side: number of elements at side feature edge.
             num_half_arc: half number of elements at fillet edge.
    """
    # import geometric values
    d, w, tf, tw, r, _ = section_reader(main_folder, section_name, scale)
    # default global mesh size for solid element segment.
    mesh_size_solid = 25. / scale
    # Make sure small sections have enough number of elements
    if math.floor((d - 2. * r - 2. * tf) / mesh_size_solid) < min_web_number * 2:
        mesh_size_web = (d - 2. * r - 2. * tf) / min_web_number / 2.
    else:
        mesh_size_web = mesh_size_solid
    if math.floor((w - 2. * r - tw) / mesh_size_solid) < min_flange_number * 2:
        mesh_size_flange = (w - 2. * r - tw) / min_flange_number / 2.
    else:
        mesh_size_flange = mesh_size_solid
    # number of elements at thickness direction
    num_flange_up = 2
    num_flange_low = 1
    num_web = 3
    # number of elements at arc and feature edges
    num_half_arc = int(round(1. / 2. * 3.14 * r / tw))
    if num_half_arc < 1.:
        num_half_arc = 1
    num_mid = num_web
    num_side = num_half_arc
    return mesh_size_solid, mesh_size_web, mesh_size_flange, num_flange_up, \
           num_flange_low, num_web, num_mid, num_side, num_half_arc


def split_node_array(node_array, xy_index, reverse, scale):
    """
    Function for splitting the selected nodes into several individual sets in parallel in the z-direction (profile nodes).
    :param node_array: the input node set.
    :param xy_index: split the input node set in x- or y- direction. 0 denotes x- and 1 denotes y-.
    :param reverse: in the selected direction, the node set order is increasing (False) or decreasing (True).
    :param scale: unit system scale factor used.
    :return: final_matrix of dimension (number of node sets, node number in each set), indicating all the node sets in a specific order.
    """
    coordinate_label_matrix = []
    for node in node_array:
        coordinate_label = [node.coordinates[0], node.coordinates[1], node.coordinates[2], int(node.label)]
        coordinate_label_matrix.append(coordinate_label)
    # (node_num, 4) matrix
    decimal = int(math.log10(scale) + 2)
    coordinate_label_matrix = np.array(coordinate_label_matrix)
    coordinate_label_matrix = np.round(coordinate_label_matrix, decimal)
    # (node_num, ) vector
    index_list = coordinate_label_matrix[:, xy_index]
    index_list = list(index_list)
    # yield count dictionary for index_list and then get the mean of counts
    dict1 = {}
    set_index_list = set(index_list)
    for key in set_index_list:
        dict1[str(key)] = index_list.count(key)
    mean_value = np.array(list(dict1.values())).mean()
    # get preserved keys and z direction length
    key_preserve_list = []
    z_length = 0
    for key, value in dict1.items():
        if value > mean_value:
            key_preserve_list.append(float(key))
            z_length = value
    # Save final matrix
    final_matrix = np.zeros((len(key_preserve_list), z_length))
    key_preserve_list.sort(reverse=reverse)
    for j in range(len(key_preserve_list)):
        k = 0
        for i in range(len(coordinate_label_matrix)):
            if coordinate_label_matrix[i, xy_index] == key_preserve_list[j]:
                final_matrix[j, k] = coordinate_label_matrix[i, 3]
                k += 1
        if k != z_length:
            raise ValueError('Problem with group!')
    return final_matrix


def split_node_array_simple(node_array, xy_index, reverse, scale):
    """
    Function for splitting the selected nodes into several individual sets in parallel in the z-direction (centerline nodes).
    :param node_array: the input node set.
    :param xy_index: split the input node set in x- or y- direction. 0 denotes x- and 1 denotes y-.
    :param reverse: in the selected direction, the node set order is increasing (False) or decreasing (True).
    :param scale: unit system scale factor used.
    :return: final_matrix of dimension (number of node sets, node number in each set), indicating all the node sets in a specific order.
    """
    coordinate_label_matrix = []
    for node in node_array:
        coordinate_label = [node.coordinates[0], node.coordinates[1], node.coordinates[2], int(node.label)]
        coordinate_label_matrix.append(coordinate_label)
    # (node_num, 4) matrix
    decimal = int(math.log10(scale) + 1)
    coordinate_label_matrix = np.array(coordinate_label_matrix)
    coordinate_label_matrix = np.round(coordinate_label_matrix, decimal)
    # (node_num, ) vector
    index_list = coordinate_label_matrix[:, xy_index]
    key_preserve_list = list(set(list(index_list)))
    z_length = int(len(coordinate_label_matrix) / len(key_preserve_list))
    # Save final matrix
    final_matrix = np.zeros((len(key_preserve_list), z_length))
    key_preserve_list.sort(reverse=reverse)
    for j in range(len(key_preserve_list)):
        k = 0
        for i in range(len(coordinate_label_matrix)):
            if coordinate_label_matrix[i, xy_index] == key_preserve_list[j]:
                final_matrix[j, k] = coordinate_label_matrix[i, 3]
                k += 1
        if k != z_length:
            raise ValueError('Problem with group!')
    return final_matrix
