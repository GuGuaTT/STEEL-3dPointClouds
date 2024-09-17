""" Functions to generate the nodal imperfections.

Imperfection shapes are computed based on the references:
@techreport{HillChartCriticalCompressive1940,
  title = {Chart for {{Critical Compressive Sress}} of {{Flat Rectangular Plates}}},
  author = {Hill, H. N.},
  year = {1940},
  address = {{Washington, D.C., United States}},
  institution = {{National Advisory Committee for Aeronautics. Langley Aeronautical Lab.}},
  number = {773},
  type = {Technical {{Note}}}
}
@phdthesis{HaaijerLocalbucklingwf1956,
  title = {Local Buckling of Wf Shapes in the Plastic Range},
  author = {Haaijer, Geerhard},
  year = {1956},
  address = {{Bethlehem, Pennsylvania, USA}},
  school = {Lehigh University},
  type = {{{PhD}}. {{Thesis}}}
}

"""

from __future__ import division, print_function
import math


def z_dir_imp_factor(z, num_waves, total_wave_length, **kwargs):
    """ Returns the imperfection factor based on the z-direction.
    :param float z: Coordinate in length direction.
    :param int num_waves: Number of half-wave lengths, either 1 or 2.
    :param float total_wave_length: Range of the local imperfection along the length.
    :return float: The factor in the length direction.
    """
    if num_waves == 2:
        z_norm_factor = 0.3849  # to normalize h to maximum of 1.0
        h = math.sin(math.pi * z / total_wave_length) ** 2 * \
            math.cos(math.pi * z / total_wave_length) / z_norm_factor
    elif num_waves == 1:
        h = math.sin(math.pi * z / total_wave_length) ** 2
    else:
        raise ValueError('num_waves should be either 1 or 2')
    return h


def flange_imperfection(x, y, z, reverse_wave, flange_width, total_wave_length, delta_flange, num_waves, epsilon, n2,
                        **kwargs):
    """ Returns the local imperfection specification for a node on the flanges.

    :param float x: x coordinate of node
    :param float y: y coordinate of node
    :param float z: z coordinate of node
    :param bool reverse_wave: reverse the direction of the local buckle wave
    :param float flange_width: flange width
    :param float total_wave_length: total length of imperfection in z-coordinate
    :param float delta_flange: maximum amplitude of the flange imperfection
    :param int num_waves: number of waves within the total wave length (1 or 2)
    :param float epsilon: web / flange restraint factor: 0 <= epsilon <= 1; 0 = SS, 1 = fixed
    :param np.ndarray n2: orientation of the n2 axis wrt the global coordinate system
    :return list: [float] nodal imperfection [x, y, z] in the global coordinates

    Notes:
        - Assumes that the z-direction is along the length of the member
        - Generates an in-plane local buckling mode
    """
    # if beyond the range, no local imperfections
    if abs(z) > total_wave_length:
        w = 0.
    else:
        # Factor in x-direction
        a1 = -4.963
        a2 = 9.852
        a3 = -9.778
        # x_b is normalized to between [-1, 1]
        x_b = x / (flange_width / 2.0)
        max_deflection = 1. + epsilon / (2. * a3) * (1. + a1 + a2 + a3)

        # take care to do the same thing on the positive and negative half-flanges
        x_b_abs = abs(x_b)
        if x_b < 0:
            x_sign = -1.0
        else:
            x_sign = 1.0
        f = x_b_abs + epsilon / (2 * a3) * (x_b_abs ** 3 + a1 * x_b_abs ** 4 + a2 * x_b_abs ** 3 + a3 * x_b_abs ** 2)
        f = f * x_sign / max_deflection
        if y < 0.:
            # Negative flange, modify to be opposite to that of the positive flange
            f *= -1.0

        # Factor in z-direction
        h = z_dir_imp_factor(z, num_waves, total_wave_length)
        # Total imperfection
        w = f * h * delta_flange
        if reverse_wave:
            w = -1.0 * w

    return list(w * n2)


def web_imperfection(y, z, reverse_wave, tf, web_depth, total_wave_length, delta_web, num_waves, epsilon, n1,
                     **kwargs):
    """ Returns the local imperfection specification for a node on the web.

    :param float y: y coordinate of node, -web_depth / 2 <= y <= web_depth / 2
    :param float z: z coordinate of node, 0 <= z <= total_wave_length
    :param bool reverse_wave: reverse the direction of the local buckle wave
    :param float web_depth: centerline depth of the web (h - tf)
    :param float tf: flange thickness
    :param float total_wave_length: total length of imperfection in z-coordinate
    :param float delta_web: maximum amplitude of the web imperfection
    :param int num_waves: number of waves within the total wave length (1 or 2)
    :param float epsilon: web / flange restraint factor: 0 <= epsilon <= 1; 0 = SS, 1 = fixed
    :param np.ndarray n1: orientation of the n1 axis wrt the global coordinate system
    :return list: [float] nodal imperfection [x, y, z] in the global coordinates

    Notes:
        - Assumes that the z-direction is along the length of the member
        - Maximum imperfection amplitude occurs at y = 0, and z = 0.304 * total_wave_length (2 waves).
        - Generates an in-plane local buckling mode
    """
    # if beyond the range, no local imperfections
    if abs(z) > total_wave_length:
        w = 0.
    else:
        # Factor in y-direction
        g_norm_factor = 1.02146
        y_b = y / web_depth
        g = math.pi * epsilon / 2. * (y_b ** 2 - 0.25) + (1. + epsilon / 2.) * math.cos(math.pi * y_b) / g_norm_factor

        # Factor in z-direction
        h = z_dir_imp_factor(z, num_waves, total_wave_length)

        w = g * h * delta_web
        if reverse_wave:
            w = -1.0 * w

    return list(w * n1)


def straightness_imperfection(z, length, delta_global_ip, delta_global_op, oos_ip_axis, oos_op_axis, **kwargs):
    """ Returns the out-of-straightness imperfection of the node.

    :param float z: z coordinate of node
    :param float length: total length of the component
    :param float delta_global_ip: maximum amplitude of the in-plane out-of-straightness global imperfection
    :param float delta_global_op: maximum amplitude of the out-of-plane out-of-straightness global imperfection
    :param np.ndarray oos_op_axis: direction of out-of-plane out-of-straightness wrt to the global coordinates
    :param np.ndarray oos_ip_axis: direction of in-plane out-of-straightness wrt to the global coordinates
    :return list: [float] nodal imperfection [x, y, z] in the global coordinates

    Notes:
        - Assumes that the z-direction is along the length of the member
    """
    w_op = -1.0 * (math.cos(2.0 * math.pi * z / length) - 1.0) * delta_global_op / 2.
    w_ip = -1.0 * (math.cos(2.0 * math.pi * z / length) - 1.0) * delta_global_ip / 2.
    return list(w_op * oos_op_axis + w_ip * oos_ip_axis)


def plumbness_imperfection(z, length, delta_plumbness_ip, delta_plumbness_op, oop_ip_axis, oop_op_axis, **kwargs):
    """  Returns the out-of-plumbness imperfection of the node.

    :param float z: z coordinate of node
    :param float length: total length of the component
    :param float delta_plumbness_ip: maximum amplitude of the in-plane out-of-plumbness global imperfection
    :param float delta_plumbness_op: maximum amplitude of the out-of-plane out-of-plumbness global imperfection
    :param np.ndarray oop_op_axis: direction of out-of-plane out-of-plumbness wrt to the global coordinates
    :param np.ndarray oop_ip_axis: direction of in-plane  out-of-plumbness wrt to the global coordinates
    :return list: [float] nodal imperfection [x, y, z] in the global coordinates
    """
    w_ip = delta_plumbness_ip * z / length
    w_op = delta_plumbness_op * z / length
    return list(w_op * oop_op_axis + w_ip * oop_ip_axis)


def twisting_imperfection(x, y, z, length, theta_twist, n1, n2, **kwargs):
    """ Returns the twisting imperfection of the node.

    :param float x: x coordinate of node
    :param float y: y coordinate of node
    :param float z: z coordinate of node
    :param float length: total length of the component
    :param float theta_twist: maximum angle of twist in the component (assumed at center point) in radians
    :param np.ndarray n1: orientation of the n1 axis wrt the global coordinate system
    :param np.ndarray n2: orientation of the n2 axis wrt the global coordinate system
    :return list: [float] nodal imperfection [x, y, z] in the global coordinates
    """
    z_total = z
    # theta = theta_twist * math.sin(2 * math.pi * z_total / length)
    theta = theta_twist * (math.sin(math.pi * z_total / length) ** 2)

    u_web = -1.0 * y * math.sin(theta)
    v_web = -1.0 * y * (1 - math.cos(theta))

    u_flange = x * (1 - math.cos(theta))
    v_flange = 1.0 * x * math.sin(theta)

    u = u_web + u_flange
    v = v_web + v_flange
    return list(u * n1 + v * n2)
