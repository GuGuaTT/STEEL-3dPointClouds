
"""
Functions to generate the imperfection geometries.
"""
import numpy as np
from .i_sec_imperfections import flange_imperfection, web_imperfection, straightness_imperfection, \
    twisting_imperfection, plumbness_imperfection

# -------------------------------------------------------------------------------------------------------------------- #


def set_imperfection_properties(component):
    """
    Sets all the imperfection properties for the component.
    Properties dictionary contains:
    n1
    n2
    length
    num_waves
    total_wave_length
    flange_width
    web_depth
    delta_flange
    delta_web
    epsilon
    delta_global
    delta_plumbness
    theta_twist
    oos_axis
    oop_axis
    """

    # Constants
    WEB_FACTOR = 300.
    FLANGE_FACTOR = 250.
    STRAIGHT_FACTOR = 1500.
    PLUMB_FACTOR = 1000.
    TWIST_FACTOR = 0.6 / 100.

    def local_amplitudes(section, dw_max, df_max):
        """ Calculate the flange and web imperfection amplitudes.
        :param section: section object.
        :param float dw_max: max allowable web amplitude.
        :param float df_max: max allowable flange amplitude.
        :return list: [float] web and flange imperfection amplitudes.
        """
        # First try dw = dw_max, geometrical compatibility
        dw = dw_max
        df = dw * section.bf / 2. / ((section.d - section.tf) / 2.)

        # Then try df = df_max if not satisfied
        if abs(df) > abs(df_max):
            df = df_max
            dw = df * ((section.d - section.tf) / 2.) / (section.bf / 2.)
            print('Flange controlled imperfection amplitude. Web amp = {0:0.10f}, flange amp = {1:0.10f}'.format(dw, df))
        else:
            print('Web controlled imperfection amplitude. Web amp = {0:0.10f}, flange amp = {1:0.10f}'.format(dw, df))

        return [dw_max, df_max]

    # Start setting properties
    props = dict()
    props['n1'] = component.coord_sys.basis[:, 0]
    props['n2'] = component.coord_sys.basis[:, 1]
    props['num_waves'] = component.imperfection_props['num_of_waves']
    props['total_wave_length'] = component.imperfection_props['wave_length_factor'] * component.section.d
    props['flange_width'] = component.section.bf
    props['web_depth'] = component.section.d - 2. * component.section.tf
    # todo: other than constant, also for flange and web?
    props['epsilon'] = 0.2
    # RBS connection properties
    if 'is_RBS' in component.imperfection_props:
        props['is_RBS'] = component.imperfection_props['is_RBS']
        props['RBS_offset'] = component.imperfection_props['RBS_offset']
    else:
        props['is_RBS'] = False
    if "is_half_model" in component.imperfection_props and component.imperfection_props['is_half_model'] is True:
        component.length = component.length * 2
    props['length'] = component.length

    # Compute maximum amplitudes
    max_flange_amp = component.section.bf / FLANGE_FACTOR * component.imperfection_props['local_scale_flange']
    max_web_amp = component.section.d / WEB_FACTOR * component.imperfection_props['local_scale_web']
    local_amps = local_amplitudes(component.section, max_web_amp, max_flange_amp)
    props['delta_web'] = local_amps[0]
    props['delta_flange'] = local_amps[1]
    oos_ip_amp = props['length'] / STRAIGHT_FACTOR * component.imperfection_props['straight_scale_ip']
    oos_op_amp = props['length'] / STRAIGHT_FACTOR * component.imperfection_props['straight_scale_op']
    props['delta_global_ip'] = oos_ip_amp
    props['delta_global_op'] = oos_op_amp
    oop_ip_amp = props['length'] / PLUMB_FACTOR * component.imperfection_props['plumb_scale_ip']
    oop_op_amp = props['length'] / PLUMB_FACTOR * component.imperfection_props['plumb_scale_op']
    props['delta_plumbness_ip'] = oop_ip_amp
    props['delta_plumbness_op'] = oop_op_amp
    props['theta_twist'] = TWIST_FACTOR * component.imperfection_props['twist_scale']
    props['oos_ip_axis'] = component.coord_sys.basis[:, 1]
    props['oos_op_axis'] = component.coord_sys.basis[:, 0]
    props['oop_ip_axis'] = component.coord_sys.basis[:, 1]
    props['oop_op_axis'] = component.coord_sys.basis[:, 0]

    component.imperfection_props = props
    pass


def generate_component_imp(component):
    """ Generates the imperfections for a component. """
    props = component.imperfection_props
    mid_height = component.length / 2.
    # Process the beam nodes
    for node_id, coords in component.beam_nodes.items():
        imp = np.array([0., 0., 0.])
        imp += i_straight_imp(coords, props)
        imp += i_plumb_imp(coords, props)
        component.node_imperfections[node_id] = imp
    # Process the continuum nodes
    for node_id, coords in component.continuum_nodes.items():
        if coords[2] > mid_height and props['is_RBS'] is False:
            is_top = True
        else:
            is_top = False
        imp = np.array([0., 0., 0.])
        imp += i_straight_imp(coords, props)
        imp += i_plumb_imp(coords, props)
        imp += i_local_imp(coords, component.section.tf, component.section.tw, component.section.d, props, is_top)
        imp += i_twist_imp(coords, props)
        component.node_imperfections[node_id] = imp
    pass


def i_local_imp(coord, tf, tw, d, props, is_top_seg):
    """ Returns the local flange/web imperfection.
    :param float tf: flange thickness
    :param float tw: web thickness
    :param float d: section depth
    :param list coord: Node position in component coordinate system.
    :param dict props: Defines the properties of the imperfection.
    :param bool is_top_seg: If yes, the treat as top segment.
    """
    x = coord[0]
    y = coord[1]
    z = coord[2]
    # Check if top segment or bottom segment
    if is_top_seg:
        # Top segment z_mod starts at the top of the column and is positive in -z direction
        z_mod = props['length'] - z
        should_reverse = True
    else:
        z_mod = z
        should_reverse = False
    # Account for the RBS offset, if RBS, will always be "bottom segment"
    if props['is_RBS']:
        # todo: this is a hack, and should be made into an option in web/flange imperfections
        if props['RBS_offset'] <= z <= props['RBS_offset'] + props['total_wave_length']:
            z_mod -= props['RBS_offset']
        else:
            # Large value so that will not be considered for local imperfections
            z_mod = 1.0e8
    # Separate flange and web
    # todo: is seperating on x==0 the best condition? - is it necessary?
    x_tol = 1.e-8
    is_web_node = (abs(x) <= x_tol + tw / 2) and (abs(y) <= x_tol + d / 2 - tf)
    # is_web_node = (abs(x) <= x_tol)
    if is_web_node:
        # Web node
        local_imp = web_imperfection(y, z_mod, should_reverse, tf, **props)
    else:  # Flange node
        local_imp = flange_imperfection(x, y, z_mod, should_reverse, **props)

    return np.array(local_imp)


def i_straight_imp(coord, props):
    """ Returns the out-of-straightness imperfection.
    :param list coord: Node position in component coordinate system.
    :param dict props: Defines the properties of the imperfection.
    """
    z = coord[2]

    return np.array(straightness_imperfection(z, **props))


def i_twist_imp(coord, props):
    """ Returns the out-of-straightness imperfection.
    :param list coord: Node position in component coordinate system.
    :param dict props: Defines the properties of the imperfection.
    """
    x = coord[0]
    y = coord[1]
    z = coord[2]

    return np.array(twisting_imperfection(x, y, z, **props))


def i_plumb_imp(coord, props):
    """ Returns the out-of-straightness imperfection.
    :param list coord: Node position in component coordinate system.
    :param dict props: Defines the properties of the imperfection.
    """
    z = coord[2]

    return np.array(plumbness_imperfection(z, **props))

# -------------------------------------------------------------------------------------------------------------------- #
