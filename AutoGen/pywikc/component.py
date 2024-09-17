import numpy as np


class ICoupling:
    """ Warping-Inclusive Kinematic Coupling for I-shaped cross-sections. """

    def __init__(self, beam_node, continuum_nodes, normal_direction, use_nonlinear, include_warping):
        """ Constructor.
        :param dict beam_node: {int: [float, float]} Beam node in the coupling in local coords.
        :param dict continuum_nodes: {int: [float, float]} Continuum nodes in the coupling in local coords.
        :param np.ndarray normal_direction: (3,) Orientation of cross-section normal vector.
        :param bool use_nonlinear: If True, then nonlinear version of coupling used.
        :param bool include_warping: If True, then use warping-inclusive coupling.
        """
        self.beam_node = beam_node
        self.continuum_nodes = continuum_nodes
        self.normal_direction = normal_direction
        self.use_nonlinear = use_nonlinear
        self.include_warping = include_warping

        self.warping_fun = self._compute_warping_fun()

    def _compute_warping_fun(self):
        """ Returns the warping function evaluated at each continuum node. """
        wfun = dict()
        for node_id, coords in self.continuum_nodes.items():
            wfun[node_id] = coords[0] * coords[1]
        return wfun


class CoordSys:
    """ Defines a right-handed coordinate system in R^3. """

    def __init__(self, point, basis):
        """ Constructor.
        :param np.ndarray point: (3, ) Origin of the coord sys wrt the global coord sys.
        :param np.ndarray basis: (3, 3) Matrix defining the local-to-global rotation, [n1, n2, n3].
        """
        self.pt = point
        self.basis = basis


class ISection:
    """ Defines an I-shaped cross-section. """

    def __init__(self, name, d, bf, tf, tw):
        """ Constructor.
        :param str name: Identifier for cross-section.
        :param float d: Total depth.
        :param float bf: Total flange width.
        :param float tf: Flange thickness.
        :param float tw: Web thickness.
        """
        self.name = name
        self.d = d
        self.bf = bf
        self.tf = tf
        self.tw = tw


class IComponent:
    """ Defines the origin, nodes, and couplings for a component with an I-shaped cross-section. """

    def __init__(self, component_id, section):
        """ Constructor.
        :param str component_id: Unique identifier for the component.
        :param ISection section: Defines the geometry of the cross-section.
        :param CoordSys coord_sys: Origin and orientation of the component.
        :param dict beam_nodes: {int: [float, float]} Defines all the beam nodes in local coords.
        :param dict continuum_nodes: {int: [float, float]} Defines all the continuum nodes in local coords.
        :param list couplings: [Coupling] Defines the couplings in the component.
        """
        self.id = component_id
        self.section = section

        self.beam_node_sets = list()
        self.continuum_node_sets = list()
        self.node_set_to_coordsys = dict()
        self.couplings_info = list()

        # Base coordinate system for the component
        self.coord_sys = None
        self.base_cys_id = None
        self.orientation_id = None
        # Offsets in component 3-axis for each coord system
        self.coord_sys_offsets = dict()
        # All beam and continuum nodes in the component
        self.beam_nodes = dict()
        self.continuum_nodes = dict()
        # Couplings in the component
        self.couplings = list()
        # Component length along n3-axis
        self.length = 0.

        # Container for imperfections
        self.node_imperfections = dict()
        # Defines the imperfection properties
        self.imperfection_props = dict()

    def _compute_length(self):
        """ Computes the length of the component from the beam and continuum node coordinates. """
        max_z = 0.
        for node_id, coords in self.beam_nodes.items():
            if coords[2] > max_z:
                max_z = coords[2]
        for node_id, coords in self.continuum_nodes.items():
            if coords[2] > max_z:
                max_z = coords[2]
        self.length = max_z

    def _check_imperfection_props(self):
        """ Checks if all the scales are defined in the imperfection props. """
        if 'local_scale_web' not in self.imperfection_props:
            self.imperfection_props['local_scale_web'] = 1.
        if 'local_scale_flange' not in self.imperfection_props:
            self.imperfection_props['local_scale_flange'] = 1.
        if 'straight_scale_op' not in self.imperfection_props:
            self.imperfection_props['straight_scale_op'] = 1.
        if 'twist_scale' not in self.imperfection_props:
            self.imperfection_props['twist_scale'] = 1.

    def get_imperfect_nodes(self):
        """ Returns the node coordinates with the imperfection. """
        self.imperfect_nodes = dict()
        for node_id, coord in self.continuum_nodes.items():
            self.imperfect_nodes[node_id] = np.array(coord) + np.array(self.node_imperfections[node_id])
        for node_id, coord in self.beam_nodes.items():
            self.imperfect_nodes[node_id] = np.array(coord) + np.array(self.node_imperfections[node_id])
        return self.imperfect_nodes
