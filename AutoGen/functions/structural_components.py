
"""
** Structural component classes.
"""

from functions.init_helper import *
import math


class StructuralComponent(object):
    def __init__(self, section_name, loading_protocol, main_folder,
                 solid_len_factor, wave_len_factor, local_scale_web, local_scale_flange,
                 straight_scale_op, straight_scale_ip, plumb_scale_op, plumb_scale_ip, twist_scale):
        """
        General structural component class.
        :param section_name: input section name.
        :param loading_protocol: loading protocol, choose among 'Symmetric', 'Collapse_consistent' and 'Monotonic'.
        :param main_folder: main folder's absolute path.
        :param solid_len_factor: the ratio of solid segment length to section depth.
        :param wave_len_factor: the ratio of local imperfection wave length to section depth.
        :param local_scale_web: web local imperfection scale factor.
        :param local_scale_flange: flange local imperfection scale factor.
        :param straight_scale_op: global out-of-plane straightness imperfection scale factor.
        :param straight_scale_ip: global in-plane straightness imperfection scale factor.
        :param plumb_scale_op: global out-of-plane plumb imperfection scale factor.
        :param plumb_scale_ip: global in-plane plumb imperfection scale factor.
        :param twist_scale: global twist imperfection scale factor.
        """
        self.section_name = section_name
        self.loading_protocol = loading_protocol
        self.main_folder = main_folder
        self.solid_len_factor = float(solid_len_factor)
        self.wave_len_factor = float(wave_len_factor)
        self.local_scale_web = float(local_scale_web)
        self.local_scale_flange = float(local_scale_flange)
        self.straight_scale_op = float(straight_scale_op)
        self.straight_scale_ip = float(straight_scale_ip)
        self.plumb_scale_op = float(plumb_scale_op)
        self.plumb_scale_ip = float(plumb_scale_ip)
        self.twist_scale = float(twist_scale)
        # Number of beam elements
        self.beam_ele_num = 30
        self.step_period = 0
        # Number of integration points on the beam flange and web
        self.beam_flange_int_num = 11
        self.beam_web_int_num = 13
        # Warning messages
        if solid_len_factor < wave_len_factor:
            raise ValueError("Solid length is too small.")
    # Define base function for making parameters
    def _make_parameters(self, mem_len, material_list, m_scale):
        # Beam column component common part
        self.mem_len = float(mem_len) / m_scale
        # Read geometrical and material properties
        self.d, self.w, self.tf, self.tw, self.r, self.ix = section_reader(self.main_folder, self.section_name, m_scale)
        self.E = float(material_list[0]) * m_scale ** 2
        self.sy0 = float(material_list[1]) * m_scale ** 2
        self.QInf = float(material_list[2]) * m_scale ** 2
        self.b = float(material_list[3])
        self.DInf = float(material_list[4]) * m_scale ** 2
        self.a = float(material_list[5])
        self.C1 = float(material_list[6]) * m_scale ** 2
        self.gamma1 = float(material_list[7])
        self.C2 = float(material_list[8]) * m_scale ** 2
        self.gamma2 = float(material_list[9])
        # Calculate spring stiffness
        self.spring = self.E * self.ix / self.mem_len
        # Calculate solid and beam lengths
        self.len_solid = round(self.solid_len_factor * self.d, int(math.log10(m_scale)))
        self.len_beam = self.mem_len - self.len_solid * 2
        # Calculate section area
        self.section_area = self.d * self.w - (self.d - 2. * self.tf) * (self.w - self.tw) + (4. - 3.14) * self.r ** 2.
        # Calculate residual stress parameters of parabolic model according to formula
        if self.d / self.w < 0.95:
            norm_db = -1
        elif self.d / self.w > 3.00:
            norm_db = 1
        else:
            norm_db = (self.d / self.w - 0.95) * 2 / (3.00 - 0.95) - 1
        if self.section_area * m_scale ** 2 < 1320:
            norm_A = -1
        elif self.section_area * m_scale ** 2 > 175000:
            norm_A = 1
        else:
            norm_A = (self.section_area * m_scale ** 2 - 1320) * 2 / (175000 - 1320) - 1
        self.a_ = (107 + 51 * norm_db + 20 * norm_A) * m_scale ** 2
        self.c_ = (-142 - 84 * norm_db) * m_scale ** 2
        print(self.a_)
        print(self.c_)
        # Obtain material matrices for beam and solid segments
        mat_prop = [self.E, self.sy0, self.QInf, self.b, self.DInf,
                    self.a, self.C1, self.gamma1, self.C2, self.gamma2]
        self.mat_prop_beam = tuple(mat_prop + [self.d, self.w, self.tf, self.tw, self.a_, self.c_,
                                               self.beam_flange_int_num, self.beam_web_int_num])
        mat_prop.insert(1, 0.3)
        self.mat_prop_solid = tuple(mat_prop)
        # Obtain transverse shear stiffness
        self.trans_shear_stiffness = self.section_area * 0.44 * self.mat_prop_beam[0] / 2. / (1. + 0.3)
        # Mesh control params
        self.mesh_size_solid, self.mesh_size_web, self.mesh_size_flange, self.num_flange_up, \
            self.num_flange_low, self.num_web, self.num_mid, self.num_side, self.num_half_arc \
            = mesh_size_output(self.main_folder, self.section_name, m_scale)
        # Warning messages
        if self.len_beam <= 0:
            raise ValueError("Length of solid segment is too large.")


class Column(StructuralComponent):
    def __init__(self, section_name, mem_len, material_list, loading_protocol, axial_load_ratio, boundary, main_folder,
                 solid_len_factor=1.5, wave_len_factor=1.5, local_scale_web=0.1, local_scale_flange=0.1,
                 straight_scale_op=1.0, straight_scale_ip=0.0, plumb_scale_op=0.0, plumb_scale_ip=0.0, twist_scale=0.0):
        """
        Column structural component class.
        :param mem_len: member length.
        :param material_list: material list containing E, s_{y,0}, Q_{\infty}, b, D_{\infty}, a, C_1, \gamma_1, C_2, \gamma_2.
        :param axial_load_ratio: axial load to yield strength ratio.
        :param boundary: boundary condition, choose between 'Fixed' and 'Flexible'.
        """
        # Python 2.X format
        super(Column, self).__init__(section_name, loading_protocol, main_folder, solid_len_factor,
                                     wave_len_factor, local_scale_web, local_scale_flange, straight_scale_op,
                                     straight_scale_ip, plumb_scale_op, plumb_scale_ip, twist_scale)
        # make parameters
        self.scale = SYS_COLUMN_SCALE
        self._make_parameters(mem_len, material_list, self.scale)
        # Define support ratio is None
        self.support_ratio = None
        # Define axial load ratio and boundary
        self.axial_load_ratio = axial_load_ratio
        if self.axial_load_ratio >= 1.:
            raise ValueError("Axial load ratio cannot go beyond 1.0!")
        self.boundary = boundary
        self.job_name = 'cl-%s-%d-%s-0%d-%s' % (section_name, int(self.mem_len * self.scale), loading_protocol[0],
                                                int(axial_load_ratio * 10), boundary[0:3:2])
        self.type = "Column"
        self.axial_load = self.axial_load_ratio * self.section_area * self.mat_prop_beam[1]
    # Inherit function for making parameters
    def _make_parameters(self, mem_len, material_list, m_scale):
        super(Column, self)._make_parameters(mem_len, material_list, m_scale)


class Beam(StructuralComponent):
    def __init__(self, section_name, mem_len, material_list, loading_protocol, support_ratio, spr_limit, main_folder,
                 solid_len_factor=2.0, wave_len_factor=1.5, local_scale_web=0.1, local_scale_flange=0.1,
                 straight_scale_op=1.0, straight_scale_ip=0.0, plumb_scale_op=0.0, plumb_scale_ip=0.0, twist_scale=0.5):
        """
        Beam structural component class.
        :param mem_len: member length.
        :param material_list: material list containing E, s_{y,0}, Q_{\infty}, b, D_{\infty}, a, C_1, \gamma_1, C_2, \gamma_2.
        :param support_ratio: beam's maximum unbraced length to section weak-axis gyration radius ratio (only for beam).
        :param spr_limit: beam's required support ratio limit.
        """
        # Python 2.X format
        super(Beam, self).__init__(section_name, loading_protocol, main_folder, solid_len_factor,
                                   wave_len_factor, local_scale_web, local_scale_flange, straight_scale_op,
                                   straight_scale_ip, plumb_scale_op, plumb_scale_ip, twist_scale)
        # make parameters
        self.scale = SYS_BEAM_SCALE
        self._make_parameters(mem_len, material_list, self.scale)
        # Define axial load and boundary
        self.axial_load_ratio = 0.
        self.boundary = "Fixed"
        # Define parameters related to supports
        self.support_ratio = support_ratio
        self.spr_limit = spr_limit
        # Calculate radius of inertia of weak-axis
        self.section_stiffness = self.tw ** 3. * (self.d - 2. * self.tf) / 12. + 2. * self.w ** 3. * self.tf / 12.
        self.radius_inertia = (self.section_stiffness / self.section_area) ** 0.5
        # Calculate the number of laterally supported segments under the given support ratio
        self.segments = math.ceil(self.mem_len / self.radius_inertia / self.support_ratio)
        # Calculate the length of laterally supported segment and warning message
        self.segment_length = self.mem_len / self.segments
        # Set axial load and default partition method
        self.axial_load = 0
        self.partition_index = 0
        if self.segment_length <= self.len_solid:
            raise ValueError("Solid length is too large for lateral support!")
        # Half beam status
        if self.support_ratio <= self.spr_limit:
            self.job_name = 'hb-%s-%d-%s-00-%s' % (section_name, int(self.mem_len * self.scale),
                                                   loading_protocol[0], int(support_ratio))
            self.type = "Half_Beam"
            # Half beam length
            self.len_beam = 0.5 * self.len_beam
            if self.segments % 2 == 0:
                self.partition_list = [i*self.segment_length/2.-self.len_beam/2. for i in range(2, int(self.segments), 2)]
                self.partition_index = 1
            else:
                self.partition_list = [i*self.segment_length/2.-self.len_beam/2. for i in range(1, int(self.segments), 2)]
        # Full beam status
        else:
            self.job_name = 'fb-%s-%d-%s-00-%s' % (section_name, int(self.mem_len * self.scale),
                                                   loading_protocol[0], int(support_ratio))
            self.type = "Full_Beam"
            self.partition_list = [i*self.segment_length-self.len_solid-self.len_beam/2. for i in range(1, int(self.segments))]
    # Inherit function for making parameters
    def _make_parameters(self, mem_len, material_list, m_scale):
        super(Beam, self)._make_parameters(mem_len, material_list, m_scale)
