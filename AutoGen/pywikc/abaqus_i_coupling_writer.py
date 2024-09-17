import os
from .abaqus_writer import AbaqusWriter
from .abaqus_writer import AbaqusNonLinearCouplingWriter


class AbaqusICouplingWriter(AbaqusNonLinearCouplingWriter):
    """ Writes the keywords for insertion from a set of ICoupling's. """

    def __init__(self, output_dir):
        AbaqusWriter.__init__(self, output_dir)
        self.KEYWFILE_BASE = 'MPC_Keywords.txt'
        self.JTYPE_DEFAULT = 0
        self._clear_output()
        return

    def write(self, couplings):
        """ Writes the coupling to file for insertion to the input file. 
        :param list couplings: [ICoupling] Couplings to write to file.
        """
        field_strings = []
        mpc_strings = []
        field_dir_strings = []
        all_normals = dict()
        for couple in couplings:
            jtype = self._gen_jtype(couple)
            beam_node = list(couple.beam_node.keys())[0]
            shell_node_ids = list(couple.continuum_nodes.keys())
            warping_values = couple.warping_fun
            field_strings += self._gen_field_strings(warping_values)
            mpc_strings += self._gen_mpc_strings(beam_node, shell_node_ids, jtype)
            all_normals[beam_node] = couple.normal_direction
        # Write the strings
        amp_strings = self._gen_amp_strings()
        field_dir_strings = self._gen_dir_field_strings(all_normals)
        self._write_keyw_file(field_strings, mpc_strings, amp_strings, field_dir_strings)
        return

    def _write_keyw_file(self, field_strings, mpc_strings, amp_strings, normal_strings):
        """ Writes the file containing the keyword additions. """
        keyw_file = os.path.join(self.output_dir, self.KEYWFILE_BASE)
        with open(keyw_file, 'w') as file:
            file.write('** MPC Keywords\n** Copy these in the model definition\n')
            file.writelines(mpc_strings)
            file.write('\n\n** Amplitude Keyword\n** Copy these in the model definition\n')
            file.writelines(amp_strings)
            file.write('\n\n** Field Keywords\n** Copy these in the first step\n')
            file.writelines(field_strings)
            file.write('\n\n** Normal Direction Field Keywords\n** Copy these in the first step\n')
            file.writelines(normal_strings)
        return

    def _gen_dir_field_strings(self, normal_directions):
        """ Returns the strings that define the initial cross-section normal directions at each beam node. """
        direc_1_strings = []
        direc_2_strings = []
        direc_3_strings = []
        for node, d in normal_directions.items():
            direc_1_strings.append(', '.join([str(node), str(d[0]) + '\n']))
            direc_2_strings.append(', '.join([str(node), str(d[1]) + '\n']))
            direc_3_strings.append(', '.join([str(node), str(d[2]) + '\n']))
        strings = []
        strings.append(', '.join(['*Field', 'variable=2', 'amplitude=warp_fun_amp\n']))
        strings += direc_1_strings
        strings.append(', '.join(['*Field', 'variable=3', 'amplitude=warp_fun_amp\n']))
        strings += direc_2_strings
        strings.append(', '.join(['*Field', 'variable=4', 'amplitude=warp_fun_amp\n']))
        strings += direc_3_strings
        return strings
