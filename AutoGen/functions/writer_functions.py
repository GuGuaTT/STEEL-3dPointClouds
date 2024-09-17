
"""
** Some functions for writing files.
"""

import os


def create_definition(def_save_folder, member):
    """
    Function for creating definition file.
    :param def_save_folder: folder for definition file storage.
    :param member: beam or column object.
    """
    txt_file = open(def_save_folder + '\\' + member.job_name + '-def.txt', 'w')
    txt_file.write('*ISection, name=section\n')
    txt_file.write('{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}\n'.format(member.d, member.w, member.tf, member.tw))
    txt_file.write('\n')
    txt_file.write('*Component, name=member, section=section\n')
    txt_file.write('*BeamNodes\n')
    txt_file.write('BEAM-ALL\n')
    txt_file.write('*ContinuumNodes\n')
    if member.type in ["Column", "Full_Beam"]:
        txt_file.write('SOLID-BOT-ALL, SOLID-TOP-ALL\n')
        txt_file.write('*Coupling, jtype=27\n')
        txt_file.write('BEAM-BOT, SOLID-BOT-TOP\n')
        txt_file.write('*Coupling, jtype=27\n')
        txt_file.write('BEAM-TOP, SOLID-TOP-BOT\n')
        txt_file.write('*Imperfection, wave_length_factor={0:.1f}, num_of_waves=1, local_scale_web={1:.2f}, '
                       'local_scale_flange ={2:.2f}, straight_scale_op={3:.2f}, straight_scale_ip={4:.2f}, '
                       'plumb_scale_op={5:.2f}, plumb_scale_ip={6:.2f}, twist_scale={7:.2f}\n'.format(
            member.wave_len_factor, member.local_scale_web, member.local_scale_flange, member.straight_scale_op,
            member.straight_scale_ip, member.plumb_scale_op, member.plumb_scale_ip, member.twist_scale))
        txt_file.write('\n')
        txt_file.write('*EndDef\n')
        txt_file.close()
    elif member.type == "Half_Beam":
        txt_file.write('SOLID-BOT-ALL\n')
        txt_file.write('*Coupling, jtype=27\n')
        txt_file.write('BEAM-BOT, SOLID-BOT-TOP\n')
        txt_file.write('*Imperfection, wave_length_factor={0:.1f}, num_of_waves=1, local_scale_web={1:.2f}, '
                       'local_scale_flange ={2:.2f}, straight_scale_op={3:.2f}, straight_scale_ip={4:.2f}, '
                       'plumb_scale_op={5:.2f}, plumb_scale_ip={6:.2f}, twist_scale={7:.2f}, is_half_model=1\n'.format(
            member.wave_len_factor, member.local_scale_web, member.local_scale_flange, member.straight_scale_op,
            member.straight_scale_ip, member.plumb_scale_op, member.plumb_scale_ip, member.twist_scale))
        txt_file.write('\n')
        txt_file.write('*EndDef\n')
        txt_file.close()
    else:
        raise ValueError


# Modify inp file (bug in shear stiffness and add imperfection)
def modify_inp(inp_save_folder, member):
    """
    Function for modifying initial inp file.
    :param inp_save_folder: folder for initial inp file storage.
    :param member: beam or column object.
    """
    inp_file0 = open(member.main_folder + '\\' + member.job_name + '-0.inp', 'r')
    lines = inp_file0.readlines()
    inp_file0.close()
    # Fix the bug
    coord_list = []
    for index, line in enumerate(lines):
        if '*Transverse Shear' in line:
            index_ts = index
        if '*Element Output' in line:
            coord_list.append(index)
        if '*Beam Section' in line:
            index_bm = index
    # Add notations to the inp file
    lines[index_ts] = '*Transverse Shear Stiffness\n'
    for i_coord in range(len(coord_list)):
        lines[coord_list[i_coord] + 1] = lines[coord_list[i_coord] + 1].strip("\n") + ", COORD" + "\n"
        lines[coord_list[i_coord] - 1] = lines[coord_list[i_coord] - 1].strip("\n") + "COORD" + "\n"
    lines.insert(index_bm + 3, "%d, %d, %d\n" % (member.beam_flange_int_num, member.beam_web_int_num, member.beam_flange_int_num))
    lines.insert(2, '** section_name=%s, d=%.4f, w=%.4f, tf=%.4f, tw=%.4f, r=%.4f, mem_len=%.2f,\n'
                    '** E=%d, sy0=%d, QInf=%.1f, b=%.1f, DInf=%.1f, a=%.1f, C1=%.1f, gamma1=%.1f, C2=%.1f,  gamma2=%.1f,\n'
                    '** axial_load_ratio=%.1f, loading_protocol=%s, boundary=%s, support_ratio=%s, solid_len_factor=%.1f,\n'
                    '** wave_len_factor=%.1f, local_scale_web=%.2f, local_scale_flange =%.2f, straight_scale_op=%.2f, straight_scale_ip=%.2f,\n'
                    '** plumb_scale_op=%.2f, plumb_scale_ip=%.2f, twist_scale=%.2f, a_=%.2f, c_=%.2f\n'
                    '**\n' % (member.section_name, member.d, member.w, member.tf, member.tw, member.r, member.mem_len,
                              member.E, member.sy0, member.QInf, member.b, member.DInf, member.a, member.C1, member.gamma1, member.C2, member.gamma2,
                              member.axial_load_ratio, member.loading_protocol, member.boundary, member.support_ratio,
                              member.solid_len_factor, member.wave_len_factor, member.local_scale_web, member.local_scale_flange, member.straight_scale_op,
                              member.straight_scale_ip, member.plumb_scale_op, member.plumb_scale_ip, member.twist_scale, member.a_, member.c_))
    inp_file = open(inp_save_folder + '\\' + member.job_name + '.inp', 'w')
    for line in lines:
        inp_file.write(line)
    inp_file.close()
    os.remove(member.main_folder + '\\' + member.job_name + '-0.inp')
    print(member.job_name + "Created.\n")
