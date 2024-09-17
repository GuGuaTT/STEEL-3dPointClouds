
"""
** This script is for automatic generation of initial inp file and component definition file.
1) Initial inp file denotes the inp file is with transverse shear stiffness bug tackled (a problem for ABAQUS 2019)
but MPC keywords and imperfections are not added.
2) Component definition file is used for generation of MPC keywords as well as imperfection file.
3) The input unit system is Newton, mm, but calculation is done by Newton, cm/dm for column/beam in ABAQUS.
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import *
from visualization import *
from odbAccess import *
import itertools
import os


def inp_generation(component_type, section_name, mem_len, material_list, loading_protocol,
                   axial_load_ratio=0.0, support_ratio=None, boundary='Fixed',
                   main_folder='D:\\AutoGen', isf_name='initial_inp', dsf_name='definition_file'):
    """
    :param component_type: component type, choose between "Beam" and "Column".
    :param section_name: input section name.
    :param mem_len: member total length.
    :param material_list: material list containing E, s_{y,0}, Q_{\infty}, b, D_{\infty}, a, C_1, \gamma_1, C_2, \gamma_2.
    :param loading_protocol: loading protocol, choose among 'Symmetric', 'Collapse_consistent' and 'Monotonic'.
    :param axial_load_ratio: axial load to yield strength ratio (only for column).
    :param support_ratio: beam's maximum unbraced length to section weak-axis gyration radius ratio (only for beam).
    :param boundary: boundary condition, choose between 'Fixed' and 'Flexible' (only for column).
    :param main_folder: main folder's absolute path.
    :param isf_name: folder name for initial inp file storage.
    :param dsf_name: folder name for definition file storage.
    """
    # Create a new model
    Mdb()
    # Remove mask
    session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)
    # Set working directories
    inp_save_folder = os.path.join(main_folder, isf_name)
    def_save_folder = os.path.join(main_folder, dsf_name)
    os.chdir(main_folder)
    # Import functions
    import functions.preprocessing as pp
    import functions.writer_functions as wf
    # Change model name and attribute
    model_name = "Model-Macro"
    mdb.models.changeKey(fromName='Model-1', toName=model_name)
    mdb.models[model_name].setValues(noPartsInputFile=ON)
    # Create member object
    member = pp.member_creator(component_type, section_name, mem_len, material_list, loading_protocol,
                               main_folder, axial_load_ratio, support_ratio, boundary, support_ratio_limit=50.)
    # Establish loading amplitude
    pp.build_amplitude(member, loading_protocol)
    # Create parts
    part_solid, part_beam = pp.part_creator(member)
    # Create instances
    instances_surfs = pp.assembly_creator(member, part_solid, part_beam)
    # Create meshes
    pp.create_mesh(member, part_solid, part_beam)
    # Create sets and boundaries
    pp.set_boundary_creator(member, instances_surfs)
    # Job definition
    job = mdb.Job(name=member.job_name + '-0', model=model_name)
    job.writeInput()
    # Write definition file and modify initial inp file
    wf.create_definition(def_save_folder, member)
    wf.modify_inp(inp_save_folder, member)


material = [185000, 388, 133, 18, 130, 205, 22500, 215, 1300, 12]
axl_list = [0.1, 0.2, 0.3, 0.4, 0.5]
boundary_list = ['Flexible', 'Fixed']
lp_list = ['Symmetric', 'Collapse_consistent', 'Monotonic']
length_list = [4000, 5000, 6000]
section_list = ['W36X361', 'W36X330', 'W36X302', 'W36X282', 'W36X262', 'W36X256', 'W36X247', 'W36X232', 'W36X231',
                'W36X210', 'W36X194', 'W33X354', 'W33X318', 'W33X291', 'W33X263', 'W33X241', 'W33X221', 'W33X201',
                'W33X169', 'W30X261', 'W30X235', 'W30X211', 'W30X191', 'W30X173', 'W30X148', 'W30X132', 'W30X124',
                'W30X116', 'W30X108', 'W27X258', 'W27X235', 'W27X217', 'W27X194', 'W27X178', 'W27X161', 'W27X146',
                'W27X129', 'W24X94',  'W24X192', 'W24X176', 'W24X162', 'W24X146', 'W24X131', 'W24X117', 'W24X104',
                'W24X103', 'W21X93',  'W21X166', 'W21X147', 'W21X132', 'W21X122', 'W21X111', 'W21X101', 'W18X97',
                'W18X86',  'W18X76',  'W18X71',  'W18X65',  'W18X60',  'W18X55',  'W18X50',  'W18X130', 'W18X119',
                'W18X106', 'W16X89',  'W16X77',  'W16X67',  'W16X57',  'W16X50',  'W16X100', 'W14X99',  'W14X90',
                'W14X82',  'W14X74',  'W14X68',  'W14X61',  'W14X109']

