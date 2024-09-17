import os
from .imperfections.generate_imperfections import set_imperfection_properties, generate_component_imp
from .imperfections.abaqus_txt_writer import AbaqusTxtWriter
from .abaqus_i_coupling_writer import AbaqusICouplingWriter
from .component_reader import AbaqusInpToComponentReader


def gen_aba_couples(input_file, definition_file, output_dir):
    """ Generates the keywords for an Abaqus model.
    :param str input_file: Path to Abaqus input file that defines the model.
    :param str definition_file: Path to the definition file for components.
    :param str output_dir: Directory that exists to write the output files.
    """

    # Read the .inp file
    reader = AbaqusInpToComponentReader()
    reader.read(input_file, definition_file)
    imp_writer = AbaqusTxtWriter(reader.components)
    # Write the coupling defintions
    couplings = []
    for c in reader.components:
        couplings += c.couplings
    couple_writer = AbaqusICouplingWriter(output_dir)
    couple_writer.write(couplings)

    return


def gen_aba_imperfections(input_file, definition_file, output_dir):
    """ Generates the imperfections for an Abaqus model.
    :param str input_file: Path to Abaqus input file that defines the model.
    :param str definition_file: Path to the definition file for components.
    :param str output_dir: Directory that exists to write the output files.
    """

    # Read the .inp file
    reader = AbaqusInpToComponentReader()
    reader.read(input_file, definition_file)
    imp_writer = AbaqusTxtWriter(reader.components)
    # Generate the imperfections
    file_name = os.path.basename(os.path.normpath(input_file))
    imp_file = os.path.join(output_dir, file_name[:-4] + '-Imp.txt')
    for c in reader.components:
        set_imperfection_properties(c)
        generate_component_imp(c)
    imp_writer.write_imperfections(imp_file)
    return


def gen_aba_couples_imperfections(input_file, definition_file, output_dir):
    """ Generates the keywords and imperfections for an Abaqus model.
    :param str input_file: Path to Abaqus input file that defines the model.
    :param str definition_file: Path to the definition file for components.
    :param str output_dir: Directory that exists to write the output files.
    """

    # Read the .inp file
    reader = AbaqusInpToComponentReader()
    reader.read(input_file, definition_file)
    imp_writer = AbaqusTxtWriter(reader.components)
    # Generate the imperfections
    file_name = os.path.basename(os.path.normpath(input_file))
    imp_file = os.path.join(output_dir, file_name[:-4] + '-im.txt')
    for c in reader.components:
        set_imperfection_properties(c)
        generate_component_imp(c)
    imp_writer.write_imperfections(imp_file)
    # Write the coupling defintions
    couplings = []
    for c in reader.components:
        couplings += c.couplings
    couple_writer = AbaqusICouplingWriter(output_dir)
    couple_writer.write(couplings)
    return
