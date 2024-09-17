from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import *
from visualization import *
from odbAccess import *
import os
import time


def result_reader(main_f, odb_result_f, pkl_f, name):
    """
    Function for extracting results from the odb file.
    :param main_f: main folder's absolute path.
    :param odb_result_f: folder name for saving odb results.
    :param pkl_f: folder name for saving pickle files.
    :param name: model job name (.inp stripped).
    """
    # Open odb file, read section type and section name
    odb = openOdb(os.path.join(odb_result_f, name) + ".odb")
    section_name = name.split('-')[1]
    section_type = name.split('-')[0]
    if section_type == "hb":
        po.hb_model_result_reader(odb, main_f, pkl_f, name, section_name)
    elif section_type == "fb":
        po.fb_model_result_reader(odb, main_f, pkl_f, name, section_name)
    elif section_type == "cl":
        po.cl_model_result_reader(odb, main_f, pkl_f, name, section_name)
    else:
        raise ValueError


# Main folder's absolute path
main_folder = 'D:\\AutoGen'
os.chdir(main_folder)
import functions.postprocessing as po

# Sub-folders
finp_save_folder = os.path.join(main_folder, 'final_inp')
odb_result_folder = os.path.join(main_folder, 'odb_file')
pkl_folder = os.path.join(main_folder, 'pkl_file')
os.chdir(odb_result_folder)

for job_name in os.listdir(finp_save_folder):
    # Select subroutine according to member type (tolerance values in subroutines are different)
    if job_name[:2] == "cl":
        subroutine = os.path.join(main_folder, 'ALLcombinedSolid_CMN.for')
    elif job_name[:2] in ["hb", "fb"]:
        subroutine = os.path.join(main_folder, 'ALLcombinedSolid_DMN.for')
    else:
        raise ValueError
    # Create and submit job
    job = mdb.JobFromInputFile(name=job_name.strip(".inp"), inputFileName=os.path.join(finp_save_folder, job_name),
                               userSubroutine=subroutine, numCpus=16, numDomains=16)
    job.submit()
    job.waitForCompletion()
    # Extract results from odb file
    time.sleep(5.0)
    result_reader(main_folder, odb_result_folder, pkl_folder, job_name.strip(".inp"))
