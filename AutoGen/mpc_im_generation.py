import os
from pywikc import processing

main_folder = 'D:\\AutoGen'

iinp_save_folder = os.path.join(main_folder, 'initial_inp')
finp_save_folder = os.path.join(main_folder, 'final_inp')
def_save_folder = os.path.join(main_folder, 'definition_file')
imp_save_folder = os.path.join(main_folder, 'imperfection_file')
linux_save_folder = "/home/gu/abaqus/imp_folder/"
mode = "Win"

for name in os.listdir(iinp_save_folder):
    input_file = os.path.join(iinp_save_folder, name)
    def_file = os.path.join(def_save_folder, name.strip(".inp")) + "-def.txt"
    processing.gen_aba_couples_imperfections(input_file, def_file, imp_save_folder)

    iinp_file = open(input_file, "r")
    ilines = iinp_file.readlines()
    iinp_file.close()

    mpc_file = open(os.path.join(imp_save_folder, "MPC_Keywords.txt"), "r")
    mlines = mpc_file.readlines()
    mpc_file.close()

    keyword_start = []
    keyword_end = []
    for mindex, mline in enumerate(mlines):
        if "** Copy these" in mline:
            keyword_start.append(mindex+1)
        if mline == "\n" and mlines[mindex+1] == "\n":
            keyword_end.append(mindex)

    code1 = mlines[keyword_start[0]:keyword_end[0]]
    code2 = mlines[keyword_start[1]:keyword_end[1]]
    code3 = mlines[keyword_start[2]:keyword_end[2]] + mlines[keyword_start[3]:]

    step_index = 0
    amp_index = 0
    im_index = 0
    for iindex, iline in enumerate(ilines):
        if (iline == "** STEP: Step-Axial\n" or iline == "** STEP: Step-Cyclic\n") and im_index == 0:
            im_index = iindex - 1
        if amp_index == 0 and "*Amplitude" in iline:
            amp_index = 1
            code1_index = iindex
        if "** MATERIALS" in iline:
            code2_index = iindex - 1
        if step_index == 0 and iline == "** OUTPUT REQUESTS\n":
            step_index = 1
            code3_index = iindex - 1

    imp_save_folder_m = ""
    for s in imp_save_folder.split("\\"):
        imp_save_folder_m = imp_save_folder_m + s + "\\\\"
    imp_save_folder_m.strip("\\\\")

    if mode == "Linux":
        imf = linux_save_folder
    else:
        imf = imp_save_folder_m

    final_iline = ilines[0:code1_index] + code1 + ilines[code1_index:code2_index] + code2 + ilines[code2_index:im_index] +\
                  ["*IMPERFECTION, INPUT=%s" % imf + name.strip(".inp") + "-im.txt\n"] + \
                  ilines[im_index:code3_index] + code3 + ilines[code3_index:]

    final_inp = open(os.path.join(finp_save_folder, name), "w")
    for fline in final_iline:
        final_inp.write(fline)
    final_inp.close()
