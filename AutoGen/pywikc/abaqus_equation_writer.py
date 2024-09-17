import os
from .abaqus_writer import AbaqusWriter


class AbaqusLinearCouplingWriter(AbaqusWriter):
    """ Writes the constraints for using in an Abaqus input file. """

    def __init__(self, output_dir, input_path_prepend=''):
        """ Constructor.
        :param str output_dir: Directory where files will be saved.
        :param str input_path_prepend: String prepended to the input specification in the keyword line.

        Notes:
            - Writes one file containing all they keywords to be added to the input file.
            - Writes one file per keyword containing the data lines.
            - Deletes any previous data or keyword files in the output directory upon construction.
            - If input_path_prepend is not empty, then the keyword line is modified as follows:
                *Equation, input=<input_path_prepend><filename>
            E.g., if input_path_prepend='constr_files/', then the keyword line will be:
                *Equation, input=constr_files/<filename>
            This parameter is useful to specify the absolute path of the files or a relative path to the
            Abaqus working directory.
        """
        AbaqusWriter.__init__(self, output_dir)
        self.input_path_prepend = input_path_prepend
        self.DATAFILE_BASE = 'Constr_Eqn_Def_'
        self.KEYWFILE_BASE = 'Equation_Keywords.txt'
        self._clear_output()
        return

    def write(self, couplings):
        """ Writes datafiles for the provided couplings.
        :param list couplings: [Coupling] The couplings to be written to file.

        Notes:
            - The keywords that need to be added are written in a single file.
            - The written files contain the data lines used in the analysis are written to separate files.
        """
        file_list = []
        constraint_num = 0
        for couple in couplings:
            for constraint in couple.constraints:
                # todo: make the filename based on the node and DOF
                filename = self.DATAFILE_BASE + str(constraint_num) + '.txt'
                filepath = os.path.join(self.output_dir, filename)
                file_list.append(self.input_path_prepend + filename)
                self._constraint_file_writer(constraint, filepath)
                constraint_num += 1
        keyword_file = os.path.join(self.output_dir, self.KEYWFILE_BASE)
        self._keyword_file_writer(file_list, keyword_file)
        return

    def _constraint_file_writer(self, constraint, file):
        """ Writes the data lines for the *EQUATION keyword for a given constraint.
        :param Constraint constraint: The constraint to write.
        :param str file: Full path to the file to write.
        """
        def term_string(t):
            return ', '.join([str(t.node), str(t.dof), str(t.coef)]) + '\n'
        with open(file, 'w') as f:
            f.write(str(len(constraint.terms)) + '\n')
            for term in constraint.terms:
                f.write(term_string(term))
        return

    def _keyword_file_writer(self, constraint_files, file):
        """ Writes the keyword file. """
        def keyword_string(input):
            return '*Equation, input=' + input + '\n'
        with open(file, 'w') as f:
            for c in constraint_files:
                f.write(keyword_string(c))
        return
