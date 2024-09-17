import os


class AbaqusTxtWriter:
    """ Writes imperfections from a components to .txt files. """

    def __init__(self, components):
        """ Constructor.
        :param list components: [IComponent] Components to write imperfections from.
        """
        self.components = components

    def write_imperfections(self, output_file):
        """ Writes the imperfection file.
        :param str output_file: File to be written.
        """
        # 6 decimal precision on the output
        with open(output_file, 'w') as f:
            for c in self.components:
                for nid, imp in c.node_imperfections.items():
                    f.write('{0:d}, {1:0.8f}, {2:0.8f}, {3:0.8f}\n'.format(nid, imp[0], imp[1], imp[2]))

        fname = os.path.basename(output_file)
        print('Usage:\n\t*IMPERFECTION, input=<path>/{0}'.format(fname))
        return
