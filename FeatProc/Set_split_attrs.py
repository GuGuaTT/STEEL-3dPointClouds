import h5py
import numpy as np

h5_file = h5py.File('/media/gary/26cb7fa1-da43-4e9a-a179-4d5b9de2a4d3/MLDB/column_s.hdf5', "a")
k = h5py.string_dtype(encoding='utf-8')
# print(h5_file.attrs['validation'])

test_set = np.array(['W21X147', 'W30X173', 'W33X263', 'W27X129', 'W14X99', 'W27X217', 'W21X111', 'W30X116'])
h5_file.attrs.create('test', test_set, dtype=k)
val_set = np.array(['W18X106', 'W33X201', 'W30X211', 'W36X210', 'W14X109', 'W24X176', 'W18X86', 'W30X124'])
h5_file.attrs.create('validation', val_set, dtype=k)

training_set = np.array([], dtype=object)
for section in h5_file.keys():
    if section not in test_set and section not in val_set:
        training_set = np.append(training_set, section)

h5_file.attrs.create('training', training_set, dtype=k)
