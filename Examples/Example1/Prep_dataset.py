import numpy as np
import h5py

root = 'data\\'
h5_file = h5py.File('column.hdf5', "r")
test_section = h5_file.attrs['test']
validation_section = h5_file.attrs['validation']
training_section = h5_file.attrs['training']

ts_pt0, vl_pt0, tr_pt0 = [], [], []
ts_pt1, vl_pt1, tr_pt1 = [], [], []
ts_rc, vl_rc, tr_rc = [], [], []
ts_name, vl_name, tr_name = [], [], []
ts_fc, vl_fc, tr_fc = [], [], []
ts_dr, vl_dr, tr_dr = [], [], []

for i1 in h5_file.keys():
    for i2 in h5_file[i1].keys():
        for i3 in h5_file[i1][i2].keys():
            for i4 in h5_file[i1][i2][i3].keys():
                for i5 in h5_file[i1][i2][i3][i4].keys():
                    model = h5_file[i1][i2][i3][i4][i5]
                    name = i1 + '-' + i2 + '-' + i3 + '-' + i4 + '-' + i5
                    print(name)

                    # Obtain normalizing factors
                    d = model.attrs['d']
                    bf = model.attrs['b_f']
                    tf = model.attrs['t_f']
                    fc = np.array([bf, d - tf, d])

                    # Obtain normalizing factors
                    ind = model['indices'][:]
                    pcs = model['Deformed_shape']['processed_center'][:]
                    rcs = model['Reaction']['reserve_capacity'][ind]
                    drs = np.abs(model['Displacement']['top_U2'][ind] / int(i2))

                    # Error test
                    if pcs.shape[0] != rcs.shape[0]:
                        print(name)
                        raise ValueError

                    # Categorize sections
                    if i1 in test_section:
                        ind_pts = 0
                    elif i1 in validation_section:
                        ind_pts = 1
                    elif i1 in training_section:
                        ind_pts = 2
                    else:
                        raise KeyError

                    # Save data
                    for i in range(pcs.shape[0]):
                        pc0, pc1, rc, dr = pcs[i, 0, ...], pcs[i, 1, ...], rcs[i], drs[i]

                        if ind_pts == 0:
                            ts_pt0.append(pc0)
                            ts_pt1.append(pc1)
                            ts_rc.append(rc)
                            ts_name.append(name)
                            ts_fc.append(fc)
                            ts_dr.append(dr)

                        elif ind_pts == 1:
                            vl_pt0.append(pc0)
                            vl_pt1.append(pc1)
                            vl_rc.append(rc)
                            vl_name.append(name)
                            vl_fc.append(fc)
                            vl_dr.append(dr)

                        else:
                            tr_pt0.append(pc0)
                            tr_pt1.append(pc1)
                            tr_rc.append(rc)
                            tr_name.append(name)
                            tr_fc.append(fc)
                            tr_dr.append(dr)


np.save(root + 'ts_pt0.npy', np.array(ts_pt0))
np.save(root + 'vl_pt0.npy', np.array(vl_pt0))
np.save(root + 'tr_pt0.npy', np.array(tr_pt0))
np.save(root + 'ts_pt1.npy', np.array(ts_pt1))
np.save(root + 'vl_pt1.npy', np.array(vl_pt1))
np.save(root + 'tr_pt1.npy', np.array(tr_pt1))

np.save(root + 'ts_rc.npy', np.array(ts_rc))
np.save(root + 'vl_rc.npy', np.array(vl_rc))
np.save(root + 'tr_rc.npy', np.array(tr_rc))

np.save(root + 'ts_name.npy', ts_name)
np.save(root + 'vl_name.npy', vl_name)
np.save(root + 'tr_name.npy', tr_name)

np.save(root + 'ts_fc.npy', np.array(ts_fc))
np.save(root + 'vl_fc.npy', np.array(vl_fc))
np.save(root + 'tr_fc.npy', np.array(tr_fc))

np.save(root + 'ts_dr.npy', np.array(ts_dr))
np.save(root + 'vl_dr.npy', np.array(vl_dr))
np.save(root + 'tr_dr.npy', np.array(tr_dr))
