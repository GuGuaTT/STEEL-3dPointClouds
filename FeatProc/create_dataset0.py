import numpy as np
import matplotlib.pyplot as plt
import h5py

root = 'ML_pts/'
# dataset = '/media/gary/26cb7fa1-da43-4e9a-a179-4d5b9de2a4d3/MLDB/column_s.hdf5'
# h5_file = h5py.File(dataset, "r")
#
# test_section = h5_file.attrs['test']
# validation_section = h5_file.attrs['validation']
# training_section = h5_file.attrs['training']
#
# ts_pts, vl_pts, tr_pts = [], [], []
# ts_rc, vl_rc, tr_rc = [], [], []
# ts_name, vl_name, tr_name = [], [], []
#
# for i1 in h5_file.keys():
#     for i2 in h5_file[i1].keys():
#         for i3 in h5_file[i1][i2].keys():
#             for i4 in h5_file[i1][i2][i3].keys():
#                 for i5 in h5_file[i1][i2][i3][i4].keys():
#
#                     model = h5_file[i1][i2][i3][i4][i5]
#                     name = i1 + '-' + i2 + '-' + i3 + '-' + i4 + '-' + i5
#                     print(name)
#                     pcs = model['pc_local'][:]
#                     ind_pcs = model['indices'][:]
#                     rcs = model['Reaction']['reserve_capacity'][ind_pcs]
#
#                     if pcs.shape[0] != rcs.shape[0]:
#                         print(name)
#                         raise ValueError
#
#                     if i1 in test_section:
#                         ind_pts = 0
#                     elif i1 in validation_section:
#                         ind_pts = 1
#                     elif i1 in training_section:
#                         ind_pts = 2
#                     else:
#                         raise KeyError
#
#                     for i in range(pcs.shape[0]):
#                         pc, rc = pcs[i, 0, ...], rcs[i]
#
#                         if ind_pts == 0:
#                             ts_pts.append(pc)
#                             ts_rc.append(rc)
#                             ts_name.append(name)
#
#                         elif ind_pts == 1:
#                             vl_pts.append(pc)
#                             vl_rc.append(rc)
#                             vl_name.append(name)
#
#                         else:
#                             tr_pts.append(pc)
#                             tr_rc.append(rc)
#                             tr_name.append(name)
#
#
# np.save(root + 'test_pts0.npy', np.array(ts_pts))
# np.save(root + 'validation_pts0.npy', np.array(vl_pts))
# np.save(root + 'training_pts0.npy', np.array(tr_pts))
#
# np.save(root + 'test_rc0.npy', np.array(ts_rc))
# np.save(root + 'validation_rc0.npy', np.array(vl_rc))
# np.save(root + 'training_rc0.npy', np.array(tr_rc))
#
# np.save(root + 'test_name0.npy', ts_name)
# np.save(root + 'validation_name0.npy', vl_name)
# np.save(root + 'training_name0.npy', tr_name)

j = np.load(root + 'training_pts0.npy')
print(j.shape)

ax = plt.subplot(projection='3d')
ax.scatter(j[1, :, 0], j[1, :, 1], j[1, :, 2], c='red', s=5)
plt.tight_layout()
plt.show()

