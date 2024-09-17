import numpy as np
import matplotlib.pyplot as plt
import h5py


h5_file = h5py.File('column_s.hdf5', "r")
W_list, S_list, T_list, F_list = np.array([]), np.array([]), np.array([]), np.array([])

for i1 in h5_file.keys():
    for i2 in h5_file[i1].keys():
        for i3 in h5_file[i1][i2].keys():
            for i4 in h5_file[i1][i2][i3].keys():
                for i5 in h5_file[i1][i2][i3][i4].keys():
                    model = h5_file[i1][i2][i3][i4][i5]

                    ind = 1
                    ids = model['indices'][:]
                    pts = model['Deformed_shape']['processed_center'][ind, 0, ...]
                    rm = model['Reaction']['bot_RM1'][:]
                    ax = model['Displacement']['top_U3'][:]
                    d, bf = model.attrs['d'], model.attrs['b_f']

                    if True:  # ids[ind] == np.argmax(np.abs(rm))

                        # Obtain the web points
                        selected_indices = []
                        for i in range(21):
                            start_index = 6 + i * 21
                            selected_indices.extend(range(start_index, start_index + 9))
                        ptsw = pts[selected_indices, :].reshape(21, 9, 3)

                        # Obtain the W value
                        ptsw1 = np.abs(ptsw - ptsw[:, 0:1, :])
                        ptsw2 = np.abs(ptsw - ptsw[:, -1:, :])
                        W = max(np.max(ptsw1[:, :, 0]), np.max(ptsw2[:, :, 0]))

                        # Obtain the S value
                        S = abs(ax[ids[ind]])

                        # Obtain the T value
                        pts1 = pts[0::21, :]
                        pts2 = pts[5::21, :]
                        pts3 = pts[15::21, :]
                        pts4 = pts[20::21, :]

                        diff1 = np.abs((pts1 - pts2)[:, 1])
                        diff2 = np.abs((pts3 - pts4)[:, 1])
                        T = max(np.max(diff1), np.max(diff2))

                        # Obtain the F value
                        pts5 = (pts[2::21, :] + pts[3::21, :]) / 2
                        pts6 = (pts[17::21, :] + pts[18::21, :]) / 2

                        diff11 = np.abs((pts1 - pts5)[:, 1])
                        diff22 = np.abs((pts2 - pts5)[:, 1])
                        diff33 = np.abs((pts3 - pts6)[:, 1])
                        diff44 = np.abs((pts4 - pts6)[:, 1])
                        F = max(np.max(diff11), np.max(diff22), np.max(diff33), np.max(diff44))

                        S_list = np.append(S_list, S)
                        W_list = np.append(W_list, W / d)
                        F_list = np.append(F_list, F / bf * 2)
                        T_list = np.append(T_list, T / bf)


# W, 45, 2.08, 0.08, 0.0036; F, 25, 0.96, 0.14, 0.0075; T, 45, 1.87, 0.125, 0.0073; S, 50, 4.10
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

name = '$2F / b_f$'
ax.set_xlabel(name, fontsize=19, labelpad=10)
ax.set_ylabel('Frequency', fontsize=19, labelpad=15)

ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax.set_xlim(0.0, 0.14)

val = np.quantile(F_list, 0.1)
print(val)
ax.hist(F_list[F_list < 0.14], bins=30, color='dimgrey')
ax.axvline(val, color='r', linestyle='-.', label='90% confidence threshold', lw=2.0)
ax.legend(loc='best', framealpha=1.0, borderpad=0.5, edgecolor='k', fancybox=False, prop={'size': 17})
plt.tight_layout()
plt.savefig('image/imgf_.jpg', dpi=300)
plt.show()
