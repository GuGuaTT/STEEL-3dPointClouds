import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import numpy as np

df = 'data\\'
tr_data, tr_rc = np.load(df + 'tr_dr.npy'), np.load(df + 'tr_rc.npy')
vl_data, vl_rc = np.load(df + 'vl_dr.npy'), np.load(df + 'vl_rc.npy')
ts_data, ts_rc = np.load(df + 'ts_dr.npy'), np.load(df + 'ts_rc.npy')

tr_data = np.append(tr_data, vl_data)
tr_rc = np.concatenate((tr_rc, vl_rc))

tr_data = tr_data.reshape(-1, 1)
ts_data = ts_data.reshape(-1, 1)

bt = BallTree(tr_data, leaf_size=30, metric='euclidean')
ind = bt.query(ts_data, k=5, return_distance=False)
pr_rc = np.mean(tr_rc[ind], axis=1)
np.save(df + 'pr_rcd.npy', pr_rc)

err = np.abs(pr_rc - ts_rc) / pr_rc
err_rcp = np.mean(err[:, 0])
err_rcn = np.mean(err[:, 1])

print(err_rcp, err_rcn)

x = np.linspace(0, 1, 10)
plt.scatter(ts_rc[:, 0], pr_rc[:, 0], alpha=0.3, c='k', s=15)
plt.plot(x, x, c='r', lw=0.5)
plt.show()
