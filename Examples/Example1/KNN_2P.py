import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import numpy as np


df = 'data\\'
tr_pt0, tr_pt1, tr_rc, tr_fc = (
    np.load(df + 'tr_pt0.npy'), np.load(df + 'tr_pt1.npy'), np.load(df + 'tr_rc.npy'), np.load(df + 'tr_fc.npy'))
vl_pt0, vl_pt1, vl_rc, vl_fc = (
    np.load(df + 'vl_pt0.npy'), np.load(df + 'vl_pt1.npy'), np.load(df + 'vl_rc.npy'), np.load(df + 'vl_fc.npy'))
ts_pt0, ts_pt1, ts_rc, ts_fc = (
    np.load(df + 'ts_pt0.npy'), np.load(df + 'ts_pt1.npy'), np.load(df + 'ts_rc.npy'), np.load(df + 'ts_fc.npy'))

tr_data = np.concatenate((tr_pt0, tr_pt1), axis=1)
vl_data = np.concatenate((vl_pt0, vl_pt1), axis=1)
tr_data = np.concatenate((tr_data, vl_data))
tr_rc = np.concatenate((tr_rc, vl_rc))
tr_fc = np.concatenate((tr_fc, vl_fc))

ts_data = np.concatenate((ts_pt0, ts_pt1), axis=1)
tr_data = (tr_data / tr_fc[:, None, :]).reshape(-1, 882 * 3)
ts_data = (ts_data / ts_fc[:, None, :]).reshape(-1, 882 * 3)

bt = BallTree(tr_data, leaf_size=30, metric='euclidean')
ind = bt.query(ts_data, k=5, return_distance=False)
pr_rc = np.mean(tr_rc[ind], axis=1)
np.save(df + 'pr_rca.npy', pr_rc)

err = np.abs(pr_rc - ts_rc) / pr_rc
err_rcp = np.mean(err[:, 0])
err_rcn = np.mean(err[:, 1])

print(err_rcp, err_rcn)

x = np.linspace(0.4, 1, 10)
plt.scatter(ts_rc[:, 0], pr_rc[:, 0], alpha=0.3, c='k', s=15)
plt.plot(x, x, c='r', lw=0.5)
plt.show()
