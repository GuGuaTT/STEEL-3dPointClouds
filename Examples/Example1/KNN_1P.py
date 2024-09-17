import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import numpy as np


def plot_xy(x, y, rc_idx):

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    rc_name = '$R_M^{(+)}$' if rc_idx == 0 else '$R_M^{(-)}$'
    ax.set_xlabel(f'True {rc_name}', fontsize=19, labelpad=10)
    ax.set_ylabel(f'Predicted {rc_name}', fontsize=19, labelpad=10)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)

    x_ = np.linspace(0.4, 1, 10)
    plt.scatter(x, y, alpha=0.3, c='k', s=15)
    plt.plot(x_, x_, c='r', lw=0.5)

    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    # plt.savefig('img_1.jpg', dpi=300)
    plt.show()


if __name__ == "__main__":

    df = 'data\\'
    tr_data, tr_rc, tr_fc = np.load(df + 'tr_pt1.npy'), np.load(df + 'tr_rc.npy'), np.load(df + 'tr_fc.npy')
    vl_data, vl_rc, vl_fc = np.load(df + 'vl_pt1.npy'), np.load(df + 'vl_rc.npy'), np.load(df + 'vl_fc.npy')
    ts_data, ts_rc, ts_fc = np.load(df + 'ts_pt1.npy'), np.load(df + 'ts_rc.npy'), np.load(df + 'ts_fc.npy')

    tr_data = np.concatenate((tr_data, vl_data))
    tr_rc = np.concatenate((tr_rc, vl_rc))
    tr_fc = np.concatenate((tr_fc, vl_fc))

    tr_data = (tr_data / tr_fc[:, None, :]).reshape(-1, 441 * 3)
    ts_data = (ts_data / ts_fc[:, None, :]).reshape(-1, 441 * 3)

    bt = BallTree(tr_data, leaf_size=30, metric='euclidean')
    ind = bt.query(ts_data, k=5, return_distance=False)
    pr_rc = np.mean(tr_rc[ind], axis=1)
    np.save(df + 'pr_rct.npy', pr_rc)
    # np.save(df + 'pr_rcb.npy', pr_rc)

    err = np.abs(pr_rc - ts_rc) / pr_rc
    err_rcp = np.mean(err[:, 0])
    err_rcn = np.mean(err[:, 1])
    print(err_rcp, err_rcn)

    plot_xy(ts_rc[:, 1], pr_rc[:, 1], 1)



