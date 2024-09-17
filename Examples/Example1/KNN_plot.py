import matplotlib.pyplot as plt
import numpy as np


def plot_sorted_xy(x, y1, y2, y3, y4, rc_idx):

    y1 = np.abs((x - y1) / x) * 100
    y2 = np.abs((x - y2) / x) * 100
    y3 = np.abs((x - y3) / x) * 100
    y4 = np.abs((x - y4) / x) * 100
    bins = np.arange(0.4, 1.02, 0.02)
    bin_indices = np.digitize(x, bins)

    medians1, medians2, medians3, medians4 = [], [], [], []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_y1, bin_y2, bin_y3, bin_y4 = y1[bin_mask], y2[bin_mask], y3[bin_mask], y4[bin_mask]
            medians1.append(np.median(bin_y1))
            medians2.append(np.median(bin_y2))
            medians3.append(np.median(bin_y3))
            medians4.append(np.median(bin_y4))
        else:
            medians1.append(np.nan)
            medians2.append(np.nan)
            medians3.append(np.nan)
            medians4.append(np.nan)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    rc_name = 'True $R_M^{(+)}$' if rc_idx == 0 else 'True $R_M^{(-)}$'
    ax.set_xlabel(rc_name, fontsize=19, labelpad=10)
    ax.set_ylabel('MAPE (%)', fontsize=19, labelpad=10)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 75)

    ax.plot(bins[1:], medians1, linestyle='--', linewidth=2, label='Top', color='k')
    ax.plot(bins[1:], medians2, linestyle='-', linewidth=2, label='Bottom', color='k')
    ax.plot(bins[1:], medians3, linestyle=':', linewidth=2, label='Top+Bottom', color='k')
    ax.plot(bins[1:], medians4, linestyle='-.', linewidth=2, label='Drift', color='k')

    ax.legend(loc='best', framealpha=1.0, borderpad=0.5, edgecolor='k', fancybox=False, prop={'size': 17})
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    # plt.savefig('img0.jpg', dpi=300)
    plt.show()


df = 'data\\'
ts_rc = np.load(df + 'ts_rc.npy')
pr_rct = np.load(df + 'pr_rct.npy')
pr_rcb = np.load(df + 'pr_rcb.npy')
pr_rca = np.load(df + 'pr_rca.npy')
pr_rcd = np.load(df + 'pr_rcd.npy')
plot_sorted_xy(ts_rc[:, 0], pr_rct[:, 0], pr_rcb[:, 0], pr_rca[:, 0], pr_rcd[:, 0], 0)
