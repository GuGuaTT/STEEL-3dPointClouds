

import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from Model import Regression
import matplotlib.pyplot as plt
import joblib


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def MAPE(y_true, y_pred):
    abs_percentage_error = np.abs((y_true - y_pred) / y_true) * 100
    mean_mape = np.mean(abs_percentage_error)
    return mean_mape


class IMKDataset(Dataset):
    def __init__(self, root, split='tr'):
        self.inp = np.load(os.path.join(root, split + '_input.npy'))
        self.out = np.load(os.path.join(root, split + '_theta.npy'))

        self.inp = self.inp[(self.out[:, 0] < 0.04) & (self.out[:, 1] < 0.07)]
        self.out = self.out[(self.out[:, 0] < 0.04) & (self.out[:, 1] < 0.07)]

        scaler = joblib.load(os.path.join(root, 'scaler_params.pkl'))
        self.inp = scaler.transform(self.inp)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inp[idx, :6]).float(), \
            torch.from_numpy(self.out[idx, 0:1]).float()


def plot_xy(x_tr, y_tr, x_vl, y_vl, x_ts, y_ts, rc_idx):

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    theta_name = '$\\theta_p$' if rc_idx == 0 else '$\\theta_{pc}$'
    ax.set_xlabel(f'True {theta_name}', fontsize=19, labelpad=10)
    ax.set_ylabel(f'Predicted {theta_name}', fontsize=19, labelpad=10)
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.8)

    x_ = np.linspace(0, 1, 10)
    plt.scatter(x_tr, y_tr, alpha=0.3, c='k', s=15, label='Training Set')
    plt.scatter(x_vl, y_vl, alpha=0.3, c='b', s=15, label='Validation Set')
    plt.scatter(x_ts, y_ts, alpha=0.3, c='r', s=15, label='Test Set')
    plt.plot(x_, x_, c='k', linestyle='--', lw=1.0)

    # ticks = [0.0, 0.04, 0.08, 0.12]
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.legend(fontsize=14, edgecolor='black', fancybox=False)
    plt.tight_layout()
    # plt.savefig('img1.jpg', dpi=300)
    plt.show()


def main():

    data_path = 'data_s0'
    save_path = 'data_s0'
    batch_size = 32

    training_set = IMKDataset(data_path, 'tr')
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_set = IMKDataset(data_path, 'vl')
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set = IMKDataset(data_path, 'ts')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_device()
    model = Regression()
    model.to(device)

    trained_weights = torch.load(os.path.join(save_path, f'bestAlr.pth'), map_location=device)
    model.load_state_dict(trained_weights)

    model.eval()
    with torch.no_grad():

        pred_tr, pred_vl, pred_ts = [], [], []
        gt_tr, gt_vl, gt_ts = [], [], []

        for batch_id, (inp, out) in enumerate(training_loader):
            inp = inp.to(device)
            out = out.to(device)
            pred = model(inp)

            pred_tr.append(pred)
            gt_tr.append(out)

        for batch_id, (inp, out) in enumerate(validation_loader):
            inp = inp.to(device)
            out = out.to(device)
            pred = model(inp)

            pred_vl.append(pred)
            gt_vl.append(out)

        for batch_id, (inp, out) in enumerate(test_loader):
            inp = inp.to(device)
            out = out.to(device)
            pred = model(inp)

            pred_ts.append(pred)
            gt_ts.append(out)

    p_tr = np.vstack([(pred_tr[i]).detach().cpu().numpy() for i in range(len(pred_tr))])
    g_tr = np.vstack([(gt_tr[i]).detach().cpu().numpy() for i in range(len(gt_tr))])
    p_vl = np.vstack([(pred_vl[i]).detach().cpu().numpy() for i in range(len(pred_vl))])
    g_vl = np.vstack([(gt_vl[i]).detach().cpu().numpy() for i in range(len(gt_vl))])
    p_ts = np.vstack([(pred_ts[i]).detach().cpu().numpy() for i in range(len(pred_ts))])
    g_ts = np.vstack([(gt_ts[i]).detach().cpu().numpy() for i in range(len(gt_ts))])

    print(MAPE(g_tr, p_tr))
    print(MAPE(g_ts, p_ts))

    # plot_xy(g_tr, p_tr, g_vl, p_vl, g_ts, p_ts, 1)


if __name__ == "__main__":
    main()
