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
    def __init__(self, root, scaler, split='tr'):
        self.inp = np.load(os.path.join(root, split + '_input.npy'))
        self.out = np.load(os.path.join(root, split + '_theta.npy'))

        self.inp = self.inp[(self.out[:, 0] < 0.04) & (self.out[:, 1] < 0.07)]
        self.out = self.out[(self.out[:, 0] < 0.04) & (self.out[:, 1] < 0.07)]

        if split == 'tr':
            scaler = MinMaxScaler()
            self.inp = scaler.fit_transform(self.inp)
            joblib.dump(scaler, 'data/scaler_params.pkl')

        else:
            scaler = joblib.load('data/scaler_params.pkl')
            self.inp = scaler.transform(self.inp)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inp[idx, :6]).float(), \
               torch.from_numpy(self.out[idx, 1:]).float()


def plot_xy(x, y, rc_idx):
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    theta_name = '$\\theta^*_p$' if rc_idx == 0 else '$\\theta^*_{pc}$'
    ax.set_xlabel(f'True {theta_name}', fontsize=19, labelpad=10)
    ax.set_ylabel(f'Predicted {theta_name}', fontsize=19, labelpad=10)
    ax.set_xlim(0.0, 0.08)
    ax.set_ylim(0.0, 0.08)

    x_ = np.linspace(0, 0.1, 10)
    plt.scatter(x, y, alpha=0.3, c='k', s=15)
    plt.plot(x_, x_, c='r', lw=0.5)

    # ticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig('img.jpg', dpi=300)
    plt.show()


def main():
    data_path = 'data'
    save_path = 'data'
    batch_size = 32

    test_set = IMKDataset(data_path, 'ts')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    device = get_device()
    model = Regression()
    model.to(device)

    trained_weights = torch.load(os.path.join(save_path, f'best.pth'), map_location=device)
    model.load_state_dict(trained_weights)

    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []

        for batch_id, (inp, out) in enumerate(test_loader):
            inp = inp.to(device)
            out = out.to(device)
            pred = model(inp)

            pred_list.append(pred)
            gt_list.append(out)

    out_pred = np.vstack([(pred_list[i]).detach().cpu().numpy() for i in range(len(pred_list))])
    out_gt = np.vstack([(gt_list[i]).detach().cpu().numpy() for i in range(len(gt_list))])

    print(MAPE(out_gt, out_pred))
    plot_xy(out_gt, out_pred, 1)


if __name__ == "__main__":
    main()
