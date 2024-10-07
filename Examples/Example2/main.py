import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from Model import Regression
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

        # self.inp = self.inp[(self.out[:, 0] < 0.04) & (self.out[:, 1] < 0.07)]
        # self.out = self.out[(self.out[:, 0] < 0.04) & (self.out[:, 1] < 0.07)]

        if split == 'tr':
            scaler = MinMaxScaler()
            self.inp = scaler.fit_transform(self.inp)
            joblib.dump(scaler, os.path.join(root, 'scaler_params.pkl'))

        else:
            scaler = joblib.load(os.path.join(root, 'scaler_params.pkl'))
            self.inp = scaler.transform(self.inp)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inp[idx, :6]).float(), \
            torch.from_numpy(self.out[idx, 0:1]).float()


def main():
    data_path = 'data_m0'
    save_path = 'data_m0'
    learning_rate, min_lr = 1e-4, 1e-9
    decay_rate = 5e-4
    batch_size = 128
    num_epochs = 5000

    training_set = IMKDataset(data_path, 'tr')
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_set = IMKDataset(data_path, 'vl')
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = Regression()
    device = get_device()

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    model.to(device)
    criterion.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters: ", total_params)

    # try:
    #     trained_weights = torch.load(os.path.join(save_path, f'best_{model_name}.pth'), map_location=device)
    #     model.load_state_dict(trained_weights)
    #     print('Use pretrain model')
    # except:
    #     print('No existing model, starting training from scratch...')

    best_error = np.inf
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pred_list = []
        gt_list = []

        for batch_id, (inp, out) in enumerate(training_loader):

            # pts: shape (bs, 3, 1024); rcs: shape (bs, 2)
            optimizer.zero_grad()
            inp = inp.to(device)
            out = out.to(device)

            # pred: shape (bs, 2); feature (_): shape (bs, 1024, 1)
            pred = model(inp)
            loss = criterion(pred, out)
            pred_list.append(pred)
            gt_list.append(out)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_id % 1000 == 0:
                print(epoch, batch_id, loss.item())

        out_pred = np.vstack([(pred_list[i]).detach().cpu().numpy() for i in range(len(pred_list))])
        out_gt = np.vstack([(gt_list[i]).detach().cpu().numpy() for i in range(len(gt_list))])
        mape_tr = MAPE(out_gt, out_pred)

        # Training loss of the epoch
        scheduler.step()
        training_loss = running_loss / len(training_loader)

        # Validation initiates
        model.eval()
        with torch.no_grad():
            pred_list = []
            gt_list = []
            running_loss = 0.0

            for inp, out in validation_loader:
                inp = inp.to(device)
                out = out.to(device)
                pred = model(inp)
                loss = criterion(pred, out)

                running_loss += loss.item()
                pred_list.append(pred)
                gt_list.append(out)

        out_pred = np.vstack([(pred_list[i]).detach().cpu().numpy() for i in range(len(pred_list))])
        out_gt = np.vstack([(gt_list[i]).detach().cpu().numpy() for i in range(len(gt_list))])
        mape_vl = MAPE(out_gt, out_pred)
        validation_loss = running_loss / len(validation_loader)

        print(
            'Epoch {}, Training Loss: {:.6f}, Validation Loss: {:.6f}, Training MAPE: {:.3f}, Validation MAPE: {:.3f}'.format(
                epoch, training_loss, validation_loss, mape_tr, mape_vl))

        if best_error > mape_vl:
            best_error = mape_vl
            print("Model saving...")
            torch.save(model.state_dict(), os.path.join(save_path, 'bestAlr.pth'))


if __name__ == "__main__":
    main()
