# TODO: Clean up imports
from pathlib import Path
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import sys
from importlib import reload
import matplotlib.gridspec as gridspec
from functools import lru_cache
import datetime
import pprint 
import traceback

def train(config):
    try:
        trainset = PTBXLDataset(file=config["data_file"], data_type=config["data_train_key"])
        train_loader = Data.DataLoader(trainset, 
            batch_size = config["train_batch_size"], 
            shuffle = True
        )

        net = config["model_class"](**config["model_kwargs"]).to(config["device"])
        optimizer = config["train_optimizer_class"](
            net.parameters(), **config["train_optimizer_kwargs"]
        )
        criterion = config["train_loss_class"]()

        loss_history = []
        accuracy_history = []
        
        print("Starting training", datetime.datetime.now())
        for epoch in range(config["train_epoches"]):  # loop over the dataset multiple times

            # for i, data in enumerate(trainloader, 0):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_y_cpu = batch_y.cpu()
                batch_x, batch_y = (
                    batch_x.type(config["X_type"]).to(config["device"]), 
                    batch_y.type(config["y_type"]).to(config["device"])
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(batch_x)

                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                # Print out
                loss_history.append(loss.item())
                # threshold = 0.5
                # batch_y_pred = outputs > threshold
                # correct = np.count_nonzero((batch_y_pred == batch_y).cpu(), axis=0)
                # accuracy = correct / len(batch_y)
                # accuracy_history.append(accuracy.tolist())
                sensitivity_specificity_curves = get_sensitivity_specificity_curves(batch_y_cpu.numpy(), outputs.cpu().detach().numpy())
                auc = np.trapz(x=1-sensitivity_specificity_curves[..., 1], y=sensitivity_specificity_curves[..., 0], axis=1)
                accuracy_history.append(auc)

                if i % config["train_print_interval"] == 0:
                    print(#'[%d, %5d] loss: %.3f, accuracy: %s' %
                        '[%d, %5d] loss: %.3f, AUC: %s' %
                        (epoch + 1, i + 1, 
                        np.mean(loss_history[-100:]),
                        np.mean(accuracy_history[-100:], axis=0))
                        )

        print('Finished Training', datetime.datetime.now())
    except Exception as e:
        traceback.print_exc()
    finally:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config
            }, config["save_model_file"])
        except Exception as e:
            traceback.print_exc()
        
        return locals()

def get_prediction(saved_file=None, data_type="test", device=None, dataset=None):
    """
    Load the saved model given the configuration
    Generate the predictions from the model 
    against the dataset given by data_type

    Returns:
        dataset : the dataset corresponding to data_type
        y_pred : the predicted y array of probabilities (model output)

    """
    # Load model from file
    loaded = torch.load(saved_file)

    if device is None:
        device = loaded["config"]["device"]

    net = loaded["config"]["model_class"](**loaded["config"]["model_kwargs"])
    net.load_state_dict(loaded["model_state_dict"])
    net.to(device)
    # Turn on evaluation mode
    net.eval()

    if dataset is None:
        dataset = PTBXLDataset(file=loaded["config"]["data_file"], data_type=data_type)
    loader = Data.DataLoader(dataset, 
        batch_size = 512, 
        shuffle = False
    )

    y_pred = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = (
                    batch_x.type(loaded["config"]["X_type"]).to(device), 
                    batch_y.type(loaded["config"]["y_type"]).to(device)
                )
            
            batch_y_pred = net(batch_x)
            y_pred.append(batch_y_pred.cpu().numpy())

    y_pred = np.concatenate(y_pred)

    return dataset, y_pred



class PTBXLDataset(Data.Dataset):
    @staticmethod
    @lru_cache(maxsize=None)
    def load_data(file, key):
        """
        Load data with cache
        """
        print(f"Loading file = {file} key = {key}")
        data_packed = np.load(file)
        return data_packed[key]

    def __init__(self, 
                 file="drive/MyDrive/data/ptbxl/export/rhythm.npz",
                 data_type="train",
                 X=None,
                 y=None):
        # print(f"Loading {file}")
        # data_packed = np.load(file)
        # self.X = data_packed[f'X_{data_type}']
        # self.y = data_packed[f'y_{data_type}']
        self.file = file
        if X is None:
            self.X = PTBXLDataset.load_data(file, f'X_{data_type}')
        else:
            self.X = X
        if y is None:
            self.y = PTBXLDataset.load_data(file, f'y_{data_type}')
        else:
            self.y = y

        # Load in description file for each class
        self.statements = pd.read_csv(file.parents[1]  / "scp_statements.csv")
        # filter only the ones corresponding to the data file
        # e.g. file.stem is "rhythm"
        if file.stem in self.statements:
            self.statements = self.statements[self.statements[file.stem]==1]

        self.data_type = data_type

        assert len(self.X) == len(self.y), "Dataset X and y must be of same length"

    def __getitem__(self, idx):
        """
        Returns tuple of data, label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.X[idx] # .astype(np.float32)
        label = self.y[idx]

        return data, label

    def __len__(self):
        return len(self.X)

    # For compatibility with old code
    @property
    def X_train(self):
        return self.X
    
    @property
    def y_train(self):
        return self.y
    ## For compatibility with old code end

    def label_counts(self, normalize=False):
        y = pd.DataFrame(self.y, columns=self.statements["description"])
        y = y.apply(pd.value_counts, normalize=normalize)
        return y

    @property
    def shape(self):
        return self.X[0].shape, self.y[0].shape


def plot_ecg(X, figsize=(12, 9), **kwargs):

    fig, axes = plt.subplots(3, 4, sharex="col", sharey=True, figsize=figsize, **kwargs)

    for i, (trace, ax, title) in enumerate(zip(X.T, axes.T.flat, ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])):
        
        t = np.array([0, 2.5]) + (i//3) * 2.5
        ts = np.arange(*t, step=0.01)
        idx = np.arange(0, 250) + (i//3) * 250
        # print(i, t, ts, idx)

        ax.plot(ts, trace[idx], "-k", clip_on=False)
        ax.set_yticks(np.arange(-2, 2, 0.1), minor=True)
        ax.set_xticks(np.arange(0, 10, 0.20), minor=True)
        ax.grid(True, "major", "y", color="m", alpha=0.5)
        ax.grid(True, "both", "x", color="m", alpha=0.5)
        ax.set_title(title, x=0.1, y=0.9)
        ax.spines['bottom'].set_color("m")
        ax.spines['top'].set_color("m")
        if (i//3) == 0:
            ax.set_ylabel("mv")
        if i == 11:
            ax.set_xlabel('sec') # loc="right"
            ax.xaxis.set_label_coords(1, -0.05)
        ax.set(xlim=t, ylim=(-2, 2),
            yticks=np.arange(-2, 2, 0.5),
            xticks=np.arange(0, 10, step=0.80))
        # ax.set_xticklabels([("%.1f"%i) for i in np.arange(*t, step=0.20)])
        # ax.set_yticklabels([])
        # break

    fig.subplots_adjust(wspace=0, hspace=0)


    return fig, axes

def get_sensitivity_specificity(pred, truth):
    """
    Input: booleans of predictions, truth 1D numpy arrays
    Returns: sensitivity, specificity
    """
    pred = pred.astype(np.bool)
    truth = truth.astype(np.bool)

    positive_cases_n = np.count_nonzero(truth)
    negative_cases_n = len(truth) - positive_cases_n

    true_positive_n = np.count_nonzero(pred[truth])
    # false_negative_n = positive_cases_n - true_positive_n

    false_positive_n = np.count_nonzero(pred[~truth])
    true_negative_n = negative_cases_n - false_positive_n

    sensitivity = true_positive_n / positive_cases_n if positive_cases_n != 0 else np.nan
    specificity = true_negative_n / negative_cases_n if negative_cases_n != 0 else np.nan

    return sensitivity, specificity


def get_sensitivity_specificity_curves(truth, pred, thresholds=None):
    """
    Parameters
    -----------------
    truth, pred : n x n_classes

    Returns
    -----------------
    n x len(thresholds) x 2
        0th channel is sensitivity
        1th channel is specificity
        See get_sensitivity_specificity
    """
    if thresholds is None:
        thresholds = np.concatenate([[0], 1 / (1 + np.exp(-np.linspace(-10, 10, 1001))), [1]])
        thresholds = thresholds[::-1]
    # Just believe this one liner works - 
    # or just try to understand it...
    sensitivity_specificity_curves =  np.array([[get_sensitivity_specificity(pred=pre>thresh, truth=tru
                                   ) for thresh in thresholds
                        ] for (tru, pre) in zip(truth.T, pred.T)])
    # To plot ROC:
    # plt.plot(1-sensitivity_specificity_curves[..., 1].T, sensitivity_specificity_curves[..., 0].T)
    # To get AUC:
    # auc = np.trapz(x=1-sensitivity_specificity_curves[..., 1], y=sensitivity_specificity_curves[..., 0], axis=1)
    return sensitivity_specificity_curves
