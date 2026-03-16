import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torchsummary import summary
from tqdm import tqdm

from reproduce_experiments.utils.features import compute_codebars
from reproduce_experiments.nn import *


def make_reproducible(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_data(save_directory, class_filter=None, encoder=None, return_miller=False, reduce_to_1_grain=False):
    ang_bins = np.load(Path(save_directory) / '..' / 'MOD_grain_classhkl_angbin.npz')['arr_1']

    data_files = Path(save_directory).glob('*.npz')
    # all_s_tth = []
    # all_s_chi = []
    all_s_miller_ind = []  # Miller indices + last scalar gives info about what material
    all_codebars = []  # Pre-processed frequencies (scaled)
    all_location = []  # Class of miller index per data point in a given laue pattern
    for file in data_files:
        data = np.load(file)
        location = data['location']
        # all_location.append(data['location'])
        remaining_idx = data['remaining_idx']
        s_tth = data['s_tth'][remaining_idx]
        s_chi = data['s_chi'][remaining_idx]
        if reduce_to_1_grain:
            s_grain_id = data['s_grain_id'][remaining_idx]
            mask = (s_grain_id == 0)
            s_tth = s_tth[mask]
            s_chi = s_chi[mask]
            location = location[mask]
        all_location.append(location)
        codebars = compute_codebars(s_tth, s_chi, ang_bins, True)
        all_codebars.append(codebars)
        # all_s_tth.append(data['s_tth'][remaining_idx])
        # all_s_chi.append(data['s_chi'][remaining_idx])
        all_s_miller_ind.append(data['s_miller_ind'][remaining_idx])

    # Stack instances and labels
    X = np.vstack(all_codebars)
    y = np.hstack(all_location)

    # Filter non-frequent classes
    if class_filter is not None:
        # train_mask = np.isin(np.argmax(y, axis=1), class_filter)
        train_mask = np.isin(y, class_filter)
        X = X[train_mask]
        y = y[train_mask]  # Filter rows
        y = y[:, np.any(y, axis=0)]  # Filter columns

    if encoder is None:
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    else:
        y = encoder.transform(y.reshape(-1, 1)).toarray()

    if return_miller:
        return X, y, encoder, all_s_miller_ind
    else:
        return X, y, encoder

def setup_model(model_class, device, input_size, output_size, input_dropout, dims, activation, verbose=True):
    if model_class == 'LaueNN':
        model = LaueNN(input_size, output_size, input_dropout, dims, activation)
    elif model_class == 'CustomNN':
        model = CustomNN(input_size, output_size, input_dropout, dims, activation)
    elif model_class == 'CustomMLP':
        model = CustomMLP(input_size, output_size, input_dropout, dims, activation)
    else:
        print("Model type not recognized. Defaulting to CustomNN.")
        model = CustomNN(input_size, output_size, input_dropout, dims, activation)

    model.to(device)

    if verbose:
        summary(model, (1,input_size))

    return model

def get_predictions_and_labels(model, device, data):
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y_pred = model(x)
            predictions.append(y_pred.cpu().detach())
            labels.append(y.cpu().detach())
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    return predictions, labels

def compute_accuracy(accs):
    counter = 0
    for i in range(len(accs)):
        if accs[i][0] == accs[i][1]:
            counter += 1
    acc = counter / len(accs)
    return acc

def train_model_epoch(model, device, data, epoch, loss_fn, optimizer, scheduler):
    losses_epoch = []
    accs = []

    model.train()
    for x, y in tqdm(data, desc='Epoch {:2d}'.format(epoch + 1)):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.cpu().detach()
        losses_epoch.append(loss)
    if scheduler is not None:
        scheduler.step()

    return losses_epoch

def validate_model_epoch(model, device, data, loss_fn):
    losses_epoch = []
    accs = []

    model.eval()
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss = loss.cpu().detach()
            losses_epoch.append(loss)

            # accuracy
            y_pred_list = torch.argmax(y_pred, dim=1).tolist()
            y_list = torch.argmax(y, dim=1).tolist()
            accs.extend(zip(y_pred_list, y_list))
    acc = compute_accuracy(accs)

    return losses_epoch, acc

def train_and_evaluate_model(model, device, train_dataloader, test_dataloader, epochs, loss_fn, optimizer, scheduler,
                             verbose=True, save_model=False, model_save_directory=None):
    train_losses_epoch = []
    val_losses_epoch = []

    train_accs_epoch = []
    val_accs_epoch = []
    best_val_acc = 0.

    train_losses, train_acc = validate_model_epoch(model, device, train_dataloader, loss_fn)
    val_losses, val_acc = validate_model_epoch(model, device, test_dataloader, loss_fn)

    train_losses_epoch.append(np.mean(train_losses))
    val_losses_epoch.append(np.mean(val_losses))
    train_accs_epoch.append(train_acc)
    val_accs_epoch.append(val_acc)
    if verbose:
        print("Training Loss: {:.4f}".format(np.mean(train_losses)) +
              " | Training Accuracy: {:.4f}".format(train_acc) +
              " | Validation Loss: {:.4f}".format(np.mean(val_losses)) +
              " | Validation Accuracy: {:.4f}".format(val_acc))
        time.sleep(0.1)

    for epoch in range(epochs):
        # training
        _ = train_model_epoch(model, device, train_dataloader, epoch, loss_fn, optimizer, scheduler)
        train_losses, train_acc = validate_model_epoch(model, device, train_dataloader, loss_fn)
        train_losses_epoch.append(np.mean(train_losses))
        train_accs_epoch.append(train_acc)

        # validation
        val_losses, val_acc = validate_model_epoch(model, device, test_dataloader, loss_fn)
        val_losses_epoch.append(np.mean(val_losses))
        val_accs_epoch.append(val_acc)

        if val_acc > best_val_acc and save_model:
            best_val_acc = val_acc
            # TODO: Use file ending .pt
            torch.save(model.state_dict(), model_save_directory)

        if verbose:
            print("Training Loss: {:.4f}".format(np.mean(train_losses)) +
                  " | Training Accuracy: {:.4f}".format(train_acc) +
                  " | Validation Loss: {:.4f}".format(np.mean(val_losses)) +
                  " | Validation Accuracy: {:.4f}".format(val_acc))
            time.sleep(0.1)

    if not verbose:
        print("Training Loss: {:.4f}".format(np.mean(train_losses)) +
              " | Training Accuracy: {:.4f}".format(train_acc) +
              " | Validation Loss: {:.4f}".format(np.mean(val_losses)) +
              " | Validation Accuracy: {:.4f}".format(val_acc))
        time.sleep(0.1)

    return train_losses_epoch, val_losses_epoch, train_accs_epoch, val_accs_epoch



class CodeTimer:
    """
    Context manager to measure execution time of code blocks.
    Can optionally print the measured time using a user-provided logger
    function and/or store the raw elapsed time for later use.

    Examples
    --------
    Basic usage:
        with CodeTimer("my task") as t:
            do_something()
        print(t.elapsed)

    Automatically print timing information:
        with CodeTimer("my task", logger=print):
            do_something()

    Store elapsed time in a dictionary:
        timings = {}
        with CodeTimer("step1", store=timings):
            foo()
        with CodeTimer("step2", store=timings):
            bar()
        # timings = {"step1": ..., "step2": ...}

    Write timing information to file:
        with open("timelog.txt", "a") as f:
            with CodeTimer("heavy task",
                           logger=lambda msg: f.write(msg + "\\n")):
                do_heavy_task()

    Parameters
    ----------
    name : str, optional
        Label for the timed code block.
    logger : callable, optional
        Function that receives a formatted timing message
        (e.g. print, list.append, file.write, logging.info).
    store : dict, optional
        Dictionary used to store the raw elapsed time in seconds.
    key : str, optional
        Key under which the elapsed time is stored in `store`
        (defaults to `name`).
    """

    def __init__(self, name=None, logger=None, store=None, key=None, decimals=2):
        self.name = name or ""
        self.logger = logger
        self.store = store
        self.key = key or self.name
        self.decimals = decimals

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self.start

        msg = f"Executed '{self.name}'. Elapsed time: {self.elapsed:.{self.decimals}f}s"

        # Print/log if logger is provided.
        if self.logger:
            self.logger(msg)

        # Store raw value if dict is provided.
        if self.store is not None:
            if isinstance(self.store, dict):
                self.store[self.key] = self.elapsed
            elif isinstance(self.store, list):
                self.store.append(self.elapsed)
            else:
                raise TypeError("store must be dict or list")