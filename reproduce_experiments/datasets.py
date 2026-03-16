import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LaueDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.points_per_image = []
        self.files = list(data_path.glob('*.npz'))
        self.classes = set()
        for file in self.files:
            data = np.load(file)
            self.points_per_image.append(len(data['codebars']))
            self.classes.update(data['location'])
        self.points_per_image = np.array(self.points_per_image)
        self.n_features = np.load(self.files[0])['codebars'].shape[1]

    def __len__(self):
        return sum(self.points_per_image)

    def __getitem__(self, idx):
        file_idx = (np.cumsum(self.points_per_image) <= idx).sum()
        if file_idx == 0:
            idx_in_file = idx
        else:
            idx_in_file = idx - np.cumsum(self.points_per_image)[file_idx-1]
        data = np.load(self.files[file_idx])
        x = torch.from_numpy(data['codebars'][idx_in_file].astype(np.float32))
        y = data['location'][idx_in_file]
        return x, y


class EvalDataset(Dataset):
    def __init__(self, data_path, give_us_all=True, give_us_some=True):
        self.data_path = data_path
        print(f"Data path: {data_path}")
        self.files = natsorted(data_path.glob('*.npz'))
        self.give_us_all = give_us_all
        self.give_us_some = give_us_some

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        if self.give_us_all:
            remaining_idx = np.arange(len(data['s_tth']))
            location = np.full_like(remaining_idx, -1)
            tmp = data['remaining_idx']
            location[tmp] = data['location']
        elif self.give_us_some:
            all_idx = np.arange(len(data['s_tth']))
            must_keep = data['remaining_idx']
            mask = np.ones(len(all_idx), dtype=bool)
            mask[must_keep] = False
            rest = all_idx[mask]
            # Randomly sample 50% of the rest.
            n_sample = len(rest) // 2
            sampled_rest = np.random.choice(rest, size=n_sample, replace=False)
            remaining_idx = np.concatenate([must_keep, sampled_rest])
            location = np.full_like(remaining_idx, -1)
            location[:len(must_keep)] = data['location']
        else:
            remaining_idx = data['remaining_idx']
            location = data['location']

        s_tth = data['s_tth'][remaining_idx]
        s_chi = data['s_chi'][remaining_idx]
        s_grain_id = data['s_grain_id'][remaining_idx]
        s_miller_ind = data['s_miller_ind'][remaining_idx,:3]
        laue_spots = np.vstack((s_tth, s_chi)).T

        return data, s_tth, s_chi, location, s_grain_id, s_miller_ind, laue_spots, idx
