import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, root_dir_fa, labels_file, return_filenames=False):
        self.root_dir_fa = root_dir_fa
        self.return_filenames = return_filenames
        self.labels = []

        # 读取标签文件，每行包含文件名和标签
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    filename, label = parts
                    self.labels.append((filename, int(label)))
                else:
                    raise ValueError("Each line in labels file must have two elements separated by a comma.")

    def __getitem__(self, idx):
        # 获取标签和文件名
        img_name, label = self.labels[idx]

        fa_img_path = os.path.join(self.root_dir_fa, img_name)
        # 加载每个模态的图像
        fa_img = nib.load(fa_img_path).get_fdata()

        # fa_img = torch.from_numpy(fa_img).float().permute(2, 0, 1).unsqueeze(0)  # (H,W,D) → (D,H,W) → (1,D,H,W)      91 109 71
        fa_img = torch.from_numpy(fa_img).float().unsqueeze(0)  # (H,W,D) → (D,H,W) → (1,D,H,W)      91 109 71

        if self.return_filenames:
            # return fa_img, label, img_name
            return fa_img, label, img_name, idx
        return fa_img, label
        # return fa_img, label, idx

    def __len__(self):
        return len(self.labels)


class AddedDataset(Dataset):
    def __init__(self, dataset, indices, pseudolb):
        self.indices = indices
        self.dataset = dataset
        self.pseudolb = pseudolb

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_raw, label, _, _ = self.dataset[self.indices[index]]
        target = self.pseudolb[self.indices[index]]
        d_label = torch.zeros(1)
        # # Convert label to tensor if it's an integer
        # if isinstance(target, int):
        #     label = torch.tensor(target, dtype=torch.long)  # Use torch.int64 for class indices
        #
        # print(target, label)

        return d_label, (img_raw, target)

    # unlabeled_mask = torch.ones(size=(len(unlabeled_indices),), dtype=torch.bool)
    # samples_indices = samples_indices[samples_indices < len(unlabeled_indices)]
    # unlabeled_mask[samples_indices] = 0
    # labeled_indices = np.hstack([labeled_indices, unlabeled_indices[~unlabeled_mask]])
    # unlabeled_indices = unlabeled_indices[unlabeled_mask]

class SubDataset(Dataset):
    def __init__(self, dataset, indices, return_filenames=False):
        self.indices = indices
        self.dataset = dataset
        self.return_filenames = return_filenames

        trg_idx = np.array(range(len(dataset)))

        unlabeled_mask = torch.ones(size=(len(dataset),), dtype=torch.bool)
        unlabeled_mask[self.indices] = 0
        self.trg_indices = trg_idx[unlabeled_mask]

    def __len__(self):
        return len(self.trg_indices)

    def __getitem__(self, index):
        # print(len(self.trg_indices), len(self.dataset))
        img_raw, target, img_name, _ = self.dataset[self.trg_indices[index]]

        if self.return_filenames:
            return img_raw, target, img_name, index
        return img_raw, target