import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.transforms.utility.dictionary import MapTransform
import json
from copy import deepcopy
import nibabel as nib


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def padding(min, max, divisible=32):
    range_y = int(np.ceil((max - min) / divisible) * divisible)
    if range_y < 96:
        range_y = 96
    pad_y = (range_y - (max - min))
    upper_pad = pad_y // 2
    lower_pad = pad_y - upper_pad
    return min - lower_pad, max + upper_pad


class CropForegroundd(MapTransform):
    def __init__(self, keys, spatial_info, pixdim, verify=True):
        super(CropForegroundd, self).__init__(keys)
        self.keys = keys
        self.verify = verify
        self.pixdim = pixdim
        with open(spatial_info) as f:
            self.spatial_info = json.load(f)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            f = os.path.basename(d[f"{key}_meta_dict"]["filename_or_obj"])
            # pixdim = d[f"{key}_meta_dict"]["pixdim"][1:4].tolist()
            # pixdim = [0.8, 0.8, 5.0]
            pixdim = [2.0, 2.0, 2.0]
            if "_0000" not in f:
                f = f.replace(".nii.gz", "_0000.nii.gz")
            (y_min, y_max), (x_min, x_max), (z_min, z_max) = self.spatial_info[f]
            y_min, y_max = int(y_min * pixdim[0] / self.pixdim[0]), int(y_max * pixdim[0] / self.pixdim[0])
            x_min, x_max = int(x_min * pixdim[1] / self.pixdim[1]), int(x_max * pixdim[1] / self.pixdim[1])
            z_min, z_max = int(z_min * pixdim[2] / self.pixdim[2]), int(z_max * pixdim[2] / self.pixdim[2])
            # y_min, y_max = padding(y_min, y_max)
            # x_min, x_max = padding(x_min, x_max)
            # z_min, z_max = padding(z_min, z_max)
            z_min = max(z_min, 0)
            z_max = min(z_max, d[key].shape[3])
            assert d[key].dim() == 4
            if self.verify:
                if key.startswith("label"):
                    assert d[key].numpy().sum() == d[key][:, y_min: y_max, x_min: x_max, z_min:z_max].numpy().sum()
            assert (x_max <= d[key].shape[2]) and (y_max <= d[key].shape[1]) and (z_max <= d[key].shape[3])
            # d[key] = d[key][:, y_min: y_max, x_min: x_max, z_min:z_max]
            d[key] = d[key][:, :, :, z_min:z_max]

        return d


class AddFollowupd(MapTransform):
    def __init__(self, keys):
        super(AddFollowupd, self).__init__(keys=keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if f"{key}_1" not in d:
                d[f"{key}_1"] = d[key]
        return d


class GetPairMaskd(MapTransform):
    def __init__(self, keys,
                 # dif_map_path="/dssg/home/acct-clsyzs/clsyzs-beigene/BTCV/consistency/rigid/dif_map/segresnet/fold_3/"):
                 # dif_map_path="/dssg/home/acct-clsyzs/clsyzs-beigene/BTCV/consistency/1128/nnunet/dif_map/fold_3/"):
                 # dif_map_path="/dssg/home/acct-clsyzs/clsyzs-beigene/BTCV/consistency/1203/nnunet/dif_map/fold_0/"
                 dif_map_path):
        super(GetPairMaskd, self).__init__(keys, dif_map_path)
        self.keys = keys
        # self.path = dif_map_path
        self.path = dif_map_path

    def __call__(self, data):
        d = dict(data)
        baseline_data, followup_data = deepcopy(d["image"]), deepcopy(d["image_1"])
        f = os.path.basename(d["image_1_meta_dict"]["filename_or_obj"])
        dif_map_mask = torch.tensor(nib.load(os.path.join(self.path, f)).get_fdata()).unsqueeze(0)
        dif_map = (baseline_data - followup_data) * dif_map_mask
        d["image"] = torch.cat((baseline_data, dif_map, followup_data), dim=0)
        d["image_1"] = torch.cat((followup_data, -dif_map, baseline_data), dim=0)
        return d


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    difference_map_path = args.dif_map_path
    train_transform = transforms.Compose(
        [
            AddFollowupd(keys=["image", "label"]),
            transforms.LoadImaged(keys=["image", "label", "image_1", "label_1"]),
            transforms.AddChanneld(keys=["image", "label", "image_1", "label_1"]),
            GetPairMaskd(keys=["image", "image_1"], dif_map_path=difference_map_path),
            transforms.EnsureTyped(keys=["image", "image_1"], dtype=torch.float32),
            transforms.RandFlipd(keys=["image", "label", "image_1", "label_1"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label", "image_1", "label_1"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label", "image_1", "label_1"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label", "image_1", "label_1"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys=["image", "image_1"], factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys=["image", "image_1"], offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label", "image_1", "label_1"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            AddFollowupd(keys=["image", "label"]),
            transforms.LoadImaged(keys=["image", "label", "image_1", "label_1"]),
            transforms.AddChanneld(keys=["image", "label", "image_1", "label_1"]),
            GetPairMaskd(keys=["image", "image_1"], dif_map_path=difference_map_path),
            transforms.EnsureTyped(keys=["image", "image_1"], dtype=torch.float32),
            transforms.ToTensord(keys=["image", "label", "image_1", "label_1"]),
        ]
    )
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    if args.use_normal_dataset:
        train_ds = data.Dataset(data=datalist, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    # val_transform(val_files[62])
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )
    loader = [train_loader, val_loader]

    return loader
