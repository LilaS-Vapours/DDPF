import copy
import math
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from . import augs_TIBA as img_trsform
from .base import BaseDataset

from PIL import Image

# https://pytorch.org/docs/stable/notes/randomness.html
'''这个函数的作用是为多线程数据加载器中的每个线程设置不同的随机种子，以确保数据加载的随机性和多样性。
   希望这能帮助您更好地理解这个函数的作用和用法。
'''
def seed_worker(worker_id):                 #
    cur_seed = np.random.get_state()[1][0]
    cur_seed += worker_id
    np.random.seed(cur_seed)
    random.seed(cur_seed)


class voc_dset(BaseDataset):
    def __init__(
        self, data_root, data_list, trs_form, trs_form_strong=None, 
        seed=0, n_sup=10582, split="val", flag_semi=False,
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ):
        super(voc_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform_weak = trs_form
        self.transform_strong = trs_form_strong
        self.flag_semi = flag_semi
        self.split = split
        # random.seed(seed) # set on the top level

        self.trf_normalize = self._get_to_tensor_and_normalize(mean, std)

        # oversamplying labeled data for semi-supervised training
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample
        
        # # ADD: shuffle the image list ===> not neccesary, random.sample also shuffle it.
        # if split == "train":
        #     np.random.shuffle(self.list_sample)
    
    @staticmethod
    def _get_to_tensor_and_normalize(mean, std):
        return img_trsform.ToTensorAndNormalize(mean, std)

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.transform_strong is None:
            image, label = self.transform_weak(image, label)
            # print(image.shape, label.shape)
            image, label = self.trf_normalize(image, label)
            if not self.flag_semi:
                return index, image, label
            else:
                return index, image, image.clone(), label
        else:
            # apply augmentation
            image_weak, label = self.transform_weak(image, label)
            image_strong = self.transform_strong(image_weak)
            # print("="*100)
            # print(index, image_weak.size, image_strong.size, label.size)
            # print("="*100)

            cutmix_box = img_trsform.obtain_cutmix_box(image_strong.size[0], p=0.5)
            ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
            _, ignore_mask = img_trsform.normalize(image_strong, ignore_mask)
            mask = torch.from_numpy(np.array(mask)).long()
            ignore_mask[mask == 254] = 255

            image_weak, label = self.trf_normalize(image_weak, label)
            image_strong, _ = self.trf_normalize(image_strong, label)


            # print(index, image_weak.shape, image_strong.shape,label.shape)

            return index, image_weak, image_strong, label,ignore_mask,cutmix_box

    def __len__(self):
        return len(self.list_sample_new)


def build_additional_strong_transform(cfg):
    assert cfg.get("strong_aug", False) != False
    strong_aug_nums = cfg["strong_aug"].get("num_augs", 2)
    flag_use_rand_num = cfg["strong_aug"].get("flag_use_random_num_sampling", True)
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug


def build_basic_transfrom(cfg, split="val", mean=[0.485, 0.456, 0.406]):
    ignore_label = cfg["ignore_label"]
    trs_form = []
    if split != "val":
        if cfg.get("rand_resize", False):
            trs_form.append(img_trsform.Resize(cfg.get("resize_base_size", 600), cfg["rand_resize"]))
        
        if cfg.get("flip", False):
            trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))

    # crop also sometime for validating
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))

    return img_trsform.Compose(trs_form)


def build_vocloader(split, all_cfg, seed=0):
    # extract augs config from "train"/"val" into the higher level.
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    # set up workers and batchsize
    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    
    # build transform
    mean, std = cfg["mean"], cfg["std"]
    trs_form = build_basic_transfrom(cfg, split=split, mean=mean)
    
    # create dataset
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, None, 
        seed, n_sup, mean=mean, std=std)

    # build sampler
    sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
        worker_init_fn=seed_worker,
    )
    return loader


def build_voc_semi_loader(split, all_cfg, seed=0):
    split = "train"
    # extract augs config from "train" into the higher level.
    cfg_dset = all_cfg["dataset"]
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))
    
    # set up workers and batchsize
    workers = cfg.get("workers", 2) 
    batch_size = cfg.get("batch_size", 2)
    n_sup = 10582 - cfg.get("n_sup", 10582) # oversample labeled data to the amount of unlabeled data

    # build transform
    mean, std = cfg["mean"], cfg["std"]
    trs_form_weak = build_basic_transfrom(cfg, split=split, mean=mean)
    if cfg.get("strong_aug", False):
        trs_form_strong = build_additional_strong_transform(cfg)
    else:
        trs_form_strong = None
    
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form_weak, None, 
                    seed, n_sup, split=split, mean=mean, std=std)    
    sample_sup = DistributedSampler(dset)

    data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
    dset_unsup = voc_dset(cfg["data_root"], data_list_unsup, trs_form_weak, trs_form_strong,
                            seed, n_sup, split,
                            flag_semi=True,
                            mean=mean, std=std)
    sample_unsup = DistributedSampler(dset_unsup)

    # create dataloader
    loader_sup = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_sup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    loader_unsup = DataLoader(
        dset_unsup,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_unsup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    return loader_sup, loader_unsup
