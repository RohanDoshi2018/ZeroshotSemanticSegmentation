#!/usr/bin/env python

import collections
import os.path as osp
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import utils
import pickle

data_dir = open('data_dir.txt', 'r').read().strip()

class VOCClassSegBase(data.Dataset):
    class_names = np.array([
        'background',    # class 0
        'aeroplane',     # class 1
        'bicycle',       # class 2
        'bird',          # class 3
        'boat',          # class 4
        'bottle',        # class 5
        'bus',           # class 6
        'car',           # class 7
        'cat',           # class 8
        'chair',         # class 9
        'cow',           # class 10
        'diningtable',   # class 11
        'dog',           # class 12
        'horse',         # class 13
        'motorbike',     # class 14
        'person',        # class 15
        'potted plant',  # class 16
        'sheep',         # class 17
        'sofa',          # class 18
        'train',         # class 19
        'tv/monitor',    # class 20
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, split='train', transform=False, embed_dim=None, one_hot_embed=False):
        self.split = split
        self._transform = transform
        self.embed_dim = embed_dim # of dimensions for the embed_dim-embeddings
        self.one_hot_embed = one_hot_embed

        if self.embed_dim or self.one_hot_embed:
            self.init_embeddings()

        # VOC2011 and others are subset of VOC2012
        dataset_dir = data_dir + '/pascal/VOCdevkit/VOC2012'
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({'img': img_file, 'lbl': lbl_file,})

    def init_embeddings(self):
        if self.one_hot_embed:
            embed_arr = utils.load_obj('embeddings/one_hot_21_dim')
        else:
            embed_arr = utils.load_obj('embeddings/unit_embed_arr_' + str(self.embed_dim))

        num_classes = embed_arr.shape[0] #  21 = background (class 0) + labels (class 1-20)
        self.embeddings = torch.nn.Embedding(num_classes, self.embed_dim)
        self.embeddings.weight.requires_grad = False
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_arr))

    def __len__(self):         
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        
        if self.embed_dim:
            # change lbl's -1 to 0 for embedding lookup because it cannot handle index -1. revert after
            mask = (lbl==-1)
            lbl[mask] = 0
            lbl_vec = self.embeddings(torch.from_numpy(lbl).long()).data
            lbl_vec = lbl_vec.permute(2,0,1)
            lbl[mask] = -1

        if self._transform:
            img, lbl = self.transform(img, lbl)
                
        if self.embed_dim:
            return img, (lbl, lbl_vec)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1) # TODO: does this work with self.pixel_embeddings
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, split='train', transform=False, embed_dim=None, one_hot_embed=False):
        super(VOC2011ClassSeg, self).__init__(
            split=split, transform=transform, embed_dim=embed_dim, one_hot_embed=one_hot_embed)
        imgsets_file = data_dir + '/pascal/seg11valid.txt'
        dataset_dir = data_dir + '/pascal/VOCdevkit/VOC2012'
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, split='train', transform=False, embed_dim=None, one_hot_embed=False):
        super(VOC2012ClassSeg, self).__init__(root, split=split, transform=transform, embed_dim=embed_dim, one_hot_embed=one_hot_embed)


class SBDClassSeg(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, split='train', transform=False, embed_dim=None, one_hot_embed=False):
        self.split = split
        self._transform = transform
        self.embed_dim = embed_dim # of dimensions for the embed_dim-embeddings
        self.one_hot_embed = one_hot_embed

        if self.embed_dim or self.one_hot_embed:
            self.init_embeddings()

        dataset_dir = data_dir + '/pascal/benchmark_RELEASE/dataset'
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({'img': img_file, 'lbl': lbl_file,})

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)   
        lbl[lbl == 255] = -1

        if self.embed_dim:
            # change lbl's -1 to 0 for embedding lookup because it cannot handle index -1. revert after
            mask = (lbl==-1)
            lbl[mask] = 0
            lbl_vec = self.embeddings(torch.from_numpy(lbl).long()).data
            lbl_vec = lbl_vec.permute(2,0,1)
            lbl[mask] = -1
        
        if self._transform:
            img, lbl = self.transform(img, lbl)
        
        if self.embed_dim:
            return img, (lbl, lbl_vec)
        else:
            return img, lbl
