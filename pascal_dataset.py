import PIL.Image
import numpy as np
import os
import os.path as osp
import pickle
import scipy.io
import shutil
import tarfile
import torch
import urllib.request
import utils

from torch.utils import data

class PascalVOC(data.Dataset):
    class_names = [
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
    ]
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, split='train', transform=False, embed_dim=None, one_hot_embed=False, data_dir='data', train_unseen=[], val_unseen=[]):
        self.split = split
        self._transform = transform
        self.embed_dim = embed_dim # of dimensions for the embed_dim-embeddings
        self.one_hot_embed = one_hot_embed
        self.data_dir = data_dir
        self.train_unseen = train_unseen
        self.val_unseen = val_unseen

        if self.split not in ['train', 'train_seen', 'val']:
            raise Exception("unexpected split for pascal dataset")

        if self.embed_dim or self.one_hot_embed:
            self.init_embeddings()

        split_file = 'datasets/pascal/%s.txt' % self.split
        self.files = []

        if self.split == 'train_seen':
            split_file = 'datasets/pascal/train.txt'

        for did in open(split_file):
            did = did.strip()

            if self.split in ['train', 'train_seen']:
                dataset_dir = self.data_dir + '/pascal/benchmark_RELEASE/dataset'
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                
                mat = scipy.io.loadmat(lbl_file)
                lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32) 

            elif self.split == 'val':
                dataset_dir = self.data_dir + '/pascal/VOCdevkit/VOC2012'
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)

            if self.split == "train":
                if self.lbl_contains_unseen(lbl, self.val_unseen):
                    continue
            elif self.split == "train_seen":
                if self.lbl_contains_unseen(lbl, self.train_unseen + self.val_unseen):
                    continue
            self.files.append({'img': img_file, 'lbl': lbl_file})

    def lbl_contains_unseen(self, lbl, unseen):
        unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
        if np.sum(unseen_pixel_mask) > 0: # ignore images with any train_unseen pixels
            return True
        return False

    def init_embeddings(self):
        if self.one_hot_embed:
            embed_arr = utils.load_obj('datasets/pascal/embeddings/one_hot_21_dim')
        else:
            embed_arr = utils.load_obj('datasets/pascal/embeddings/norm_embed_arr_' + str(self.embed_dim))

        num_classes = embed_arr.shape[0]
        self.embeddings = torch.nn.Embedding(num_classes, self.embed_dim)
        self.embeddings.weight.requires_grad = False
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_arr))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        if self.split in ['train', 'train_seen']:
            mat = scipy.io.loadmat(lbl_file)
            lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)  
        elif self.split == 'val': 
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
        img = img.transpose(2, 0, 1)
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

def download(data_dir):
    os.chdir(os.path.dirname(__file__)) # confirm in currenet working directory

    if not osp.exists(data_dir+'/pascal'):
        os.makedirs(data_dir+'/pascal', exist_ok=True)
    
    if not osp.exists(data_dir + '/pascal/benchmark_RELEASE'):
        os.chdir(data_dir + '/pascal')
        URL = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
        urllib.request.urlretrieve(URL, 'benchmark.tar')
        untar('benchmark.tar') # outputs benchmark_RELEASE in data_dir
        os.remove('benchmark.tar')

    if not osp.exists(data_dir + '/pascal/VOCdevkit/VOC2012'):
        os.chdir(data_dir + '/pascal')
        URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        urllib.request.urlretrieve(URL, 'VOCtrainval_11-May-2012.tar')
        untar('VOCtrainval_11-May-2012.tar') # outputs VOCdevkit in data_dir
        os.remove('VOCtrainval_11-May-2012.tar')

    # reset current working directory
    os.chdir(os.path.dirname(__file__))

def untar(fname):
    tar = tarfile.open(fname)
    tar.extractall()
    tar.close()