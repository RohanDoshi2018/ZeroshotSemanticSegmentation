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

class PascalContext():
    class_names = [
        'aeroplane',    # class #0
        'bicycle',      # class #1
        'bird',         # class #2
        'boat',         # class #3
        'bottle',       # class #4
        'bus',          # class #5
        'car',          # class #6
        'cat',          # class #7
        'chair',        # class #8
        'cow',          # class #9
        'diningtable',  # class #10
        'dog',          # class #11
        'horse',        # class #12
        'motorbike',    # class #13
        'person',       # class #14
        'pottedplant',  # class #15
        'sheep',        # class #16
        'sofa',         # class #17
        'train',        # class #18
        'tvmonitor',    # class #19
        'sky',          # class #20
        'grass',        # class #21
        'ground',       # class #22
        'road',         # class #23
        'building',     # class #24
        'tree',         # class #25
        'water',        # class #26
        'mountain',     # class #27
        'wall',         # class #28
        'floor',        # class #29
        'track',        # class #30
        'keyboard',     # class #31
        'ceiling',      # class #32
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
            raise Exception("unexpected split for context dataset")

        if self.embed_dim or self.one_hot_embed:
            self.init_embeddings()

        dataset_dir = self.data_dir + '/pascal/VOCdevkit/VOC2012'
        split_file = 'datasets/context/%s.txt' % self.split
        self.files = []

        if self.split == 'train_seen':
            split_file = 'datasets/context/train.txt'

        for did in open(split_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(self.data_dir, 'context/33_context_labels/%s.png' % did)

            lbl = PIL.Image.open(lbl_file)
            lbl = np.array(lbl, dtype=np.int32)
            lbl = lbl - 1

            # -1 is an invalid label for the context dataset
            if self.split == "train":
                if self.lbl_contains_unseen(lbl, [-1] + self.val_unseen):
                    continue
            elif self.split == "train_seen":
                if self.lbl_contains_unseen(lbl, [-1] + self.train_unseen + self.val_unseen):
                    continue
            elif self.split == "val":
                if self.lbl_contains_unseen(lbl, [-1]):
                    continue
            self.files.append({'img': img_file, 'lbl': lbl_file})

    def lbl_contains_unseen(self, lbl, unseen):
        unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
        if np.sum(unseen_pixel_mask) > 0: # ignore images with any train_unseen pixels
            return True
        return False

    def init_embeddings(self):
        if self.one_hot_embed:
            embed_arr = utils.load_obj('datasets/context/embeddings/one_hot_33_dim')
        else:
            embed_arr = utils.load_obj('datasets/context/embeddings/norm_embed_arr_' + str(self.embed_dim))

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
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl = lbl - 1 # modify to work with embedding lookup; 0-index, not 1-index

        if self.embed_dim:
            mask = (lbl==-1)
            lbl[mask] = 0 # arbitrary class 0 chosen
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

    if not osp.exists(data_dir + '/context/33_context_labels'):
        os.makedirs(data_dir+'/context', exist_ok=True)
        os.chdir(data_dir + '/context')
        URL = 'https://cs.stanford.edu/~roozbeh/pascal-context/33_context_labels.tar.gz'
        urllib.request.urlretrieve(URL, '33_context_labels.tar.gz')
        untar('33_context_labels.tar.gz') # outputs 33_context_labels in data_dir
        os.remove('33_context_labels.tar.gz')

    if not osp.exists(data_dir + '/pascal/VOCdevkit/VOC2012'):
        os.makedirs(data_dir+'/pascal', exist_ok=True)
        os.chdir(data_dir + '/pascal')
        URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        urllib.request.urlretrieve(URL, 'VOCtrainval_11-May-2012.tar')
        untar('VOCtrainval_11-May-2012.tar') # outputs VOCdevkit in data_dir
        os.remove('VOCtrainval_11-May-2012.tar')

    # reset path to current working directory
    os.chdir(os.path.dirname(__file__))

def untar(fname):
    tar = tarfile.open(fname)
    tar.extractall()
    tar.close()