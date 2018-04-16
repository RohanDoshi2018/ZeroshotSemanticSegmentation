#!/usr/bin/env python
import argparse
import os
import os.path as osp
import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import tqdm
import pascal_dataset, models, utils 
from tensorboardX import SummaryWriter
import context_dataset

import PIL.Image

def visualize_segmentation(**kwargs):
    """Visualize segmentation.
    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.
    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    unseen = kwargs.pop('unseen', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if unseen:
        n_col = 4
    else:
        n_col = 3
        
    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred is not None:
            lbl_pred[mask_unlabeled] = 0

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            fcn.utils.label2rgb(lbl_true, label_names=label_names, n_labels=n_class),
            fcn.utils.label2rgb(lbl_true, img, label_names=label_names, n_labels=n_class),
        ]
        if unseen:
            viz_trues.append(make_seen_mask(img, lbl_true, unseen, n_class))
        
        for i in range(1, n_col):
            viz_trues[i][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(fcn.utils.get_tile_image(viz_trues, (1, n_col)))

    if lbl_pred is not None:
        viz_preds = [
            img,
            fcn.utils.label2rgb(lbl_pred, label_names=label_names, n_labels=n_class),
            fcn.utils.label2rgb(lbl_pred, img, label_names=label_names, n_labels=n_class),
        ]
        if unseen:
            viz_preds.append(make_seen_mask(img, lbl_pred, unseen, n_class))
        
        if mask_unlabeled is not None and viz_unlabeled is not None:
            for i in range(1, n_col):
                viz_preds[i][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(fcn.utils.get_tile_image(viz_preds, (1, n_col)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return fcn.utils.get_tile_image(vizs, (2, 1))
    else:
        raise RuntimeError

def make_seen_mask(img, lbl, unseen, n_class):
    seen = [x for x in range(n_class) if x not in unseen]
    mask_seen = np.in1d(lbl.ravel(), seen).reshape(lbl.shape)
    mask_seen = (mask_seen * 255).astype(np.uint8)
    mask_seen = np.repeat(mask_seen[:, :, np.newaxis], 3, axis=2)
    return mask_seen







# ARGUMENTS
data_dir = '/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation'
model_name = '/20D_pascal_CFG_5_MAX_EPOCH_50_LR_5e-05_MOMENTUM_None_WEIGHT_DECAY_None_EMBED_DIM_20_ONE_HOT_EMBED_False_LOSS_FUNC_cos_BK_LOSS_True_UNSEEN_False_DATASET_pascal_OPTIMIZER_adam_TIME-20180408-234743'
embed_dim = 20
# val_dataset = pascal_dataset.VOC2011ClassSeg(split='seg11valid', transform=True, embed_dim=20, one_hot_embed=False, data_dir=data_dir)
val_dataset = context_dataset.VOCContext(split='val', transform=True, embed_dim=20, one_hot_embed=False, data_dir=data_dir)
embed_arr = utils.load_obj('datasets/pascal/embeddings/one_hot_21_dim') # inference candidate embeddings

os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
cuda = torch.cuda.is_available()
    
# 1. val DATASET
kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}

val_dataset = context_dataset.VOCContext(split='val', transform=True, embed_dim=20, one_hot_embed=False, data_dir=data_dir)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

# 3. INFERENCE on val dataset
embeddings = Variable(torch.from_numpy(embed_arr).cuda().float(), requires_grad=False)
n_class = len(val_loader.dataset.class_names)

lbl_preds, lbl_trues = [], []
for batch_idx, (data, target) in enumerate(val_loader):    
    target, target_embed = target
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    target_embed = Variable(target_embed.cuda())
    
    img = data.data.cpu()[0]
    lt = target.data.cpu()[0]
    lp = np.copy(lt)
    
    img, lt = val_loader.dataset.untransform(img, lt)

    im_visualize = PIL.Image.fromarray(np.uint8(img))

    viz = visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,unseen=[6,7,8,9,10])
        
    break






PIL.Image.fromarray(np.uint8(viz))