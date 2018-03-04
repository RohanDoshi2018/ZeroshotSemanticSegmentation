import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from distutils.version import LooseVersion

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin-1')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def cross_entropy2d(score, target, weight=None, background_loss=True, size_average=False):
    """
    ARGS
      score: (n, c, h, w)
      target: (n, h, w)
      background_loss: boolean, include background classes in loss?
    """

    n, c, h, w = score.size()
    
    # for each pixel, normalize 21D vector via log-softmax.
    # interpret each vector element as a probability of a given class.
    # each element is between 0 and 1, and all element probabilities sum to 1.
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
      # ==0.2.X
      log_p = F.log_softmax(score) # log_p: (n, c, h, w)
    else:
      # >=0.3
      log_p = F.log_softmax(score, dim=1) # log_p: (n, c, h, w)
    
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous() # log_p: (n, h, w, c)

    if background_loss:
      mask = target >= 0 # ignore -1 (unknown classes); don't ignore 0 (background)
    else:
      mask = target > 0 # ignore -1 (unknown classes) and 0 (background)

    mask_tensor = mask.view(n, h, w, 1).repeat(1, 1, 1, c)
    log_p = log_p[mask_tensor]
    log_p = log_p.view(-1, c) # log_p: (n*h*w, c)
    target = target[mask] # target: (n*h*w,)

    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
      loss /= mask.data.sum()
    return loss

def mse_loss(score, target, target_embed, background_loss=False, size_average=False):
    """Mean Square Vector between two (n,c,h,w) volumes (score and target).
    ARGS
      score: (n, c, h, w)
      target: (n, h, w)
      target_embed: (n, c, h, w)
      background_loss: boolean, compute loss on background pixels?
    RET
      loss -> scalar
    """
    n, c, h, w = score.size()

    # TODO: correctly normalize score to same range as target for each pixel 
    # clipping?

    # apply mask to score and target, and turn into 1d vectors for comparision
    if background_loss:
      mask = target >= 0 # ignore -1 (unknown classes); don't ignore 0 (background)
    else:
      mask = target > 0 # ignore -1 (unknown classes) and 0 (background)
    mask_tensor = mask.view(n,1,h,w).repeat(1,c,1,1)
    score_masked = score[mask_tensor]
    target_embed_masked = target_embed[mask_tensor]

    # # calculate loss on masked score and target
    # same as: loss = (torch.sum((score_masked - target_embed_masked)**2))
    loss = F.mse_loss(score_masked, target_embed_masked, size_average=False)

    if size_average:
      loss /= mask.data.sum()

    return loss

def neg_cosine_loss(score, target, target_embed, background_loss=False, size_average=False):
    """Negative Cosine Similarity Loss between two (n,c,h,w) volumes (score and target).
    ARGS
      score: (n, c, h, w)
      target: (n, h, w)
      target_embed: (n, c, h, w)
      background_loss: boolean, compute loss on background pixels?
    RET 
      loss -> scalar
    """

    n, c, h, w = score.size()

    # TODO: normalize score to same range as target for each pixel 

    # apply mask to score and target
    if background_loss:
      mask = target >= 0 # ignore -1 (unknown classes); don't ignore 0 (background)
    else:
      mask = target > 0 # ignore -1 (unknown classes) and 0 (background)
    mask_size = mask.data.sum()
    mask_tensor = mask.view(n,1,h,w).repeat(1,c,1,1)
    score_masked = score[mask_tensor]
    target_embed_masked = target_embed[mask_tensor]

    loss = -1 * torch.sum(score_masked * target_embed_masked)
    if size_average:
      loss /= mask_size # divide loss by number of non-masked pixels
    return loss

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

# get nearest label prediction for pixel embeddings of size (n,c, h, w) 
# score: torch (n,c,h,w)
# embed_arr: torch (c, embed_dim) e.g. (21,50)
def get_lbl_pred(score, embed_arr, cuda=False):
  n, c, h, w = score.size()
  num_class, num_dim = embed_arr.shape

  score = score.transpose(1,2).transpose(2,3).contiguous().view(h*w,c)
  embeddings = embed_arr.transpose(1,0)
  similarity_scores = torch.mm(score, embeddings)
  max_val, indices = similarity_scores.max(1) # min along correct dimension?

  if cuda:
    return indices.view(1,h,w).data.cpu().numpy() # TODO: why dim 1
  else:
    return indices.view(1,h,w).data.numpy() # TODO: why dim 1

# useful for visualizing heatmaps, masks etc.
def tensor_to_img(tensor):
  return True
