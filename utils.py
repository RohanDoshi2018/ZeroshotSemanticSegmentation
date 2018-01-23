import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin-1')

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def mse_embedding(score, target, target_embed, size_average=True):
    """Mean Square Vector between two (n,c,h,w) volumes (score and target).

    Args:
      score: torch.Size([1, 50, 366, 500])
      target: torch.Size([1, 366, 500])
      target_embed: torch.Size([1, 366, 500, 50])
    Return: 
      loss -> scalar
    """

    n, c, h, w = score.size()

    # target: torch.Size([1, 366, 500, 50]) -> should be torch.Size([1, 50, 366, 500])
    target_embed = target_embed.permute(0,3,1,2)

    # apply mask to score and target, and turn into 1d vectors for comparision
    mask = target >= 0
    mask_tensor = mask.view(1,1,h,w).repeat(1,c,1,1)
    score_masked = score[mask_tensor]
    target_embed_masked = target_embed[mask_tensor]
    
    # calculate loss on masked score and target
    loss = F.mse_loss(score_masked, target_embed_masked, size_average=False)
    
    if size_average:
        loss /= mask.data.sum()

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
# score: torch.Size([1, 50, 366, 500])
def get_lbl_pred(score, embed_arr, cuda=False):
  n, c, h, w = score.size()
  n_classes = embed_arr.shape[0]
  embeddings = embed_arr.transpose(1,0).repeat(1,h*w,1,1)
  score = score.view(1,h*w,c,1).repeat(1,1,1,n_classes)
  dist = score.data - embeddings
  dist = dist.pow(2).sum(2).sqrt()
  min_val, indices = dist.min(2)
  if cuda:
    return indices.view(1,h,w).cpu().numpy()
  else:
    return indices.view(1,h,w).numpy()  


