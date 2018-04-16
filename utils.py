import numpy as np
import pickle
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from distutils.version import LooseVersion
import os
import fcn

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin-1')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def cross_entropy2d(score, target, weight=None, size_average=False):
    """
    ARGS
      score: (n, c, h, w)
      target: (n, h, w)
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

    mask = target >= 0 # ignore -1 (unknown classes); don't ignore 0 (background)
    mask_tensor = mask.view(n, h, w, 1).repeat(1, 1, 1, c)
    log_p = log_p[mask_tensor]
    log_p = log_p.view(-1, c) # log_p: (n*h*w, c)
    target = target[mask] # target: (n*h*w,)

    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
      loss /= mask.data.sum()
    return loss

def mse_loss(score, target, target_embed, size_average=False):
    """Mean Square Vector between two (n,c,h,w) volumes (score and target).
    ARGS
      score: (n, c, h, w)
      target: (n, h, w)
      target_embed: (n, c, h, w)
    RET
      loss -> scalar
    """
    n, c, h, w = score.size()

    # TODO: correctly normalize score to same range as target for each pixel 
    # clipping?

    # apply mask to score and target, and turn into 1d vectors for comparision
    mask = target >= 0 # ignore -1 (unknown classes); don't ignore 0 (background)
    mask_tensor = mask.view(n,1,h,w).repeat(1,c,1,1)
    score_masked = score[mask_tensor]
    target_embed_masked = target_embed[mask_tensor]

    # # calculate loss on masked score and target
    # same as: loss = (torch.sum((score_masked - target_embed_masked)**2))
    loss = F.mse_loss(score_masked, target_embed_masked, size_average=False)

    if size_average:
      loss /= mask.data.sum()

    return loss

def cosine_loss(score, target, target_embed, size_average=False):
    """Negative Cosine Similarity Loss between two (n,c,h,w) volumes (score and target).
    ARGS
      score: (n, c, h, w)
      target: (n, h, w)
      target_embed: (n, c, h, w)
    RET 
      loss -> scalar
    """
    n, c, h, w = score.size()

    # normalize score and target
    score_norm = torch.norm(score, p=2, dim=1) 
    score = score / score_norm

    target_embed_norm = torch.norm(target_embed, p=2, dim=1)  
    target_embed = target_embed/ target_embed_norm

    # apply mask to score and target
    mask = target >= 0 # ignore -1 (unknown classes); don't ignore 0 (background)
    mask_size = mask.data.sum()
    mask_tensor = mask.view(n,1,h,w).repeat(1,c,1,1)
    score_masked = score[mask_tensor]
    target_embed_masked = target_embed[mask_tensor]

    loss = mask_size - torch.sum(score_masked * target_embed_masked)
    if size_average:
      loss /= mask_size # divide loss by number of non-masked pixels
    return loss

def _fast_hist(label_true, label_pred, n_class, target='all', unseen=None):
    mask = (label_true >= 0) & (label_true < n_class)

    if target == 'unseen':
            mask_unseen = np.in1d(label_true.ravel(), unseen).reshape(label_true.shape)
            mask = mask & mask_unseen 
            
    elif target == 'seen':
        seen = [x for x in range(n_class) if x not in unseen]
        mask_seen = np.in1d(label_true.ravel(), seen).reshape(label_true.shape)
        mask = mask & mask_seen
            
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def _hist_to_metrics(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def label_accuracy_score(label_trues, label_preds, n_class, unseen=None):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    
    if unseen:
        unseen_hist, seen_hist = np.zeros((n_class, n_class)), np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class, target='all')
        if unseen:
            seen_hist += _fast_hist(lt.flatten(), lp.flatten(), n_class, target='seen', unseen=unseen)
            unseen_hist += _fast_hist(lt.flatten(), lp.flatten(), n_class, target='unseen', unseen=unseen)
    
    metrics = _hist_to_metrics(hist)
    if unseen:
        seen_metrics, unseen_metrics = _hist_to_metrics(seen_hist), _hist_to_metrics(unseen_hist)      
        metrics = metrics, seen_metrics, unseen_metrics

    return metrics

# infer lbl for a joint-embedding using nearest neighboring embedding (NNE) inference
# score: Variable (n,c,h,w)
# embed_arr: Variable (c, embed_dim) e.g. (21,20)
def infer_lbl(score, embed_arr, cuda=False, flag=False):

  # if flag: 
  #   import pdb; pdb.set_trace()

  score = score.data
  embed_arr = embed_arr.data

  n, c, h, w = score.size()
  num_class, num_dim = embed_arr.shape

  score = score.transpose(1,2).transpose(2,3).contiguous().view(h*w,c)
  embeddings = embed_arr.transpose(1,0)

  similarity_scores = torch.mm(score, embeddings)

  # normalize by norm of score and embeddings
  score_norm = torch.norm(score, p=2, dim=1).view(h*w,1).repeat(1,num_class)
  embeddings_norm = torch.norm(embeddings, p=2, dim=0).view(1,num_class).repeat(h*w,1)
  embeddings_norm[embeddings_norm == 0] = 1 # prevents division by zero
  norm = score_norm * embeddings_norm

  similarity_scores = similarity_scores / norm

  max_val, indices = similarity_scores.max(1) # max along correct dimension

  if cuda:
    return indices.view(1,h,w).cpu().numpy()
  else:
    return indices.view(1,h,w).numpy()

# for seen pixels, inference along all classes. for unseen pixels, inference only among unseen classes
def infer_lbl_forced_unseen(score, target, all_embed_arr, unseen_embed_arr, unseen, cuda=False):
  target =  target.data.cpu().numpy()

  unseen_mask = np.in1d(target.ravel(), unseen).reshape(target.shape)

  all_infer_lbl =  infer_lbl(score, all_embed_arr, cuda=cuda) # inferences among all classes
  unseen_infer_lbl = infer_lbl(score, unseen_embed_arr, cuda=cuda, flag=True) # inferences among just the unseen classes

  forced_unseen_lbl = all_infer_lbl
  forced_unseen_lbl[unseen_mask] = unseen_infer_lbl[unseen_mask]
  return forced_unseen_lbl