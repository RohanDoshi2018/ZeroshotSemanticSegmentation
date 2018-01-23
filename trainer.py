import datetime
import math
import os
import os.path as osp
import shutil
import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import pickle
import utils
import utils 

class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                    train_loader, val_loader, out, max_iter,
                    size_average=False, interval_validate=None,
                    pixel_embeddings=None):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.pixel_embeddings = pixel_embeddings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.timestamp_start = datetime.datetime.now(pytz.timezone('US/Eastern'))
        self.size_average = size_average

        if self.pixel_embeddings:
            self.embeddings = utils.load_obj('embeddings/label2vec_arr_' + str(pixel_embeddings))
            if cuda:
                self.embeddings = torch.from_numpy(self.embeddings).cuda().float()
            else:
                self.embeddings = torch.from_numpy(self.embeddings).float()

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')
        self.epoch = 0
        self.validation_epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []

        for batch_idx, (data, target) in enumerate(self.val_loader):
            if self.pixel_embeddings:
                target, target_embed = target
            
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            if self.pixel_embeddings:
                if self.cuda:
                    target_embed = target_embed.cuda()
                target_embed = Variable(target_embed)

            score = self.model(data)
            
            if self.pixel_embeddings:
                loss = utils.mse_embedding(score, target, target_embed, size_average=self.size_average)
            else:
                loss = utils.cross_entropy2d(score, target, size_average=self.size_average)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)


            print("Test Epoch %d | Iteration %d | Loss %f" % (self.validation_epoch, self.iteration, val_loss))
            print(self.model.score_fr.weight.sum())

            imgs = data.data.cpu()
            if self.pixel_embeddings:	
                lbl_pred = utils.get_lbl_pred(score, self.embeddings, self.cuda)
            else:
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)

        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('US/Eastern')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth'), osp.join(self.out, 'model_best.pth'))
        
        self.validation_epoch += 1

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
               self.validate()

            assert self.model.training

            if self.pixel_embeddings:
                 target, target_embed = target
            if self.cuda:
                 data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            if self.pixel_embeddings:
                if self.cuda:
                    target_embed = target_embed.cuda()
                target_embed = Variable(target_embed)

            self.optim.zero_grad()
            score = self.model(data)

            if self.pixel_embeddings:
                loss = mse_embedding(score, target, target_embed, size_average=self.size_average)
            else:
                loss = cross_entropy2d(score, target, size_average=self.size_average)

            loss /= len(data)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')

            loss.backward()
            self.optim.step()
            print("Train Epoch %d | Iteration %d | Loss %.1f | score_fr grad sum %.0f | upscore grad sum %.0f | score sum %.0f"  % 
                            (self.epoch,
                             self.iteration,
                             float(loss.data[0]),
                             float(self.model.score_fr.weight.grad.sum().data[0]),
                             float(self.model.upscore.weight.grad.sum().data[0]),
                             float(score.sum().data[0])
                             ))

            metrics = []
            if self.pixel_embeddings: 
                lbl_pred = utils.get_lbl_pred(score, self.embeddings)
            else:
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('US/Eastern')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in range(max_epoch): 
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
