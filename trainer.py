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
from tensorboardX import SummaryWriter

class Trainer(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, out, max_epoch,
                    size_average=False,
                    pixel_embeddings=None, training_loss_func=None, background_loss=True):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.pixel_embeddings = pixel_embeddings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.timestamp_start = datetime.datetime.now(pytz.timezone('US/Eastern'))
        self.size_average = size_average
        self.training_loss_func = training_loss_func
        self.background_loss = background_loss

        if self.pixel_embeddings:

            # pascal embeddings, *all* with unit length (aka 1); use for cosine similarity for lbl inference
            self.unit_embeddings = utils.load_obj('embeddings/unit_embed_arr_' + str(pixel_embeddings))

            if cuda:
                self.unit_embeddings = Variable(torch.from_numpy(self.unit_embeddings).cuda().float(), requires_grad=False)
            else:
                self.unit_embeddings = Variable(torch.from_numpy(self.unit_embeddings).float(), requires_grad=False)

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
        self.iteration = 0
        self.max_epoch = max_epoch
        self.best_mean_iu = 0

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in enumerate(self.train_loader):

            if self.pixel_embeddings:
                 target, target_embed = target

            if self.cuda:
                 data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            if self.pixel_embeddings:
                if self.cuda:
                    target_embed = target_embed.cuda()
                target_embed = Variable(target_embed)

            self.optim.zero_grad() # TODO: Understand this.
            score = self.model(data)
            
            # # normalize score to unit length 
            # # TODO: is this normalization needed?
            # if self.pixel_embeddings:
            #     norms = torch.norm(score, p=2, dim=1) 
            #     score = score / norms

            if self.pixel_embeddings:
                if self.training_loss_func == "neg_cos":
                    loss = utils.neg_cosine_loss(score, target, target_embed, background_loss=self.background_loss, size_average=self.size_average) # TODO: fix this
                elif self.training_loss_func == "mse":
                    loss = utils.mse_loss(score, target, target_embed, background_loss=self.background_loss, size_average=True) # TODO: fix this
                else:
                    raise Exception("Unknown Loss Function")
            else:
                loss = utils.cross_entropy2d(score, target, background_loss=self.background_loss, size_average=self.size_average)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')

            loss.backward()
            self.optim.step() # TODO: understand this in context of loss.backward; updates all weights?

            print("Train Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | score_fr grad sum {:15.0f} | upscore grad sum {:15.0f} | score sum {:10.5f}".format(
                int(self.epoch),
                int(batch_idx),
                float(loss.data[0]),
                float(self.model.score_fr.weight.grad.sum().data[0]),
                float(self.model.upscore.weight.grad.sum().data[0]),
                float(score.sum().data[0])))


            # if self.pixel_embeddings:
            #     lbl_pred = utils.get_lbl_pred(score, self.unit_embeddings, self.cuda) # TODO: fix lbl_pred
            # else:
            # TODO: revert above
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()

            acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(lbl_true, lbl_pred, n_class=n_class)
            metrics = [acc, acc_cls, mean_iu, fwavacc]
            self.iteration += 1

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + metrics + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

    def validate(self):
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
            data, target = Variable(data), Variable(target)

            if self.pixel_embeddings:
                if self.cuda:
                    target_embed = target_embed.cuda()
                target_embed = Variable(target_embed)

            score = self.model(data)
            # normalize score to unit length
            # TODO: is this normalization needed?
            # if self.pixel_embeddings:
            #     norms = torch.norm(score, p=2, dim=1) 
            #     score = score / norms # TODO: is this along the correct dimension?

            if self.pixel_embeddings:
                if self.training_loss_func == "neg_cos":
                    loss = utils.neg_cosine_loss(score, target, target_embed, background_loss=self.background_loss, size_average=self.size_average) # TODO: fix this
                elif self.training_loss_func == "mse":
                    loss = utils.mse_loss(score, target, target_embed, background_loss=self.background_loss, size_average=self.size_average) # TODO: fix this
                else:
                    raise Exception("unknown loss function for joint-embedding model")
            else:
                loss = utils.cross_entropy2d(score, target, background_loss=self.background_loss, size_average=self.size_average)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')

            val_loss += float(loss.data[0])

            print("Test Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | Score Sum {:10.5f}".format(int(self.epoch), int(batch_idx), float(loss.data[0]), float(score.sum().data[0])  ))
    
            imgs = data.data.cpu()
            # if self.pixel_embeddings:	
            #     lbl_pred = utils.get_lbl_pred(score, self.unit_embeddings, self.cuda) # TODO: fix lbl_pred
            # else:
            # TODO: revert above
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)

        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class) # TODO: Understand how metrics are calculated.
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%d.jpg' % self.epoch)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 5 + [val_loss] + list(metrics) + [elapsed_time]
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
        }, osp.join(self.out, 'checkpoint')) 

        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint'), osp.join(self.out, 'best'))

        # TODO: This conditional may be superfluous. Leave for now, and test removing this later.
        if self.model.training: 
            self.model.train()

    def train(self):    
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.validate()