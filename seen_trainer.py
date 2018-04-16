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
import vis_utils

# trainer for the binary seen/unseen classifier
class Trainer(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, out, dataset, max_epoch, tb_writer, size_average=False, 
                    pixel_embeddings=None, loss_func=None, unseen=None, label_names=None, forced_unseen=False):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.pixel_embeddings = pixel_embeddings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = out
        self.timestamp_start = datetime.datetime.now(pytz.timezone('US/Eastern'))
        self.size_average = size_average
        self.loss_func = loss_func
        self.tb_writer = tb_writer
        self.unseen = unseen
        self.label_names = label_names
        self.dataset = dataset
        self.forced_unseen = forced_unseen

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.best_mean_iu = 0
        self.n_class = len(self.train_loader.dataset.class_names)

        self.train_unseen_img = 0
        self.train_ignored_img = 0

        self.val_ignored_img = 0

        # set up embeddings
        if self.pixel_embeddings:
            # each embedding has norm between 0 and 1
            embed_arr = utils.load_obj('datasets/%s/embeddings/norm_embed_arr_%s' % (dataset, str(pixel_embeddings)))
            if cuda:
                self.embeddings = Variable(torch.from_numpy(embed_arr).cuda().float(), requires_grad=False)
            else:
                self.embeddings = Variable(torch.from_numpy(embed_arr).float(), requires_grad=False)

        if self.forced_unseen:
            unseen_embed_arr = np.zeros(embed_arr.shape)  # default embeddings should 
            unseen_embed_arr[self.unseen, :] = embed_arr[self.unseen, :]
            if cuda:
                self.unseen_embeddings = Variable(torch.from_numpy(unseen_embed_arr).cuda().float(), requires_grad=False)
            else:               
                self.unseen_embeddings = Variable(torch.from_numpy(unseen_embed_arr).float(), requires_grad=False)

        # set up tensorboard and other logging
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.train_log_headers = ['epoch', 'iteration', 'train/loss', 'train/pxl_acc', 'train/class_acc', 'train/mean_iu', 'train/fwavacc', 'elapsed_time']

        if self.unseen:
            self.val_log_headers = ['epoch', 'iteration', 'val/loss', 'val/pxl_acc', 'val/class_acc', 'val/mean_iu', 'val/fwavacc', 
                                    'val/seen/pxl_acc', 'val/seen/class_acc', 'val/seen/mean_iu', 'val/seen/fwavacc',         
                                    'val/unseen/pxl_acc', 'val/unseen/class_acc', 'val/unseen/mean_iu', 'val/unseen/fwavacc','elapsed_time']
        else:
            self.val_log_headers = ['epoch', 'iteration', 'val/loss', 'val/pxl_acc', 'val/class_acc', 'val/mean_iu', 'val/fwavacc', 'elapsed_time']

        if not osp.exists(osp.join(self.out, 'train_log.csv')):
            with open(osp.join(self.out, 'train_log.csv'), 'w') as f:
                f.write(','.join(self.train_log_headers) + '\n')

        if not osp.exists(osp.join(self.out, 'val_log.csv')):
            with open(osp.join(self.out, 'val_log.csv'), 'w') as f:
                f.write(','.join(self.val_log_headers) + '\n')

    def forward(self, data, target):
        #  get score
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

        # get lossf
        if self.pixel_embeddings:
            if self.loss_func == "cos":
                loss = utils.cosine_loss(score, target, target_embed, size_average=True)
            elif self.loss_func == "mse":
                loss = utils.mse_loss(score, target, target_embed, size_average=True)
            else:
                raise Exception("Unknown Loss Function")
        else:
            loss = utils.cross_entropy2d(score, target, size_average=self.size_average)

        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')

        # evaluate inference 
        if self.pixel_embeddings:
            if self.forced_unseen:
                lbl_pred = utils.infer_lbl_forced_unseen(score, target, self.embeddings, self.unseen_embeddings, self.unseen, self.cuda)
            else:
                lbl_pred = utils.infer_lbl(score, self.embeddings, self.cuda)
        else:
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()

        return score, loss, lbl_pred, lbl_true

    def is_target_good(self, target, mode):
        if self.pixel_embeddings:
            target = target[0]
        target = target.numpy()

        # context dataset: check if image contains invalid class (-1)
        # note: we don't care if pascal's target has -1 because we treat 
        #   as backgorund (0) for embeddings and ignore for val metrics.
        #   pascal's -1 class signifies boundaries. context has no boundary
        #   pixels, so -1 signfies an error
        if self.dataset == 'context':
            mask = target<0
            if np.sum(mask) > 0:
                if mode == 'train' and self.epoch == 0:
                    self.train_ignored_img += 1
                elif mode == 'val' and self.epoch == 0:
                    self.val_ignored_img += 1
                return False

        # training: check if image contains unseen class
        if mode == 'train':
            mask = np.in1d(target.ravel(), self.unseen).reshape(target.shape)
            if np.sum(mask) > 0:
                return False

        return True

    def train_epoch(self):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):

            if not self.is_target_good(target, 'train'):
                continue

            score, loss, lbl_pred, lbl_true = self.forward(data, target)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            print("Train Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | score_fr grad sum {:15.0f} | upscore grad sum {:15.0f} | score sum {:10.5f}".format(
                int(self.epoch), int(batch_idx), float(loss.data[0]), float(self.model.score_fr.weight.grad.sum().data[0]),
                float(self.model.upscore.weight.grad.sum().data[0]), float(score.sum().data[0])))

            # update the training logs
            metrics = utils.label_accuracy_score(lbl_true.numpy(), lbl_pred, self.n_class)
            with open(osp.join(self.out, 'train_log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + list(metrics) + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            # write to tensorboard
            self.tb_writer.add_scalar('train/loss', loss.data[0], self.iteration)
            self.tb_writer.add_scalar('train/pxl_acc', metrics[0], self.iteration)
            self.tb_writer.add_scalar('train/class_acc', metrics[1], self.iteration)
            self.tb_writer.add_scalar('train/mean_iu', metrics[2], self.iteration)
            self.tb_writer.add_scalar('train/fwavacc', metrics[3], self.iteration)

            self.iteration += 1

        if self.epoch == 0:
            self.train_seen = len(self.train_loader) - self.train_ignored_img - self.train_unseen_img
            self.tb_writer.add_text('train/seen_img',  str(self.train_seen))
            self.tb_writer.add_text('train/unseen_img', str(self.train_unseen_img))
            self.tb_writer.add_text('train/ignored_img', str(self.train_ignored_img))

    def validate(self):
        self.model.eval()

        val_loss = 0
        lbl_trues, lbl_preds, visualizations = [], [], []
        self.val_ignored_img = 0

        for batch_idx, (data, target) in enumerate(self.val_loader):

            if not self.is_target_good(target, 'val'):
                self.val_ignored_img += 1
                continue

            score, loss, lbl_pred, lbl_true = self.forward(data, target)

            val_loss += float(loss.data[0])
            print("Test Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | Score Sum {:10.5f}".format(int(self.epoch), int(batch_idx), float(loss.data[0]), float(score.sum().data[0])))

            img, lt, lp = data[0], lbl_true[0], lbl_pred[0] # eliminate first dimension (n=1) for visualization
            img, lt = self.val_loader.dataset.untransform(img, lt)
            lbl_trues.append(lt)
            lbl_preds.append(lp)

            # generate visualization for first 9 images of val_loader
            if len(visualizations) < 9:
                viz = vis_utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class, label_names=self.label_names, unseen=self.unseen)
                visualizations.append(viz)

        # write validation set summary to tensorboard
        self.val_total_img = len(self.val_loader) - self.val_ignored_img
        if self.epoch == 0:
            self.tb_writer.add_text('val/total_img', str(self.val_total_img))
            self.tb_writer.add_text('val/ignored_images', str(self.val_ignored_img))

        # save the visualizaton image
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%d.jpg' % self.epoch)
        viz_img = fcn.utils.get_tile_image(visualizations)
        scipy.misc.imsave(out_file, viz_img)

        # update the validation log for the current epoch
        if self.unseen:
            metrics = utils.label_accuracy_score(lbl_trues, lbl_preds, self.n_class, unseen=self.unseen)
            metrics, seen_metrics, unseen_metrics = metrics

            self.tb_writer.add_scalar('val/seen/pxl_acc', seen_metrics[0], self.epoch)
            self.tb_writer.add_scalar('val/seen/class_acc', seen_metrics[1], self.epoch)
            self.tb_writer.add_scalar('val/seen/mean_iu', seen_metrics[2], self.epoch)
            self.tb_writer.add_scalar('val/seen/fwavacc', seen_metrics[3], self.epoch)

            self.tb_writer.add_scalar('val/unseen/pxl_acc', unseen_metrics[0], self.epoch)
            self.tb_writer.add_scalar('val/unseen/class_acc', unseen_metrics[1], self.epoch)
            self.tb_writer.add_scalar('val/unseen/mean_iu', unseen_metrics[2], self.epoch)
            self.tb_writer.add_scalar('val/unseen/fwavacc', unseen_metrics[3], self.epoch)

        else:
            metrics = utils.label_accuracy_score(lbl_trues, lbl_preds, self.n_class)
        
        val_loss /= self.val_total_img # val loss is averaged across all the images

        with open(osp.join(self.out, 'val_log.csv'), 'a') as f:
            elapsed_time = datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start
            if self.unseen:
                log = [self.epoch, self.iteration] + [val_loss] + list(metrics) + list(seen_metrics) + list(unseen_metrics) + [elapsed_time]
            else:
                log = [self.epoch, self.iteration] + [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # write metrics to tensorboard
        self.tb_writer.add_scalar('val/loss', val_loss, self.epoch)
        self.tb_writer.add_scalar('val/pxl_acc', metrics[0], self.epoch)
        self.tb_writer.add_scalar('val/class_acc', metrics[1], self.epoch)
        self.tb_writer.add_scalar('val/mean_iu', metrics[2], self.epoch)
        self.tb_writer.add_scalar('val/fwavacc', metrics[3], self.epoch)
        self.tb_writer.add_image('segmentations', viz_img, self.epoch)

        # track and update the best mean intersection over union
        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu

        # checkpoint the model's weights
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint')) 

        # save the weights for the best performing model so far
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint'), osp.join(self.out, 'best'))

    def train(self):
        target_iter = len(self.train_loader) * 50

        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.validate()

            # stop early for zeroshot if target number of images have been processed
            cur_iter = self.epoch * self.train_seen
            if cur_iter > target_iter:
                break

        # train binary classifier 
        if self.pixel_embeddings and self.seen_classifier:
            for epoch in range(self.max_seen_clf_epoch):
                self.seen_clf_epoch = epoch
                self.train_seen_clf()
                self.validate_seen_clf()