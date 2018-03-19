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

class Trainer(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, out, max_epoch, tb_writer,
                    size_average=False, pixel_embeddings=None, training_loss_func=None, background_loss=True):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.pixel_embeddings = pixel_embeddings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = out
        self.timestamp_start = datetime.datetime.now(pytz.timezone('US/Eastern'))
        self.size_average = size_average
        self.training_loss_func = training_loss_func
        self.background_loss = background_loss
        self.tb_writer = tb_writer

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.best_mean_iu = 0
        self.n_class = len(self.train_loader.dataset.class_names)

        # set up embeddings
        if self.pixel_embeddings:
            # pascal embeddings; eeach embedding has norm between 0 and 1
            self.embeddings = utils.load_obj('embeddings/norm_embed_arr_' + str(pixel_embeddings))
            if cuda:
                self.embeddings = Variable(torch.from_numpy(self.embeddings).cuda().float(), requires_grad=False)
            else:
                self.embeddings = Variable(torch.from_numpy(self.embeddings).float(), requires_grad=False)

        # set up tensorboard and other logging
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.train_log_headers = ['epoch', 'iteration', 'train/loss', 'train/pxl_acc', 'train/class_acc', 'train/mean_iu', 'train/fwavacc', 'elapsed_time']
        self.val_log_headers = ['epoch', 'iteration', 'valid/loss', 'valid/pxl_acc', 'valid/class_acc', 'valid/mean_iu', 'valid/fwavacc', 'elapsed_time']

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

        # get loss
        if self.pixel_embeddings:
            if self.training_loss_func == "cos":
                loss = utils.cosine_loss(score, target, target_embed, background_loss=self.background_loss, size_average=True) # TODO: fix this
            elif self.training_loss_func == "mse":
                loss = utils.mse_loss(score, target, target_embed, background_loss=self.background_loss, size_average=True)
            else:
                raise Exception("Unknown Loss Function")
        else:
            loss = utils.cross_entropy2d(score, target, background_loss=self.background_loss, size_average=self.size_average)

        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')

        ## evaluate inference 
        # TODO: uncomment these 3 lines once inference is infer_lbl is working
        if self.pixel_embeddings:
            lbl_pred = utils.infer_lbl(score, self.embeddings, self.cuda) # TODO: fix lbl_pred
        else:
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()

        return score, loss, lbl_pred, lbl_true

    def train_epoch(self):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):

            score, loss, lbl_pred, lbl_true = self.forward(data, target)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            print("Train Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | score_fr grad sum {:15.0f} | upscore grad sum {:15.0f} | score sum {:10.5f}".format(
                int(self.epoch), int(batch_idx), float(loss.data[0]), float(self.model.score_fr.weight.grad.sum().data[0]),
                float(self.model.upscore.weight.grad.sum().data[0]), float(score.sum().data[0])))

            # update the training logs
            metrics = utils.label_accuracy_score(lbl_true.numpy(), lbl_pred, n_class=self.n_class)
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

    def validate(self):
        self.model.eval()

        val_loss = 0
        lbl_trues, lbl_preds, visualizations = [], [], []

        for batch_idx, (data, target) in enumerate(self.val_loader):

            score, loss, lbl_pred, lbl_true = self.forward(data, target)

            val_loss += float(loss.data[0])
            print("Test Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | Score Sum {:10.5f}".format(int(self.epoch), int(batch_idx), float(loss.data[0]), float(score.sum().data[0])))

            # generate visualization for first 9 images of val_loader
            img, lt, lp = data[0], lbl_true[0], lbl_pred[0] # eliminate first dimension (n=1) for visualization
            img, lt = self.val_loader.dataset.untransform(img, lt)
            lbl_trues.append(lt)
            lbl_preds.append(lp)
            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class)
                visualizations.append(viz)

        # save the visualizaton image
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%d.jpg' % self.epoch)
        viz_img = fcn.utils.get_tile_image(visualizations)
        scipy.misc.imsave(out_file, viz_img)

        # update the validation log for the current epoch
        metrics = utils.label_accuracy_score(lbl_trues, lbl_preds, self.n_class)
        val_loss /= len(self.val_loader) # val loss is averaged across all the images
        with open(osp.join(self.out, 'val_log.csv'), 'a') as f:
            elapsed_time = datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start
            log = [self.epoch, self.iteration] + [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # write to tensorboard
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
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.validate()