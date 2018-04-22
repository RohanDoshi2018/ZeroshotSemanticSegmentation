import datetime
import fcn
import math
import numpy as np
import os
import os.path as osp
import pickle
import pytz
import scipy.misc
import shutil
import torch
import torch.nn.functional as F
import tqdm
import utils
import vis_utils

from torch.autograd import Variable

class Trainer(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, log_dir, dataset, max_epoch, tb_writer, checkpoint, unseen):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.dataset = dataset
        self.max_epoch = max_epoch
        self.tb_writer = tb_writer
        self.checkpoint = checkpoint
        self.unseen = unseen

        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0
        self.n_class = len(self.train_loader.dataset.class_names)
        self.timestamp_start = datetime.datetime.now(pytz.timezone('US/Eastern'))

        self.train_log_headers = ['epoch', 'iteration', 'train/loss', 'train/pxl_acc', 'train/class_acc', 'train/mean_iu', 'train/fwavacc', 'elapsed_time']
        if not osp.exists(osp.join(self.log_dir, 'seenmask_train_log.csv')):
            with open(osp.join(self.log_dir, 'seenmask_train_log.csv'), 'w') as f:
                f.write(','.join(self.train_log_headers) + '\n')

        self.val_log_headers = ['epoch', 'iteration', 'val/loss', 'val/pxl_acc', 'val/class_acc', 'val/mean_iu', 'val/fwavacc', 'elapsed_time']
        if not osp.exists(osp.join(self.log_dir, 'seenmask_val_log.csv')):
            with open(osp.join(self.log_dir, 'seenmask_val_log.csv'), 'w') as f:
                f.write(','.join(self.val_log_headers) + '\n')

    def forward(self, data, target):
        target, target_embed = target
        target = target.numpy()

        # reshape target into binary seenmask
        seen = [x for x in range(self.n_class) if x not in self.unseen]
        target = np.in1d(target.ravel(), seen).reshape(target.shape).astype(int)

        target = torch.from_numpy(target)

        if self.cuda:
             data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        score = self.model(data, mode='seenmask')
        loss = utils.cross_entropy2d(score, target, size_average=True)

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

            print("Seenmask Train Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | seenmask_score grad sum {:7.8f} | seenmask_upscore grad sum {:7.8f} | score sum {:10.5f}".format(
                int(self.epoch), int(batch_idx), float(loss.data[0]), float(self.model.seenmask_score.weight.grad.sum().data[0]),
                float(self.model.seenmask_upscore.weight.grad.sum().data[0]), float(score.sum().data[0])))

            metrics = utils.label_accuracy_score(lbl_true.numpy(), lbl_pred, self.n_class)

            with open(osp.join(self.log_dir, 'seenmask_train_log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + list(metrics) + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            # write to tensorboard
            self.tb_writer.add_scalar('seenmask/train/loss', loss.data[0], self.iteration)
            self.tb_writer.add_scalar('seenmask/train/pxl_acc', metrics[0], self.iteration)
            self.tb_writer.add_scalar('seenmask/train/class_acc', metrics[1], self.iteration)
            self.tb_writer.add_scalar('seenmask/train/mean_iu', metrics[2], self.iteration)
            self.tb_writer.add_scalar('seenmask/train/fwavacc', metrics[3], self.iteration)

            self.iteration += 1

    def validate(self):
        self.model.eval()

        val_loss = 0
        lbl_trues, lbl_preds, visualizations = [], [], []

        for batch_idx, (data, target) in enumerate(self.val_loader):

            score, loss, lbl_pred, lbl_true = self.forward(data, target)

            val_loss += float(loss.data[0])
            print("Seenmask Test Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | Score Sum {:10.5f}".format(int(self.epoch), int(batch_idx), float(loss.data[0]), float(score.sum().data[0])))

            img, lt, lp = data[0], lbl_true[0], lbl_pred[0] # eliminate first dimension (n=1) for visualization
            img, lt = self.val_loader.dataset.untransform(img, lt)
            lbl_trues.append(lt)
            lbl_preds.append(lp)

            # generate visualization for first few images of val_loader
            if len(visualizations) < 25:
                viz = vis_utils.visualize_seenmask(lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class, unseen=self.unseen)
                visualizations.append(viz)

        # save the visualizaton image
        out = osp.join(self.log_dir, 'seenmask_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%d.jpg' % self.epoch)
        viz_img = fcn.utils.get_tile_image(visualizations)
        scipy.misc.imsave(out_file, viz_img)



        metrics = utils.label_accuracy_score(lbl_trues, lbl_preds, self.n_class)
        val_loss /= len(self.val_loader) # val loss is averaged across all the images

        with open(osp.join(self.log_dir, 'seenmask_val_log.csv'), 'a') as f:
            elapsed_time = datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start
            log = [self.epoch, self.iteration] + [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # write metrics to tensorboard
        self.tb_writer.add_scalar('seenmask/val/loss', val_loss, self.epoch)
        self.tb_writer.add_scalar('seenmask/val/pxl_acc', metrics[0], self.epoch)
        self.tb_writer.add_scalar('seenmask/val/class_acc', metrics[1], self.epoch)
        self.tb_writer.add_scalar('seenmask/val/mean_iu', metrics[2], self.epoch)
        self.tb_writer.add_scalar('seenmask/val/fwavacc', metrics[3], self.epoch)
        self.tb_writer.add_image('fcn/segmentations', viz_img, self.epoch)

        print('pxl_acc: %.3f'%metrics[0])
        print('class_acc: %.3f'%metrics[1])
        print('mean_iu: %.3f'%metrics[2])
        print('fwavacc: %.3f'%metrics[3])

        # track and update the best mean intersection over union
        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu

        self.checkpoint['model_state_dict'] = self.model.state_dict() # TODO: verify
        torch.save(self.checkpoint, osp.join(self.log_dir, 'best')) 

    def train(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.validate()