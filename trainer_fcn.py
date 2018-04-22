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

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, log_dir, dataset, max_epoch, tb_writer, 
                    pixel_embeddings=None, loss_func=None, unseen=None, val_unseen=None, label_names=None, forced_unseen=False):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.dataset = dataset
        self.max_epoch = max_epoch
        self.tb_writer = tb_writer
        self.pixel_embeddings = pixel_embeddings
        self.loss_func = loss_func
        self.unseen = unseen # all unseen classes (both train_unseen and val_unseen)
        self.val_unseen = val_unseen
        self.label_names = label_names
        self.forced_unseen = forced_unseen

        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0
        self.n_class = len(self.train_loader.dataset.class_names)
        self.timestamp_start = datetime.datetime.now(pytz.timezone('US/Eastern'))
        self.seen =  [x for x in range(self.n_class) if x not in self.unseen]

        # set up embeddings
        if self.pixel_embeddings:
            # each embedding has norm between 0 and 1
            embed_arr = utils.load_obj('datasets/%s/embeddings/norm_embed_arr_%s' % (dataset, str(pixel_embeddings)))
            if cuda:
                self.embeddings = Variable(torch.from_numpy(embed_arr).cuda().float(), requires_grad=False)
            else:
                self.embeddings = Variable(torch.from_numpy(embed_arr).float(), requires_grad=False)

            # embeddings used for forced_unseen or testing both seenmask and fcn in combination
            seen_embed_arr, unseen_embed_arr = np.zeros(embed_arr.shape), np.zeros(embed_arr.shape)
            seen_embed_arr[self.seen, :] = embed_arr[self.seen, :]
            unseen_embed_arr[self.unseen, :] = embed_arr[self.unseen, :]
            if cuda:
                self.seen_embeddings = Variable(torch.from_numpy(seen_embed_arr).cuda().float(), requires_grad=False)
                self.unseen_embeddings = Variable(torch.from_numpy(unseen_embed_arr).cuda().float(), requires_grad=False)
            else:               
                self.seen_embeddings = Variable(torch.from_numpy(seen_embed_arr).float(), requires_grad=False)
                self.unseen_embeddings = Variable(torch.from_numpy(unseen_embed_arr).float(), requires_grad=False)

        self.train_log_headers = ['epoch', 'iteration', 'train/loss', 'train/pxl_acc', 'train/class_acc', 'train/mean_iu', 'train/fwavacc', 'elapsed_time']

        if self.unseen:
            self.val_log_headers = ['epoch', 'iteration', 'val/loss', 'val/pxl_acc', 'val/class_acc', 'val/mean_iu', 'val/fwavacc', 
                                    'val/seen/pxl_acc', 'val/seen/class_acc', 'val/seen/mean_iu', 'val/seen/fwavacc',         
                                    'val/unseen/pxl_acc', 'val/unseen/class_acc', 'val/unseen/mean_iu', 'val/unseen/fwavacc','elapsed_time']
        else:
            self.val_log_headers = ['epoch', 'iteration', 'val/loss', 'val/pxl_acc', 'val/class_acc', 'val/mean_iu', 'val/fwavacc', 'elapsed_time']

        if not osp.exists(osp.join(self.log_dir, 'train_log.csv')):
            with open(osp.join(self.log_dir, 'train_log.csv'), 'w') as f:
                f.write(','.join(self.train_log_headers) + '\n')

        if not osp.exists(osp.join(self.log_dir, 'val_log.csv')):
            with open(osp.join(self.log_dir, 'val_log.csv'), 'w') as f:
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

        score = self.model(data, mode='fcn')

        # get loss
        if self.loss_func == "cos":
            loss = utils.cosine_loss(score, target, target_embed)
        elif self.loss_func == "mse":
            loss = utils.mse_loss(score, target, target_embed)
        elif self.loss_func == "cross_entropy":
            loss = utils.cross_entropy2d(score, target, size_average=False)

        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')

        # inference 
        if self.pixel_embeddings:
            if self.forced_unseen:
                lbl_pred = utils.infer_lbl_forced_unseen(score, target, self.seen_embeddings, self.unseen_embeddings, self.unseen, self.cuda)
            else:
                lbl_pred = utils.infer_lbl(score, self.embeddings, self.cuda)
        else:
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()

        return score, loss, lbl_pred, lbl_true


    def forward_szn(self, data, target):
        #  get score
        target, target_embed = target

        if self.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        if self.cuda:
            target_embed = target_embed.cuda()
        target_embed = Variable(target_embed)

        fcn_score, seen_mask_score = self.model(data, mode='both')

        # get fcn loss
        if self.loss_func == "cos":
            loss = utils.cosine_loss(fcn_score, target, target_embed)
        elif self.loss_func == "mse":
            loss = utils.mse_loss(fcn_score, target, target_embed)

        lbl_pred = utils.infer_lbl_szn(fcn_score, seen_mask_score, self.seen_embeddings, self.unseen_embeddings, self.cuda)

        lbl_true = target.data.cpu()

        return fcn_score, loss, lbl_pred, lbl_true

    def train_epoch(self):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):

            score, loss, lbl_pred, lbl_true = self.forward(data, target)

            self.optim.zero_grad()
            loss.backward()            
            self.optim.step()

            print("FCN Train Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | score_fr grad sum {:15.0f} | upscore grad sum {:15.0f} | score sum {:10.5f}".format(
                int(self.epoch), int(batch_idx), float(loss.data[0]), float(self.model.score_fr.weight.grad.sum().data[0]),
                float(self.model.upscore.weight.grad.sum().data[0]), float(score.sum().data[0])))

            metrics = utils.label_accuracy_score(lbl_true.numpy(), lbl_pred, self.n_class)

            # update the training logs
            with open(osp.join(self.log_dir, 'train_log.csv'), 'a') as f:
                elapsed_time = (datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + list(metrics) + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            # write to tensorboard
            self.tb_writer.add_scalar('fcn/train/loss', loss.data[0], self.iteration)
            self.tb_writer.add_scalar('fcn/train/pxl_acc', metrics[0], self.iteration)
            self.tb_writer.add_scalar('fcn/train/class_acc', metrics[1], self.iteration)
            self.tb_writer.add_scalar('fcn/train/mean_iu', metrics[2], self.iteration)
            self.tb_writer.add_scalar('fcn/train/fwavacc', metrics[3], self.iteration)

            self.iteration += 1

    def validate(self, both_fcn_and_seenmask=False):
        self.model.eval()

        val_loss = 0
        lbl_trues, lbl_preds, visualizations = [], [], []

        for batch_idx, (data, target) in enumerate(self.val_loader):

            if both_fcn_and_seenmask:
                score, loss, lbl_pred, lbl_true = self.forward_szn(data, target)
            else:
                score, loss, lbl_pred, lbl_true = self.forward(data, target)

            val_loss += float(loss.data[0])
            print("Test Epoch {:<5} | Iteration {:<5} | Loss {:5.5f} | Score Sum {:10.5f}".format(int(self.epoch), int(batch_idx), float(loss.data[0]), float(score.sum().data[0])))

            img, lt, lp = data[0], lbl_true[0], lbl_pred[0] # eliminate first dimension (n=1) for visualization
            img, lt = self.val_loader.dataset.untransform(img, lt)
            lbl_trues.append(lt)
            lbl_preds.append(lp)

            # generate visualization for first few images of val_loader
            if len(visualizations) < 25:
                viz = vis_utils.visualize_segmentation(lbl_pred=lp, lbl_true=lt, img=img, n_class=self.n_class, label_names=self.label_names, unseen=self.val_unseen)
                visualizations.append(viz)

        # save the visualizaton image
        if both_fcn_and_seenmask:
            out = osp.join(self.log_dir, 'szn_viz')
        else:
            out = osp.join(self.log_dir, 'fcn_viz')        
    
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%d.jpg' % self.epoch)

        viz_img = fcn.utils.get_tile_image(visualizations)
        scipy.misc.imsave(out_file, viz_img)

        # update the validation log for the current epoch
        if self.unseen:
            metrics = utils.label_accuracy_score(lbl_trues, lbl_preds, self.n_class, unseen=self.val_unseen)
            metrics, seen_metrics, unseen_metrics = metrics

            self.tb_writer.add_scalar('fcn/val/seen/pxl_acc', seen_metrics[0], self.epoch)
            self.tb_writer.add_scalar('fcn/val/seen/class_acc', seen_metrics[1], self.epoch)
            self.tb_writer.add_scalar('fcn/val/seen/mean_iu', seen_metrics[2], self.epoch)
            self.tb_writer.add_scalar('fcn/val/seen/fwavacc', seen_metrics[3], self.epoch)

            self.tb_writer.add_scalar('fcn/val/unseen/pxl_acc', unseen_metrics[0], self.epoch)
            self.tb_writer.add_scalar('fcn/val/unseen/class_acc', unseen_metrics[1], self.epoch)
            self.tb_writer.add_scalar('fcn/val/unseen/mean_iu', unseen_metrics[2], self.epoch)
            self.tb_writer.add_scalar('fcn/val/unseen/fwavacc', unseen_metrics[3], self.epoch)

            print('seen pxl_acc: %.3f'%seen_metrics[0])
            print('seen class_acc: %.3f'%seen_metrics[1])
            print('seen mean_iu: %.3f'%seen_metrics[2])
            print('seen fwavacc: %.3f'%seen_metrics[3])

            print('unseen pxl_acc: %.3f'%unseen_metrics[0])
            print('unseen class_acc: %.3f'%unseen_metrics[1])
            print('unseen mean_iu: %.3f'%unseen_metrics[2])
            print('unseen fwavacc: %.3f'%unseen_metrics[3])


        else:
            metrics = utils.label_accuracy_score(lbl_trues, lbl_preds, self.n_class)
        
        val_loss /= len(self.val_loader) # val loss is averaged across all the images
        
        with open(osp.join(self.log_dir, 'val_log.csv'), 'a') as f:
            elapsed_time = datetime.datetime.now(pytz.timezone('US/Eastern')) - self.timestamp_start
            if self.unseen:
                log = [self.epoch, self.iteration] + [val_loss] + list(metrics) + list(seen_metrics) + list(unseen_metrics) + [elapsed_time]
            else:
                log = [self.epoch, self.iteration] + [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # write metrics to tensorboard
        self.tb_writer.add_scalar('fcn/val/loss', val_loss, self.epoch)
        self.tb_writer.add_scalar('fcn/val/pxl_acc', metrics[0], self.epoch)
        self.tb_writer.add_scalar('fcn/val/class_acc', metrics[1], self.epoch)
        self.tb_writer.add_scalar('fcn/val/mean_iu', metrics[2], self.epoch)
        self.tb_writer.add_scalar('fcn/val/fwavacc', metrics[3], self.epoch)
        self.tb_writer.add_image('fcn/segmentations', viz_img, self.epoch)

        print('overall pxl_acc: %.3f'%metrics[0])
        print('overall class_acc: %.3f'%metrics[1])
        print('overall mean_iu: %.3f'%metrics[2])
        print('overall fwavacc: %.3f'%metrics[3])

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
        }, osp.join(self.log_dir, 'checkpoint')) 

        # save the weights for the best performing model so far
        if is_best:
            shutil.copy(osp.join(self.log_dir, 'checkpoint'), osp.join(self.log_dir, 'best'))

    def train(self):
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            self.validate()

            # stop early for zeroshot if the number of images processed exceeds the 
            # number of images processed after 50 epochs without zeroshot
            cur_iter = self.epoch * len(self.train_loader)
            if self.dataset == 'pascal' and cur_iter > 425000:
                break
            if self.dataset == 'context' and cur_iter > 247000:
                break