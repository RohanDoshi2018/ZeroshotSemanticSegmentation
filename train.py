#!/usr/bin/env python
import argparse
import datetime
import os
import os.path as osp
import pytz
import torch
import yaml
import pascal_dataset, context_dataset
import models, trainer
import torch.nn as nn
from tensorboardX import SummaryWriter

configurations = {

    ## PASCAL

    # fcn baseline: no embeddings, softmax output (mse optimizer)
    1: dict(
        max_epoch=50,
        lr=1e-10,
        momentum=0.99,
        weight_decay=0.0005,
        embed_dim=0,
        one_hot_embed=False,
        loss_func=None, # not used
        bk_loss=True,
        unseen=None,
        dataset='pascal',
        optimizer='sgd',
    ),

    # fcn baseline, but with adam optimizer; 1 vs 2 conclusion: use adam optimizer 
    2: dict(
        max_epoch=50,
        lr=1e-5,
        momentum=None,
        weight_decay=0,
        embed_dim=0,
        one_hot_embed=False,
        loss_func=None,
        bk_loss=True,
        unseen=None,
        dataset='pascal',
        optimizer='adam',
    ),


    # 21D one-hot embeddings, mse loss
    3: dict(
        max_epoch=50,
        lr=1e-5,
        momentum=None,
        weight_decay=0,
        embed_dim=21,
        one_hot_embed=True,
        loss_func='mse',
        bk_loss=True,
        unseen=None,
        dataset='pascal',
        optimizer='adam',
    ),

    # 21D one-hot, cosine loss; 3 vs 4 conclusion: use cosine loss
    4: dict(
        max_epoch=50,
        lr=1e-5,
        momentum=None,
        weight_decay=0,
        embed_dim=21,
        one_hot_embed=True,
        loss_func='cos',
        bk_loss=True,
        unseen=None,
        dataset='pascal',
        optimizer='adam',
    ),

    # 20D; mse vs cos args conclusion: use cosine loss
    5: dict(
        max_epoch=50, # 8498 training images
        lr=5e-5,
        momentum=None,
        weight_decay=None,
        embed_dim=20,
        one_hot_embed=False,
        loss_func='cos',
        bk_loss=True,
        unseen=None,
        dataset='pascal',
        optimizer='adam',
    ),

    # 20D zeroshot (unseen: 10 classes)
    6: dict(
        max_epoch=130, # 3311 seen and 5187 unseen training images
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        embed_dim=20,
        one_hot_embed=False,
        loss_func='cos',
        bk_loss=True,
        unseen=[6, 7, 13, 14, 15, 16, 17, 18, 19, 20],
        dataset='pascal',
        optimizer='adam',
    ),

    ## CONTEXT

    # 20D context
    7: dict(
        max_epoch=50,
        lr=5e-5,
        momentum=None,
        weight_decay=None,
        embed_dim=20,
        one_hot_embed=False,
        loss_func='cos',
        bk_loss=True,
        unseen=None,
        dataset='context',
        optimizer='adam',
    ),

    # 20D context zeroshot (unseen: 16 classes)
    8: dict(
        max_epoch=100,
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        embed_dim=20,
        one_hot_embed=False,
        loss_func='cos',
        bk_loss=True,
        unseen=[2,4,6,8,10],
        # unseen=[2+2*i for i in range(16)], # 2, 4, ..., 32
        dataset='context',
        optimizer='adam',
    ),

    # 20D zeroshot with forced unseen (unseen: 10 classes)
    9: dict(
        max_epoch=130,
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        embed_dim=20,
        one_hot_embed=False,
        loss_func='cos',
        bk_loss=True,
        unseen=[6, 7, 13, 14, 15, 16, 17, 18, 19, 20],
        dataset='pascal',
        optimizer='adam',
    ),

}

def get_log_dir(model_name, cfg, cfg_num, data_dir):
    if model_name:
        name = '%s' % model_name
    else:
        name = ''
    name += "_CFG_%d" % int(cfg_num)
    for k, v in cfg.items():
        if k == 'unseen':
            if cfg['unseen']:
                name += '_%s_%s' % (k.upper(), str(True))
            else:
                name += '_%s_%s' % (k.upper(), str(False))
        else:
            name += '_%s_%s' % (k.upper(), str(v))

    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')

    # output
    log_dir = osp.join(data_dir, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return name, log_dir

def write_cfg_to_tb(cfg, writer):
    for k, v in cfg.items():
        writer.add_text("cfg/%s" % k.upper(), str(v))

def get_parameters(model, bias=False):
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        models.FCN32s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None, help='name of checkpoint folder')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number; -1 for cpu')
    parser.add_argument('-r', '--resume', help='checkpoint path')
    parser.add_argument('-c', '--config', type=int, default=3, choices=configurations.keys()) # reset to default 1 eventually
    parser.add_argument('-me', '--max_epoch', type=int, help='maximum number of training epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate')
    parser.add_argument('-e', '--embed_dim', type=int, help='dimensionality of joint embeddings space')
    parser.add_argument('-loss', '--loss_func', type=str, choices=['cos','mse', None], help='training loss function if using embeddings')
    parser.add_argument('-bkl', '--bk_loss', type=bool, help='compute loss on background pixels?')
    parser.add_argument("-d", "--dataset", type=str, choices=['pascal', 'context'], help='dataset name')
    parser.add_argument("-dir", "--data_dir", type=str, default='/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation', help=' path where to store dataset, logs, and models')
    parser.add_argument("-tb", "--tb_dir", type=str, default='/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/tb', help='path to tensorboard directory')
    parser.add_argument('-o', '--optim', type=str, choices=['sgd','adam'], help='optimizer for updating model')
    parser.add_argument('-u', '--unseen', type=str, help='delimited list input for zero-shot unseen classes')

    # extract args
    args = parser.parse_args()
    name = args.name
    gpu = args.gpu
    resume = args.resume
    cfg = configurations[args.config]
    data_dir = args.data_dir
    tb_dir = args.tb_dir

    # update cfg arguments
    if args.max_epoch:
        cfg['max_epoch'] = args.max_epoch
    if args.embed_dim:
        cfg['embed_dim'] = args.embed_dim
    if args.learning_rate:
        cfg['lr'] = args.learning_rate
    if args.loss_func:
        cfg['loss_func'] = args.loss_func
    if args.bk_loss:
        cfg['bk_loss'] = args.bk_loss
    if args.dataset:
        cfg['dataset'] = args.dataset
    if args.optim:
        cfg['optimizer'] = args.optim
    if args.unseen:
        cfg['unseen'] = [int(item) for item in args.unseen.split(',')]

    if cfg['one_hot_embed'] and cfg['embed_dim'] != 21 and cfg['dataset'] == "pascal":
        raise Exception('joint-embedding space must be size of one-hot embedding space')

    # initialize logging
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    name, out = get_log_dir(name, cfg, args.config, data_dir)
    
    # initialize tensorboard writer
    tb_path = osp.join(tb_dir, name)
    tb_writer = SummaryWriter(tb_path)
    write_cfg_to_tb(cfg, tb_writer)

    # set up CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    if cuda == -1:
        cuda=False
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    label_names = None
    if cfg['dataset'] == "pascal":

        pascal_dataset.download(data_dir)

        train_dataset = pascal_dataset.SBDClassSeg(split='train', transform=True, embed_dim=cfg['embed_dim'], one_hot_embed=cfg['one_hot_embed'], data_dir=data_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        val_dataset = pascal_dataset.VOC2011ClassSeg(split='seg11valid', transform=True, embed_dim=cfg['embed_dim'], one_hot_embed=cfg['one_hot_embed'], data_dir=data_dir)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

        label_names = train_dataset.class_names

    elif cfg['dataset'] == "context":

        context_dataset.download(data_dir)

        train_dataset = context_dataset.VOCContext(split='train', transform=True, embed_dim=cfg['embed_dim'], one_hot_embed=cfg['one_hot_embed'], data_dir=data_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        val_dataset = context_dataset.VOCContext(split='val', transform=True, embed_dim=cfg['embed_dim'], one_hot_embed=cfg['one_hot_embed'], data_dir=data_dir)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)  

        label_names = train_dataset.class_names 

    else:

        raise Exception("Datasets apart from Pascal not implemented")

    # 2. model
    if cfg['embed_dim']:
        model = models.FCN32s(n_class=cfg['embed_dim'])
    else:
        model = models.FCN32s(n_class=21)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = models.VGG16(pretrained=True, data_dir=data_dir)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer
    if cfg['optimizer'] == "sgd":
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0}, # Conv2D bias
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == "adam":
        optim = torch.optim.Adam([
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True), 'lr': 2 * cfg['lr']}, # Conv2D bias
        ],
        lr=cfg['lr'])
    else:
        raise Exception("Unknown optimizer chosen")

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # 4. training

    fcn_trainer = trainer.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        dataset=cfg['dataset'],
        max_epoch=cfg['max_epoch'],
		pixel_embeddings=cfg['embed_dim'],
        loss_func=cfg['loss_func'],
        background_loss=cfg['bk_loss'],
        tb_writer=tb_writer,
        unseen=cfg['unseen'],
        label_names=label_names,
    )
    fcn_trainer.epoch = start_epoch
    fcn_trainer.iteration = start_iteration
    fcn_trainer.train()


if __name__ == '__main__':
    main()
