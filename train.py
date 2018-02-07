#!/usr/bin/env python
import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
import pytz
import torch
import yaml
import dataset, models, trainer
import torch.nn as nn
import subprocess

configurations = {
    # no embeddings, softmax output
    1: dict(
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        embed_dim=None,
        zeroshot=False,
    ),

    # 50D embeddings
    2: dict(
        lr=1.0e-15,
        momentum=0.99,
        weight_decay=.0005,
        embed_dim=50,
        zeroshot=False,
    ),
}

data_dir = open('data_dir.txt', 'r').read()

def get_log_dir(model_name, cfg):
    if model_name:
        name = '%s' % model_name
    else:
        name = ''
    for k, v in cfg.items():
        name += '_%s_%s' % (k.upper(), str(v))
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')

    # output
    log_dir = osp.join(data_dir, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


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
    parser.add_argument('--name', type=str, default=None, help='Name of checkpoint folder')
    parser.add_argument('-g', '--gpu', type=int, default=0, required=False)
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    parser.add_argument("--max_epoch", default=15, type=int, help="maximum number of training epochs")
    parser.add_argument("--dataset", type=str, default="pascal", help="pascal") # TODO: add other datasets here eventually
    args = parser.parse_args()

    gpu = args.gpu
    max_epoch = args.max_epoch
    cfg = configurations[args.config]
    embed_dim = cfg['embed_dim']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    out = get_log_dir(args.name, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset.SBDClassSeg(split='train', transform=True, embed_dim=embed_dim),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        dataset.VOC2011ClassSeg(split='seg11valid', transform=True, embed_dim=embed_dim),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model
    if embed_dim:
        model = models.FCN32s(n_class=embed_dim)    
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
        vgg16 = models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0}, # Conv2D bias
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    fcn_trainer = trainer.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_epoch=max_epoch,
		pixel_embeddings=embed_dim
    )
    fcn_trainer.epoch = start_epoch
    fcn_trainer.iteration = start_iteration
    fcn_trainer.train()


if __name__ == '__main__':
    main()
