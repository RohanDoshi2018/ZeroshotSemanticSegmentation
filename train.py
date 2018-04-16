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
from configs import configurations

def get_log_dir(model_name, cfg, cfg_num, data_dir):
    if model_name:
        name = '%s' % model_name
    else:
        name = ''
    name += "_CFG_%d" % int(cfg_num)
    for k, v in cfg.items():
        if k == 'resume_model_path':
            continue
        elif k == 'unseen':
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

def get_parameters(model, bias=False, seen_clf=False, fixed_vgg=False):
    if seen_clf:
        for p in model.seen_clf_score.parameters():
            yield p
    elif fixed_vgg:
        for p in model.score_fr.parameters():
            yield p
        for p in model.upscore.parameters():
            yield p
    else:
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
    parser.add_argument('-m', '--mode', type=str, choices=['train','test'], help='train or test; must provide model for test')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number; -1 for cpu')
    parser.add_argument('-r', '--resume', help='model checkpoint path; required for test mode')
    parser.add_argument('-c', '--config', type=int, default=5, choices=configurations.keys()) # reset to default 1 eventually
    parser.add_argument('-me', '--max_epoch', type=int, help='maximum number of training epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate')
    parser.add_argument('-e', '--embed_dim', type=int, help='dimensionality of joint embeddings space')
    parser.add_argument('-loss', '--loss_func', type=str, choices=['cos','mse', None], help='training loss function if using embeddings')
    parser.add_argument("-d", "--dataset", type=str, choices=['pascal', 'context'], help='dataset name')
    parser.add_argument("-dir", "--data_dir", type=str, default='/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation', help=' path where to store dataset, logs, and models')
    parser.add_argument("-tb", "--tb_dir", type=str, default='/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/tb', help='path to tensorboard directory')
    parser.add_argument('-o', '--optim', type=str, choices=['sgd','adam'], help='optimizer for updating model')
    parser.add_argument('-u', '--unseen', type=str, help='delimited list input for zero-shot unseen classes')

    # extract args
    args = parser.parse_args()
    name = args.name
    gpu = args.gpu
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
    if args.dataset:
        cfg['dataset'] = args.dataset
    if args.optim:
        cfg['optimizer'] = args.optim
    if args.unseen:
        cfg['unseen'] = [int(item) for item in args.unseen.split(',')]
    if args.mode:
        cfg['mode'] = args.mode
    if args.resume:
        cfg['resume_model_path'] = args.resume

    if cfg['one_hot_embed'] and cfg['embed_dim'] != 21 and cfg['dataset'] == "pascal":
        raise Exception('joint-embedding space must be size of one-hot embedding space')

    if cfg['mode'] == 'test' and not cfg.get('resume_model_path'):
        raise Exception('must load model path via -r flag for test mode')

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
    # TODO: preprocess and partition the training data into train_seen and train_unseen; build data loaders
    kwargs = {'transform': True, 'embed_dim': cfg['embed_dim'], 'one_hot_embed': cfg['one_hot_embed'], 'data_dir': data_dir}
    if cfg['dataset'] == "pascal":
        pascal_dataset.download(data_dir)
        train_dataset = pascal_dataset.PascalVOC(split='train', **kwargs)
        val_dataset = pascal_dataset.PascalVOC(split='val', **kwargs)
    elif cfg['dataset'] == "context":
        context_dataset.download(data_dir)
        train_dataset = context_dataset.PascalContext(split='train', **kwargs)
        val_dataset = context_dataset.PascalContext(split='val', **kwargs)
    else:
        raise Exception("unknown dataset")

    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)  
    label_names = train_dataset.class_names 

    # 2. model
    if cfg['embed_dim']:
        model = models.FCN32s(n_class=cfg['embed_dim'])
    else:
        model = models.FCN32s(n_class=21)
    start_epoch = 0
    start_iteration = 0
    
    if cfg.get('resume_model_path'):
        load_path =  osp.join(data_dir, 'logs', cfg['resume_model_path'], 'best')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = models.VGG16(pretrained=True, data_dir=data_dir)
        model.copy_params_from_vgg16(vgg16)

    # treat VGG as visual feature extractor by freezing weights upto fc7
    if cfg['fixed_vgg']:
        for param in model.parameters():
            param.requires_grad = False
        for p in model.score_fr.parameters():
            p.requires_grad = True
        for p in model.upscore.parameters():
            p.requires_grad = True

    if cuda:
        model = model.cuda()

    # 3. optimizer

    # TODO: initialize optimizer for seen classifier
    # TODO: figure out how to fix VGG weights when training the seen classifier

    if cfg['optimizer'] == "sgd":
        if cfg['fixed_vgg']:
            params = [{'params': get_parameters(model, fixed_vgg=True)}]
        else:
            params =  [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0}, # Conv2D bias
            ]

        optim = torch.optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == "adam":
        if cfg['fixed_vgg']:
            params = [{'params': get_parameters(model, fixed_vgg=True)}]
        else:
            params = [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2}, # Conv2D bias
            ]

        optim = torch.optim.Adam(params,lr=cfg['lr'])
    else:
        raise Exception("Unknown optimizer chosen")

    if cfg.get('resume_model_path'):
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # 4. train visual network
    forced_unseen = cfg.get('forced_unseen')

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
        tb_writer=tb_writer,
        unseen=cfg['unseen'],
        label_names=label_names,
        forced_unseen=forced_unseen,
    )
    fcn_trainer.epoch = start_epoch
    fcn_trainer.iteration = start_iteration

    # 5. TODO: train seen pixel clasifier

    # # TODO: fix vgg weights, only learn final linear layer 
    # # for binary seen pixel classification
    # if cfg['train_unseen']:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for p in model.seen_clf_score.parameters():
    #         p.requires_grad = True

    #     seen_pxl_trainer = seen_trainer.Trainer(
    #         cuda=cuda,
    #         model=model,
    #         optimizer=optim, # TODO: change
    #         train_loader=train_loader, # TODO: change
    #         val_loader=val_loader, # TODO: change
    #         out=out,
    #         dataset=cfg['dataset'], # TODO: change
    #         max_epoch=cfg['max_epoch'], # TODO: change
    #         pixel_embeddings=cfg['embed_dim'], # TODO: change
    #         loss_func=cfg['loss_func'], # TODO: change
    #         tb_writer=tb_writer,
    #         unseen=cfg['unseen'], # TODO: change
    #         label_names=label_names, # TODO: change
    #         forced_unseen=forced_unseen, # TODO: change
    #     )

    if cfg['mode'] == 'train':
        fcn_trainer.train()
    elif cfg['mode'] == 'test':
        fcn_trainer.validate()

if __name__ == '__main__':
    main()
