#!/usr/bin/env python
import argparse
import datetime
import models, trainer_fcn, trainer_seenmask
import os
import os.path as osp
import pascal_dataset, context_dataset
import pytz
import torch
import torch.nn as nn
import yaml

from configs import configurations
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser()

    # default value args
    parser.add_argument('-n', '--name', type=str, default=None, help='name of checkpoint folder')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu number; -1 for cpu')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
    parser.add_argument("-dir", "--data_dir", type=str, default='/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation', help='path for storing dataset, logs, and models')
    parser.add_argument("-tb", "--tb_dir", type=str, default='/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/tb', help='path to tensorboard directory')

    # override cfg args
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'test_fcn', 'test_all'], help='choose among five training/testing mode choices')
    parser.add_argument("-d", "--dataset", type=str, choices=['pascal', 'context'], help='dataset name')
    parser.add_argument('-tu', '--train_unseen', type=str, help='delimited list input for zero-shot train split unseen classes')
    parser.add_argument('-vu', '--val_unseen', type=str, help='delimited list input for zero-shot val split unseen classes')
    parser.add_argument('-e', '--embed_dim', type=int, choices=[2, 5, 10, 20, 21, 50, 100, 200, 300], help='dimensionality of joint embeddings space')
    parser.add_argument('-ve', '--fcn_epochs', type=int, help='maximum number of training epochs for FCN')
    parser.add_argument('-lr', '--fcn_learning_rate', type=float, help='FCN\'s learning rate')
    parser.add_argument('-loss', '--fcn_loss', type=str, choices=['cos','mse', 'cross_entropy'], help='FCN training loss function if using embeddings')
    parser.add_argument('-o', '--fcn_optim', type=str, choices=['sgd','adam'], help='optimizer for updating FCN model')
    parser.add_argument('-se', '--seenmask_epochs', type=int, help='max number of training epochs for the seenmask classifier')
    parser.add_argument('-slr', '--seenmask_learning_rate', type=float, help='seenmask layer learning rate')

    # update optional cfg arg
    parser.add_argument('-oh', '--one_hot_embed', help='make embeddings one-hot embeddings for updating model', action='store_true')
    parser.add_argument('-fu', '--forced_unseen', help='only predict along unseen classes for unseen pixel during val', action='store_true')
    parser.add_argument('-r', '--resume', type=str, help='fcn model checkpoint path')
    
    # parse args and update cfg
    args = parser.parse_args()
    name, gpu, cfg, data_dir, tb_dir = args.name, args.gpu, configurations[args.config], args.data_dir, args.tb_dir # extract default value args
    cfg = update_cfg_with_args(cfg, args)
    validate_cfg(cfg)

    # initialize logging and tensorboard writer
    log_dir = get_log_dir(name, args.config, cfg, data_dir)
    run_name = log_dir.split('/')[-1]
    tb_path = osp.join(tb_dir, run_name)
    tb_writer = SummaryWriter(tb_path)
    output_cfg(cfg, log_dir, tb_writer)

    # initialize CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    if cuda == -1:
        cuda=False
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    kwargs = {'val_unseen': cfg['val_unseen'], 'transform': True, 'embed_dim': cfg['embed_dim'], 'one_hot_embed': cfg['one_hot_embed'], 'data_dir': data_dir}

    if cfg['dataset'] == "pascal":
        pascal_dataset.download(data_dir)
        train_dataset = pascal_dataset.PascalVOC(split='train', **kwargs)
        train_seen_dataset = pascal_dataset.PascalVOC(split='train_seen', train_unseen=cfg['train_unseen'], **kwargs)
        val_dataset = pascal_dataset.PascalVOC(split='val', **kwargs)
    elif cfg['dataset'] == "context":
        context_dataset.download(data_dir)
        train_dataset = context_dataset.PascalContext(split='train', **kwargs)
        train_seen_dataset = context_dataset.PascalContext(split='train_seen', train_unseen=cfg['train_unseen'], **kwargs)
        val_dataset = context_dataset.PascalContext(split='val', **kwargs)

    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    train_seen_loader = torch.utils.data.DataLoader(train_seen_dataset, batch_size=1, shuffle=True, **kwargs) # TODO: add val_unseen to everything
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)  
    label_names = train_dataset.class_names

    # output tb/log counts on train_seen, train_unseen, val
    n_train_seen = str(len(train_seen_loader))
    n_train_unseen = str(len(train_loader) - len(train_seen_loader))
    n_val = str(len(val_loader))

    tb_writer.add_text('num/train_seen',  n_train_seen)
    tb_writer.add_text('num/train_unseen', n_train_unseen)
    tb_writer.add_text('num/val', n_val)

    if not osp.exists(osp.join(log_dir, 'counts.csv')):
        with open(osp.join(log_dir, 'counts.csv'), 'w') as f:
            f.write(','.join(['train_seen', 'train_unseen', 'val']) + '\n')
            f.write(','.join([n_train_seen, n_train_unseen, n_val]) + '\n')

    # 2. model
    if cfg['embed_dim']:
        model = models.FCN32s(n_class=cfg['embed_dim'])
    else:
        model = models.FCN32s(n_class=21)
    start_epoch = 0
    start_iteration = 0 
    
    # load fcn with saved weights
    checkpoint = None
    if cfg['load_fcn_path']:
        load_path =  osp.join(data_dir, 'logs', cfg['load_fcn_path'], 'best')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict is False for backwards compatibility
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    # initialize fcn with vgg weights
    else:
        vgg16 = models.VGG16(pretrained=True, data_dir=data_dir)
        model.copy_params_from_vgg16(vgg16)

    if cuda:
        model = model.cuda()

    # 3. fcn optimizer and trainer
    if cfg['fcn_optim'] == "sgd":
        params =  [{'params': get_parameters(model, bias=False)}, 
            {'params': get_parameters(model, bias=True), 'lr': cfg['fcn_lr'] * 2, 'weight_decay': 0}] # conv2d bias
        optim = torch.optim.SGD(params, lr=cfg['fcn_lr'], momentum=.99, weight_decay=0.0005)
    elif cfg['fcn_optim'] == "adam":
        params = [{'params': get_parameters(model, bias=False)}, 
            {'params': get_parameters(model, bias=True), 'lr': cfg['fcn_lr'] * 2}] # conv2d bias
        optim = torch.optim.Adam(params, lr=cfg['fcn_lr'])

    if cfg['load_fcn_path']:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # train fcn
    all_unseen = cfg['train_unseen'] + cfg['val_unseen']
    fcn_trainer = trainer_fcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_seen_loader,
        val_loader=val_loader,
        log_dir=log_dir,
        dataset=cfg['dataset'],
        max_epoch=cfg['fcn_epochs'],
        pixel_embeddings=cfg['embed_dim'],
        loss_func=cfg['fcn_loss'],
        tb_writer=tb_writer,
        unseen=all_unseen,
        val_unseen=cfg['val_unseen'],
        label_names=label_names,
        forced_unseen=cfg['forced_unseen'],
    )
    fcn_trainer.epoch, fcn_trainer.iteration = start_epoch, start_iteration

    if cfg['mode'] == 'train':
        if cfg['fcn_epochs'] > 0:
            fcn_trainer.train()

        # 4. train seenmask
        if cfg['seenmask_epochs'] > 0:
            # fix fcn's VGG weights. learn final linear layer mapping fc7 to seenmask
            for param in model.parameters():
                param.requires_grad = False
            for p in model.seenmask_score.parameters():
                p.requires_grad = True
            for p in model.seenmask_upscore.parameters():
                p.requires_grad = True

            # optimizer
            params = [{'params': get_parameters(model, seenmask=True)}]
            optim = torch.optim.Adam(params, lr=cfg['seenmask_lr'])

            if not checkpoint:
                load_path =  osp.join(data_dir, 'logs', run_name, 'best')
                checkpoint = torch.load(load_path)

            seenmask_trainer = trainer_seenmask.Trainer(
                cuda=cuda,
                model=model,
                optimizer=optim,
                train_loader=train_loader, 
                val_loader=val_loader,
                log_dir=log_dir,
                dataset=cfg['dataset'],
                max_epoch=cfg['seenmask_epochs'], 
                tb_writer=tb_writer,
                checkpoint=checkpoint,
                unseen=cfg['train_unseen'],
            )
            seenmask_trainer.train()

    # 5. test
    elif cfg['mode'] == 'test_fcn':
        fcn_trainer.validate(both_fcn_and_seenmask=False) 
    elif cfg['mode'] == 'test_all':
        fcn_trainer.validate(both_fcn_and_seenmask=True)

def update_cfg_with_args(cfg, args):
    # override cfg 
    if args.mode:
        cfg['mode'] = args.mode
    if args.dataset:
        cfg['dataset'] = args.dataset
    if args.train_unseen:
        cfg['train_unseen'] = [int(item) for item in args.train_unseen.split(',')]
    if args.val_unseen:
        cfg['val_unseen'] = [int(item) for item in args.val_unseen.split(',')]
    if args.embed_dim:
        cfg['embed_dim'] = args.embed_dim
    if args.fcn_epochs:
        cfg['fcn_epochs'] = args.fcn_epochs
    if args.fcn_learning_rate:
        cfg['fcn_lr'] = args.fcn_learning_rate     
    if args.fcn_loss:
        cfg['fcn_loss'] = args.fcn_loss
    if args.fcn_optim:
        cfg['fcn_optim'] = args.fcn_optim
    if args.seenmask_learning_rate:
        cfg['seenmask_lr'] = args.seenmask_learning_rate

    # update optional cfg arg; ensure all cfg fields have default value if no arg given
    cfg['one_hot_embed'] = args.one_hot_embed if args.one_hot_embed else cfg.get('one_hot_embed')
    cfg['forced_unseen'] = args.forced_unseen if args.forced_unseen else cfg.get('forced_unseen')
    cfg['load_fcn_path'] = args.resume if args.resume else cfg.get('load_fcn_path')

    return cfg

def validate_cfg(cfg):
    # TODO: add more

    if cfg['one_hot_embed'] and cfg['embed_dim'] != 21 and cfg['dataset'] == "pascal":
        raise Exception('joint-embedding space must be size of one-hot embedding space')

    if cfg['one_hot_embed'] and cfg['embed_dim'] != 33 and cfg['dataset'] == "context":
        raise Exception('joint-embedding space must be size of one-hot embedding space')

    if cfg['mode'] in ['test_fcn', 'test_all'] and not cfg['load_fcn_path']:
        raise Exception('must load model path via -r flag for test mode')

    if cfg['fcn_epochs'] < 1 and not cfg['load_fcn_path']:
        raise Exception('must load model path via -r flag for test mode')

    if cfg['seenmask_epochs'] > 0 and len(cfg['train_unseen']) < 1:
        raise Exception("can't train the seenmask classifier without train_unseen specified")

    if cfg['embed_dim'] == 0 and cfg['fcn_loss'] in ['cos', 'mse']:
        raise Exception("invalid loss function because pixel embedding dimensionality not defined")

def get_log_dir(model_name, cfg_num, cfg, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if model_name:
        name = '%s_' % model_name
    else:
        name = ''

    name += "CFG_%d_" % int(cfg_num)
    
    for k, v in cfg.items():

        # ignore optional arguments...
        if k in ['one_hot_embed','forced_unseen'] and not v:
            continue
        elif k == 'load_fcn_path':
            continue
        elif k in ['train_unseen', 'val_unseen']:
            if v:
                name += '%s_%s_' % (k.upper(), str(True))
            else:
                name += '%s_%s_' % (k.upper(), str(False))
        else:
            name += '%s_%s_' % (k.upper(), str(v))

    now = datetime.datetime.now(pytz.timezone('US/Eastern'))
    name += 'TIME_%s_' % now.strftime('%Y%m%d-%H%M%S')

    log_dir = osp.join(data_dir, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

def output_cfg(cfg, log_dir, writer):

    # print cfg to stdout
    for k, v in cfg.items():
        print(k, v)

    # write cfg to yaml log
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    # write cfg to tensorboard        
    cfg_str = '\n'.join(['%s: %s'%(k, str(v)) for k,v in cfg.items()])
    writer.add_text("cfg", cfg_str)

def get_parameters(model, bias=False, seenmask=False):
    if seenmask:
        for p in model.seenmask_score.parameters():
            yield p
        for p in model.seenmask_upscore.parameters():
            yield p
    else:
        modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            nn.Sequential,
            models.FCN32s,
        )
        for name, m in model.named_modules():
            if name in ['seenmask_score', 'seenmask_upscore']:
                continue
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, nn.ConvTranspose2d):
                # weight is frozen because it is just a bilinear upmodule_list.extendsampling
                if bias:
                    assert m.bias is None
            elif isinstance(m, modules_skipped):
                continue
            else:
                raise ValueError('Unexpected module: %s' % str(m))


if __name__ == '__main__':
    main()
