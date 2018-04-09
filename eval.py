#!/usr/bin/env python
import argparse
import os
import os.path as osp
import fcn
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import tqdm
import pascal_dataset, models, utils 
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    # ARGUMENTS
    data_dir = '/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation'
    name = 'run' # name of the tensorboard folder
    model_name = '/baseline_sgd_CFG_1_MAX_EPOCH_50_LR_1e-10_MOMENTUM_0.99_WEIGHT_DECAY_0.0005_EMBED_DIM_0_ONE_HOT_EMBED_False_TRAIN_LOSS_FUNC_None_BACKGROUND_LOSS_True_ZEROSHOT_False_DATASET_pascal_OPTIMIZER_sgd_TIME-20180306-225909'
    embed_dim = 20
    val_dataset = pascal_dataset.VOC2011ClassSeg(split='seg11valid', transform=True, embed_dim=embed_dim, one_hot_embed=True, data_dir=data_dir)
    embed_arr = utils.load_obj('datasets/pascal/embeddings/one_hot_21_dim') # inference candidate embeddings

    # initialize TENSORBOARD writer
    tb_path = osp.join(data_dir,'tb', name)
    tb_writer = SummaryWriter(tb_path)

    # 1. val DATASET
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    # 2. load MODEL 
    model_file = data_dir + '/logs' + model_name + '/best'
    model = models.FCN32s(n_class=embed_dim)
    if cuda:
        model = model.cuda()
    model_data = torch.load(model_file)

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    # 3. INFERENCE on val dataset
    embeddings = Variable(torch.from_numpy(embed_arr).cuda().float(), requires_grad=False)

    lbl_preds, lbl_trues = [], []
    for batch_idx, (data, target) in enumerate(val_loader):

        target, target_embed = target
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target_embed = Variable(target_embed.cuda())
        score = model(data)

        lbl_pred = utils.infer_lbl(score, embeddings, cuda)[0]
        lbl_true = target.data.cpu().numpy()[0]

        lbl_preds.append(lbl_pred)
        lbl_trues.append(lbl_true)

    n_class = len(val_loader.dataset.class_names)
    metrics = np.array(utils.label_accuracy_score(lbl_trues, lbl_preds, n_class=n_class)) * 100

    print('''\
Nearest Neighboring Inference
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

if __name__ == '__main__':
    main()
