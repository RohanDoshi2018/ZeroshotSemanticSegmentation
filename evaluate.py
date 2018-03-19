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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    data_dir = '/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation'
    model_name = '/baseline_sgd_CFG_1_MAX_EPOCH_50_LR_1e-10_MOMENTUM_0.99_WEIGHT_DECAY_0.0005_EMBED_DIM_0_ONE_HOT_EMBED_False_TRAIN_LOSS_FUNC_None_BACKGROUND_LOSS_True_ZEROSHOT_False_DATASET_pascal_OPTIMIZER_sgd_TIME-20180306-225909'
    suffix = '/best'
    model_file = data_dir + '/logs' + model_name + suffix

    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    val_loader = torch.utils.data.DataLoader(
        pascal_dataset.VOC2011ClassSeg(split='seg11valid', transform=True, embed_dim=21, one_hot_embed=True, data_dir=data_dir),
        batch_size=1, shuffle=False, **kwargs)

    n_class = len(val_loader.dataset.class_names)
    model = models.FCN32s(n_class=21)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model_data = torch.load(model_file)

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    # one hot embedings
    embed_arr = utils.load_obj('embeddings/one_hot_21_dim')
    embeddings = Variable(torch.from_numpy(embed_arr).cuda().float(), requires_grad=False)

    label_nni_preds, label_argmax_preds, label_trues = [], [], []
    for batch_idx, (data, target) in enumerate(val_loader):

        target, target_embed = target
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target_embed = Variable(target_embed.cuda())
        score = model(data)

        lbl_pred_nni = utils.infer_lbl(score, embeddings, cuda)[0]
        lbl_pred_argmax = score.data.max(1)[1].cpu().numpy()[:, :, :][0]
        lbl_true = target.data.cpu().numpy()[0]

        label_nni_preds.append(lbl_pred_nni)
        label_argmax_preds.append(lbl_pred_argmax)
        label_trues.append(lbl_true)

    metrics_nni = np.array(utils.label_accuracy_score(label_trues, label_nni_preds, n_class=n_class)) * 100
    metrics_argmax = np.array(utils.label_accuracy_score(label_trues, label_argmax_preds, n_class=n_class)) * 100

    print('''\
Nearest Neighboring Inference
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics_nni))

    print('''\
Argmax Inference
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics_argmax))

if __name__ == '__main__':
    main()
