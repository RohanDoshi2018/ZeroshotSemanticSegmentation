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
import dataset, models, utils 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    root = '/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/logs/'
    model_file = root + '_MAX_EPOCH_15_LR_1e-06_MOMENTUM_0.99_WEIGHT_DECAY_0.0005_EMBED_DIM_21_ONE_HOT_EMBED_True_TRAIN_LOSS_FUNC_mse_BACKGROUND_LOSS_False_ZEROSHOT_False_TIME-20180227-220242/best'
    # model_file = root + '_MAX_EPOCH_15_LR_1e-06_MOMENTUM_0.99_WEIGHT_DECAY_0.0005_EMBED_DIM_21_ONE_HOT_EMBED_True_TRAIN_LOSS_FUNC_mse_BACKGROUND_LOSS_True_ZEROSHOT_False_TIME-20180228-082313/best'
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    val_loader = torch.utils.data.DataLoader(
        dataset.VOC2011ClassSeg(split='seg11valid', transform=True, embed_dim=21, one_hot_embed=True),
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


    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in enumerate(val_loader):

        target, target_embed = target
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target_embed = Variable(target_embed.cuda())
        score = model(data)

        imgs = data.data.cpu()
        


        # lbl_pred = utils.get_lbl_pred(score, embeddings, cuda)
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_loader.dataset.class_names)
                visualizations.append(viz)
    metrics = utils.label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = fcn.utils.get_tile_image(visualizations)
    skimage.io.imsave('/opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/logs/viz_evaluate.png', viz)

if __name__ == '__main__':
    main()
