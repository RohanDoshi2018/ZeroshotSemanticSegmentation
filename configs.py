configurations = {

    ## PASCAL

    # fcn baseline: no embeddings, softmax output (mse optimizer)
    1: dict(
        mode='train',
        dataset='pascal',
        unseen=None,
        embed_dim=0,
        max_epoch=50,
        lr=1e-10,
        loss_func=None, # not used
        optimizer='sgd',
        momentum=0.99,
        weight_decay=0.0005,
        one_hot_embed=False,
        fixed_vgg=False,
    ),

    # fcn baseline, but with adam optimizer; 1 vs 2 conclusion: use adam optimizer 
    2: dict(
        mode='train',
        dataset='pascal',
        unseen=None,
        embed_dim=0,
        max_epoch=50,
        lr=1e-5,
        momentum=None,
        loss_func=None,
        optimizer='adam',
        weight_decay=0,
        one_hot_embed=False,
        fixed_vgg=False,
    ),


    # 21D one-hot embeddings, mse loss
    3: dict(
        mode='train',
        dataset='pascal',
        unseen=None,
        embed_dim=21,        
        max_epoch=50,
        lr=1e-5,
        loss_func='mse',
        optimizer='adam',
        momentum=None,
        weight_decay=0,
        one_hot_embed=True,
        fixed_vgg=False,
    ),

    # 21D one-hot, cosine loss; 3 vs 4 conclusion: use cosine loss
    4: dict(
        mode='train',
        dataset='pascal',
        unseen=None,
        embed_dim=21,
        max_epoch=50,
        lr=1e-5,
        loss_func='cos',
        optimizer='adam',
        momentum=None,
        weight_decay=0,
        one_hot_embed=True,
        fixed_vgg=False,
    ),

    # 20D; mse vs cos args conclusion: use cosine loss
    5: dict(
        mode='train',
        dataset='pascal',
        unseen=None,
        embed_dim=20,
        max_epoch=50, # 8498 training images
        lr=1e-5,
        loss_func='cos',
        optimizer='adam',
        momentum=None,
        weight_decay=None,
        one_hot_embed=False,
        fixed_vgg=False,
    ),

    # 20D zeroshot (unseen: 10 classes)
    6: dict(
        mode='train',
        dataset='pascal',
        unseen=[6, 7, 13, 14, 15, 16, 17, 18, 19, 20],
        embed_dim=20,
        max_epoch=130, # 3311 seen and 5187 unseen training images
        lr=1e-5,
        loss_func='cos',
        optimizer='adam',
        momentum=None,
        weight_decay=None,
        one_hot_embed=False,
        fixed_vgg=False,
    ),

    ## CONTEXT

    # 20D context
    7: dict(
        mode='train',
        dataset='context',
        unseen=None,
        embed_dim=20,        
        max_epoch=50,
        lr=1e-5,
        loss_func='cos',
        optimizer='adam',
        momentum=None,
        weight_decay=None,
        one_hot_embed=False,
        fixed_vgg=False,
    ),

    # 20D context zeroshot (unseen: 16 classes)
    8: dict(
        mode='train',
        dataset='context',
        unseen=[1,13,16,27,32],
        embed_dim=20,
        max_epoch=200, # should be roughly 152
        lr=1e-5,
        loss_func='cos',
        optimizer='adam',
        momentum=None,
        weight_decay=None,
        one_hot_embed=False,
        fixed_vgg=False,
    ),

    # EXPERIMENTAL

    # testing: 20D 10/10 zeroshot (cfg6) with/without forced unseen inference
    9: dict(
        mode='test',
        dataset='pascal',
        unseen=[6, 7, 13, 14, 15, 16, 17, 18, 19, 20],
        embed_dim=20,
        max_epoch=200,
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        loss_func='cos',
        optimizer='adam',
        one_hot_embed=False,
        fixed_vgg=False,
        forced_unseen=True, # experimental feature
        resume_model_path='20D_pascal_zeroshot_CFG_6_MAX_EPOCH_130_LR_1e-05_MOMENTUM_None_WEIGHT_DECAY_None_EMBED_DIM_20_ONE_HOT_EMBED_False_LOSS_FUNC_cos_BK_LOSS_True_UNSEEN_True_DATASET_pascal_OPTIMIZER_adam_TIME-20180409-130600',
    ),

    # 20D 18/2 zeroshot
    10: dict(
        mode='train',
        dataset='pascal',
        unseen=[1, 17],
        embed_dim=20,
        max_epoch=200,
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        loss_func='cos',
        optimizer='adam',
        one_hot_embed=False,
        fixed_vgg=False,
    ),


    # testing: 20D 18/2 zeroshot
    13: dict(
        mode='test',
        dataset='pascal',
        unseen=[1, 17],
        embed_dim=20,
        max_epoch=200,
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        loss_func='cos',
        optimizer='adam',
        one_hot_embed=False,
        fixed_vgg=False,
        forced_unseen=True, # experimental feature
        resume_model_path='20D_pascal_zeroshot_18_2_CFG_10_MODE_train_DATASET_pascal_UNSEEN_True_EMBED_DIM_20_MAX_EPOCH_200_LR_1e-05_MOMENTUM_None_WEIGHT_DECAY_None_LOSS_FUNC_cos_OPTIMIZER_adam_ONE_HOT_EMBED_False_TIME-20180410-155755',
    ),

    # 20D 18/2 zeroshot with fixed vgg
    11: dict(
        mode='train',
        dataset='pascal',
        unseen=[1, 17],
        embed_dim=20,
        max_epoch=200,
        lr=1e-3,
        momentum=None,
        weight_decay=None,
        loss_func='cos',
        optimizer='adam',
        one_hot_embed=False,
        fixed_vgg=True, 
    ),

    # testing: 20D 18/2 zeroshot with fixed vgg(cfg11) with/without forced unseen inference
    12: dict(
        mode='test',
        dataset='pascal',
        unseen=[1, 17],
        embed_dim=20,
        max_epoch=200,
        lr=1e-5,
        momentum=None,
        weight_decay=None,
        loss_func='cos',
        optimizer='adam',
        one_hot_embed=False,
        fixed_vgg=True,
        forced_unseen=True, # experimental feature
        resume_model_path='20D_pascal_zeroshot_18_2_fixed_vgg_CFG_11_MODE_train_DATASET_pascal_UNSEEN_True_EMBED_DIM_20_MAX_EPOCH_200_LR_0.001_MOMENTUM_None_WEIGHT_DECAY_None_LOSS_FUNC_cos_OPTIMIZER_adam_ONE_HOT_EMBED_False_FIXED_VGG_True_TIME-20180412-143324',
    ),

    # # 20D 18/2 pascal zeroshot with seen/unseen classifier (aka train_unseen)
    # 14: dict(
    #     mode='train',
    #     dataset='pascal',
    #     unseen=[1, 17],
    #     train_unseen=[2,16],
    #     embed_dim=20,
    #     max_epoch=200,
    #     lr=1e-5,
    #     momentum=None,
    #     weight_decay=None,
    #     loss_func='cos',
    #     optimizer='adam',
    #     one_hot_embed=False,
    #     fixed_vgg=False,
    # ),
}