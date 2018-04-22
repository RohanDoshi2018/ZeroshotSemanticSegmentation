configurations = {

    # fcn baseline with softmax inference
    1: dict(
        mode='train',
        dataset='pascal',
        train_unseen=[],
        val_unseen=[],
        embed_dim=0,
        fcn_epochs=30,   
        fcn_lr=1e-10,
        fcn_loss='cross_entropy',
        fcn_optim='sgd',
        seenmask_epochs=0, 
        seenmask_lr=1e-3,
    ),

    # one-hot
    2: dict(
        mode='train',
        dataset='pascal',
        train_unseen=[],
        val_unseen=[],
        embed_dim=21,        
        fcn_epochs=30,
        fcn_lr=1e-5, # finetune
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=0,    
        seenmask_lr=1e-3,
        one_hot_embed=False,
    ),

    # 20D pascal
    4: dict(
        mode='train',
        dataset='pascal',
        train_unseen=[],
        val_unseen=[],
        embed_dim=20,
        fcn_epochs=30, # 8498 training images
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=0,  
        seenmask_lr=1e-3,
    ),

    # train seenmask: 20D 8/2/10 pascal zeroshot with seenmask
    14: dict(
        mode='train',
        dataset='pascal',
        train_unseen=[1, 13],
        val_unseen=[6, 7, 10, 14, 15, 16, 17, 18, 19, 20],
        embed_dim=20,
        fcn_epochs=90,
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=10,
        seenmask_lr=1e-3,
    ),

    # test: 20D 8/2/10 pascal zeroshot with seenmask
    15: dict(
        mode='test_all',
        dataset='pascal',
        train_unseen=[1, 13],
        val_unseen=[6, 7, 10, 14, 15, 16, 17, 18, 19, 20],
        embed_dim=20,
        fcn_epochs=0,
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=0, 
        seenmask_lr=1e-3,
        load_fcn_path="8_2_10_CFG_14_MODE_train_DATASET_pascal_TRAIN_UNSEEN_True_VAL_UNSEEN_True_EMBED_DIM_20_FCN_EPOCHS_90_FCN_LR_1e-05_FCN_LOSS_cos_FCN_OPTIM_adam_SEENMASK_EPOCHS_10_SEENMASK_LR_0.001_TIME_20180421-163751_",
    ),


    # train: 20D 16/2/2 pascal zeroshot with seenmask
    16: dict(
        mode='train',
        dataset='pascal',
        train_unseen=[1,13],
        val_unseen=[17, 19],
        embed_dim=20,
        fcn_epochs=36,
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=10,
        seenmask_lr=1e-3,
    ),

    # test: 20D 16/2/2 pascal zeroshot with seenmask
    17: dict(
        mode='test_all',
        dataset='pascal',
        train_unseen=[1, 13],
        val_unseen=[17, 19],
        embed_dim=20,
        fcn_epochs=0,
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=0, 
        seenmask_lr=1e-3,
        forced_unseen=False,
        load_fcn_path="16_2_2_CFG_16_MODE_train_DATASET_pascal_TRAIN_UNSEEN_True_VAL_UNSEEN_True_EMBED_DIM_20_FCN_EPOCHS_36_FCN_LR_1e-05_FCN_LOSS_cos_FCN_OPTIM_adam_SEENMASK_EPOCHS_10_SEENMASK_LR_0.001_TIME_20180421-163803_", # TODO: path from cfg 16
    ),

    # train: 20D 31/2/2 context zeroshot with seenmask
    18: dict(
        mode='train',
        dataset='context',
        train_unseen=[0,12],
        val_unseen=[16, 18],
        embed_dim=20,
        fcn_epochs=59,  # TODO: fix this
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=10,
        seenmask_lr=1e-3,
    ),

    # test: 20D 31/2/2 context zeroshot with seenmask
    19: dict(
        mode='test_all',
        dataset='context',
        train_unseen=[0, 12],
        val_unseen=[16, 18],
        embed_dim=20,
        fcn_epochs=0,
        fcn_lr=1e-5,
        fcn_loss='cos',
        fcn_optim='adam',
        seenmask_epochs=0, 
        seenmask_lr=1e-3,
        load_fcn_path="", # TODO: path from cfg 18
    ),

}