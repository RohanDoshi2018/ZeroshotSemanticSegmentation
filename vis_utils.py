import fcn
import numpy as np

def visualize_seenmask(**kwargs):
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    unseen = kwargs.pop('unseen', None)
    n_class = kwargs.pop('n_class', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    mask_unlabeled = None
    viz_unlabeled = None

    mask_unlabeled = (lbl_true == -1)
    lbl_true[mask_unlabeled] = 0
    lbl_pred[mask_unlabeled] = 0
    viz_unlabeled = (np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255).astype(np.uint8)

    viz_trues = [
        img,
        make_seen_mask(lbl_true, [0], 2),
        make_seen_mask(lbl_pred, [0], 2),
    ]
    
    for i in range(1, 3):
        viz_trues[i][mask_unlabeled] = viz_unlabeled[mask_unlabeled]

    return fcn.utils.get_tile_image(viz_trues, (1, 3))


def visualize_segmentation(**kwargs):
    """Visualize segmentation.
    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.
    Returns
    -------
    img_array: ndarray
        Visualized image.
    """

    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    unseen = kwargs.pop('unseen', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if unseen:
        n_col = 4
    else:
        n_col = 3
        
    if lbl_true is None or lbl_pred is None:
        raise ValueError('lbl_true and lbl_pred must be not None.')

    # get mask for borders and random values (viz_unlabeled) to fill mask areas
    mask_unlabeled = lbl_true == -1
    lbl_true[mask_unlabeled] = 0
    viz_unlabeled = (np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255).astype(np.uint8)
    # lbl_pred[mask_unlabeled] = 0

    vizs = []

    # lbl_true row
    viz_trues = [
        img,
        fcn.utils.label2rgb(lbl_true, label_names=label_names, n_labels=n_class),
        fcn.utils.label2rgb(lbl_true, img, label_names=label_names, n_labels=n_class),
    ]

    # lbl_pred row
    viz_preds = [
        img,
        fcn.utils.label2rgb(lbl_pred, label_names=label_names, n_labels=n_class),
        fcn.utils.label2rgb(lbl_pred, img, label_names=label_names, n_labels=n_class),
    ]

    # add extra seenmask column to each row
    if unseen:
        viz_trues.append(make_seen_mask(lbl_true, unseen, n_class))
        viz_preds.append(make_seen_mask(lbl_pred, unseen, n_class))

    # apply borders to images
    for i in range(1, n_col):
        viz_trues[i][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_preds[i][mask_unlabeled] = viz_unlabeled[mask_unlabeled]

    # add the two rows to final visualization
    vizs.append(fcn.utils.get_tile_image(viz_trues, (1, n_col)))
    vizs.append(fcn.utils.get_tile_image(viz_preds, (1, n_col)))

    return fcn.utils.get_tile_image(vizs, (2, 1))

def make_seen_mask(lbl, unseen, n_class):
    seen = [x for x in range(n_class) if x not in unseen]
    mask_seen = np.in1d(lbl.ravel(), seen).reshape(lbl.shape)
    mask_seen = (mask_seen * 255).astype(np.uint8)
    mask_seen = np.repeat(mask_seen[:, :, np.newaxis], 3, axis=2)
    return mask_seen