"""
Image preprocessing functions.
"""

import os
import random
import numpy as np
import imgaug.augmenters as iaa
from skimage import io, transform
from tensorflow.keras.utils import to_categorical


def labeled_data(images_path,
                 labels_path,
                 resize=False,
                 n_augments=None,
                 **kwargs):
    """
    Pre process images and corresponding labels into training arrays.  Add 
    chosen augmentations.

    If an integer is assigned to the the n_augments argument then that number 
    of augmentations will randomly be chosen.  If n_augments is None then the
    augmentations can be chosen manually from the following list by passing the
    name of the augmentations as a key word argument and assigning it's value 
    to True.

    Available augmentations
    =======================

    - scale
    - translate
    - rotate
    - flip
    - blur
    - noise
    - gamma
    - sigmoid

    Parameters
    ==========

    :images_path:  string -- The path to the unprocessed images
    
    :labels_path:  string -- The path to the corresponding labels.  Note: the 
        filename of the label must match it's corresponding image.
    
    :n_augments:  int -- The number of augmentations to be randomly chosen 
        (max 8).
        NOTE: If this is used then no augmentations can be manually chosen.
        NOTE: Each augmentation method may produce several new images that are
        variations - be ware of choosing high numbers as all variations of all
        augmentations will be applied.
    
    :kwargs:  See "Available augmentations" above.

    Return
    ======

    :tuple (X_train array, y_train array) where X_train is the array of model
    inputs and y_train the targets.

    """
    x_train = []
    y_train = []

    for filename in os.listdir(images_path):
        image_filepath = os.path.join(images_path, filename)
        label_filepath = os.path.join(labels_path, filename)
        image = io.imread(image_filepath)
        label = io.imread(label_filepath)
        # practice labels are 0 - 255 so need to be converted:
        # label = label / 255
        if resize:
            image = transform.resize(image, resize, order=1)
            label = transform.resize(
                label,
                resize,
                order=0,
                preserve_range=True,
                anti_aliasing=False).astype(int)
        if n_augments:
            image_augs, label_augs = augment_random(image, label, n_augments)
        else:
            image_augs, label_augs = augment_all(image, label, **kwargs)
        x_train += image_augs
        y_train += label_augs

    image_shape = x_train[0].shape
    x_train = np.concatenate(x_train, axis=0)
    x_train = x_train.reshape(-1, image_shape[0], image_shape[1], 3)
    y_train = np.concatenate(y_train, axis=0)
    y_train = y_train.reshape(-1, image_shape[0], image_shape[1], 1)

    y_train = to_categorical(y_train)

    return (x_train, y_train)


def augment_all(image, 
                label,
                scale=False,
                translate=False,
                rotate=False,
                flip=False,
                blur=False,
                noise=False,
                gamma=False,
                sigmoid=False):
    """
    Take an image, label pair and return augmented image list and augmented 
    label list pair. This will include all augmentations set to True.

    WARNING: If all augmentations are set to true the list sizes will be very
    large.
    
    Note: original image/label will be in list.
    """
    image_augs = [image]
    label_augs = [label]

    if scale:
        aug_scale(image_augs, label_augs)
    if translate:
        aug_translate(image_augs, label_augs)
    if rotate:
        aug_rotate(image_augs, label_augs)
    if flip:
        aug_flip(image_augs, label_augs)
    if blur:
        aug_blur(image_augs, label_augs)
    if noise:
        aug_noise(image_augs, label_augs)
    if gamma:
        aug_gamma(image_augs, label_augs)
    if sigmoid:
        aug_sigmoid(image_augs, label_augs)
    
    return (image_augs, label_augs)

    
def augment_random(image, label, n_augments):
    """
    Take an image, label pair and return augmented image list and augmented 
    label list pair. n_augments will be chosen randomly.
    
    Note: original image/label will be in list. 
    """
    image_augs = [image]
    label_augs = [label]
    
    augmenters = [
        aug_scale,
        aug_translate,
        aug_rotate,
        aug_flip,
        aug_blur,
        aug_noise,
        aug_gamma,
        aug_sigmoid
    ]

    for augmenter in random.sample(augmenters, n_augments):
        augmenter(image_augs, label_augs)
    
    return (image_augs, label_augs)


def aug_scale(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.geometric.Affine(scale=0.7).augment_image(image_augs[i])
        augmented_image_2 = iaa.geometric.Affine(scale=0.9).augment_image(image_augs[i])
        augmented_image_3 = iaa.geometric.Affine(scale=1.1).augment_image(image_augs[i])
        augmented_image_4 = iaa.geometric.Affine(scale=1.3).augment_image(image_augs[i])
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        image_augs.append(augmented_image_3)
        image_augs.append(augmented_image_4)
        augmented_label_1 = iaa.geometric.Affine(scale=0.7, order=0).augment_image(label_augs[i])
        augmented_label_2 = iaa.geometric.Affine(scale=0.9, order=0).augment_image(label_augs[i])
        augmented_label_3 = iaa.geometric.Affine(scale=1.1, order=0).augment_image(label_augs[i])
        augmented_label_4 = iaa.geometric.Affine(scale=1.3, order=0).augment_image(label_augs[i])
        label_augs.append(augmented_label_1)
        label_augs.append(augmented_label_2)
        label_augs.append(augmented_label_3)
        label_augs.append(augmented_label_4)


def aug_translate(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.geometric.TranslateX(0.05).augment_image(image_augs[i])
        augmented_image_2 = iaa.geometric.TranslateX(-0.05).augment_image(image_augs[i])
        augmented_image_3 = iaa.geometric.TranslateY(0.05).augment_image(image_augs[i])
        augmented_image_4 = iaa.geometric.TranslateY(-0.05).augment_image(image_augs[i])
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        image_augs.append(augmented_image_3)
        image_augs.append(augmented_image_4)
        augmented_label_1 = iaa.geometric.TranslateX(0.05).augment_image(label_augs[i])
        augmented_label_2 = iaa.geometric.TranslateX(-0.05).augment_image(label_augs[i])
        augmented_label_3 = iaa.geometric.TranslateY(0.05).augment_image(label_augs[i])
        augmented_label_4 = iaa.geometric.TranslateY(-0.05).augment_image(label_augs[i])
        label_augs.append(augmented_label_1)
        label_augs.append(augmented_label_2)
        label_augs.append(augmented_label_3)
        label_augs.append(augmented_label_4)


def aug_rotate(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.geometric.Rotate(-30).augment_image(image_augs[i])
        augmented_image_2 = iaa.geometric.Rotate(-5).augment_image(image_augs[i])
        augmented_image_3 = iaa.geometric.Rotate(5).augment_image(image_augs[i])
        augmented_image_4 = iaa.geometric.Rotate(30).augment_image(image_augs[i])    
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        image_augs.append(augmented_image_3)
        image_augs.append(augmented_image_4)
        augmented_label_1 = iaa.geometric.Rotate(-30, order=0).augment_image(label_augs[i])
        augmented_label_2 = iaa.geometric.Rotate(-5, order=0).augment_image(label_augs[i])
        augmented_label_3 = iaa.geometric.Rotate(5, order=0).augment_image(label_augs[i])
        augmented_label_4 = iaa.geometric.Rotate(30, order=0).augment_image(label_augs[i])
        label_augs.append(augmented_label_1)
        label_augs.append(augmented_label_2)
        label_augs.append(augmented_label_3)
        label_augs.append(augmented_label_4)


def aug_flip(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.Fliplr(p=1).augment_image(image_augs[i])
        augmented_image_2 = iaa.Flipud(p=1).augment_image(image_augs[i])
        augmented_image_3 = iaa.Fliplr(p=1).augment_image(augmented_image_2)
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        image_augs.append(augmented_image_3)
        augmented_label_1 = iaa.Fliplr(p=1).augment_image(label_augs[i])
        augmented_label_2 = iaa.Flipud(p=1).augment_image(label_augs[i])
        augmented_label_3 = iaa.Fliplr(p=1).augment_image(augmented_label_2)
        label_augs.append(augmented_label_1)
        label_augs.append(augmented_label_2)
        label_augs.append(augmented_label_3)


def aug_blur(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.blur.AverageBlur(3).augment_image(image_augs[i])
        augmented_image_2 = iaa.blur.AverageBlur(10).augment_image(image_augs[i])
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        # Labels are not blurred
        label_augs.append(label_augs[i])
        label_augs.append(label_augs[i])


def aug_noise(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.imgcorruptlike.SpeckleNoise(1, seed=1).augment_image(image_augs[i])
        augmented_image_2 = iaa.imgcorruptlike.SpeckleNoise(5, seed=1).augment_image(image_augs[i])
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        # Labels are not noised
        label_augs.append(label_augs[i])
        label_augs.append(label_augs[i])


def aug_gamma(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.contrast.GammaContrast(0.5).augment_image(image_augs[i])
        augmented_image_2 = iaa.contrast.GammaContrast(2).augment_image(image_augs[i])
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        # Labels are not altered
        label_augs.append(label_augs[i])
        label_augs.append(label_augs[i])


def aug_sigmoid(image_augs, label_augs):
    size = len(image_augs)
    for i in range(size):
        augmented_image_1 = iaa.contrast.SigmoidContrast(3).augment_image(image_augs[i])
        augmented_image_2 = iaa.contrast.SigmoidContrast(7).augment_image(image_augs[i])
        image_augs.append(augmented_image_1)
        image_augs.append(augmented_image_2)
        # Labels are not altered
        label_augs.append(label_augs[i])
        label_augs.append(label_augs[i])

