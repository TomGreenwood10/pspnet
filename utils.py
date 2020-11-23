# This file contains utility functions for computer vision defect detection

# Imports
import numpy as np
import json
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import random
import os


def get_label(image):
    """
    Takes an image (as an array) and returns the pixel labels (1 for HA, 0 for
    everything else).

    Note: This function has been manually iterated through whilst adjusting the
    filer criteria and the input image - it is not intended for production / 
    automated use.

    :param: image, numpy array.  The manually coloured image with red 
        (255, 0, 0) pixels as the HA. Note: Of other pixels are being picked up
        they can be colored in blue to remove them.
    
    :return: pixel labels, numpy array.  The same shape as the input image.
    """

    import numpy as np

    shape = image.shape[:2]
    categorised = np.zeros_like(image[:, :, 0]).ravel()
    for pixel, channels in enumerate(image.reshape(-1, 3)):
        if channels[0] > 254: # and channels[1:3].sum() == 0:
            categorised[pixel] = 1
        else:
            categorised[pixel] = 0

    return categorised.reshape(shape)


def get_labels_from_json(json_path,
                         train_images_path,
                         train_labels_path,
                         test_images_path=None,
                         test_labels_path=None,
                         train_test_split=0.1,
                         classes=(
                             'None',
                             'coated surface',
                             'tape',
                             'blast surface',
                             'polished surface'
                         ),
                         verbose=1):
    """
    Extracts labels from LabelBox json export and saves them in specified 
    directories.

    :param json_path: string -- File path to json file as exported from 
        LabelBox.com.
    
    :param train_images_path: string -- The file path to save the training raw
        images to.
    
    :param train_labels_path: string -- the file path to save the training 
        labels to.
    
    :param test_images_path: string, optional -- The path to save the test 
        images to.  If None then all images will be saved to training paths.
        Note; if a path is entered then a test label path bust also be 
        specified.

    :param test_labels_path: string, optional -- The path to save the test
        labels to.  If None then all images will be saved to training paths.
        Note; if a path is entered then a test image path bust also be 
        specified.
    
    :param train_test_split: float (between 0 and 1 inclusive), optional -- The
        proportion of the images to be saves in the testing paths if specified.

    :param classes, optional -- List of classes contained. Note the numerical
        class returned for each pixel will be the index of it's title in this
        list.

    :param verbose: int, optional -- If 1 then info on images saved will be 
        printed.

    :return: None -- Images are saved in the specified locations.
    """

    # Check for correct parameter entry
    if test_images_path is not None:
        if test_labels_path is None:
            raise AttributeError(
        'If test_images_path is given, test_labels_path must also be given')

    if test_labels_path is not None:
        if test_images_path is None:
            raise AttributeError(
        'If test_labels_path is given, test_images_path must also be given')

    # Extract contend from json file
    with open(json_path) as f:
        j = json.load(f)

    # Constants
    n_labels = len(j)

    # Instantiate list to collect images and labels
    images = []
    labels = []

    # Loop through labels
    for label in range(n_labels):
        while True:
            try:
                # Get original image
                img = io.imread(j[label]['Labeled Data'])

                # Fund number of objects and store object in a list for leter 
                #  iteration
                n_objs = len(j[label]['Label']['objects'])
                obj_list = j[label]['Label']['objects']

                # Instantiate a return array of zeros the same size as the 
                #  first object
                combined_label = np.zeros_like(
                    io.imread(obj_list[0]['instanceURI'])[:, :, 0]
                )

                # Loop through objects
                for obj_dict in obj_list:

                    # Collect the class and the object image
                    cls = classes.index(obj_dict['title'])
                    lab = io.imread(obj_dict['instanceURI'])
                    lab = lab[:, :, 0]

                    # Loop through pixels where object is identified; alter 
                    #  rtn_label to indicate class
                    for row in range(lab.shape[0]):
                        for col in range(lab.shape[1]):
                            if lab[row, col] == 255:
                                combined_label[row, col] = cls

                # Append image and label to list
                images.append(img)
                labels.append(combined_label)
            except:
                print('Error encountered.  Trying again...')
                continue
            break

    # Split lists and save images
    n_test_samples = int(round(len(images) * train_test_split, 0))
    train_idxs = list(range(len(images)))
    test_idxs = random.sample(train_idxs, n_test_samples)
    train_idxs = [idx for idx in train_idxs if idx not in test_idxs]
    for idx in train_idxs:
        io.imsave(
            os.path.join(train_images_path, '{}.png'.format(idx)),
            images[idx]
        )
        io.imsave(
            os.path.join(train_labels_path, '{}.png'.format(idx)),
            labels[idx]
        )
    for idx in test_idxs:
        io.imsave(
            os.path.join(test_images_path, '{}.png'.format(idx)),
            images[idx]
        )
        io.imsave(
            path=os.path.join(test_labels_path, '{}.png'.format(idx)),
            labels[idx]
        )

    # Inform user
    if verbose == 1:
        print('{} image-label pairs total'.format(len(images)))
        print('{} pairs saved in training paths'.format(len(train_idxs)))
        print('{} images saved in testing paths'.format(len(test_idxs)))


def training_graphs(log_path,
                    metrics=['loss', 'acc', 'f1_score'],
                    val=True,
                    val_focus=True,
                    colours=['red', 'blue', 'green'],
                    size=(30, 5),
                    zoomed=False):
    """
    Draws summary training graphs from model training log.

    :param log_path:
    :param metrics:
    :param val:
    :param val_focus:

    :return:
    """

    # Load dataframe
    if type(log_path) is str:
        df = pd.read_csv(log_path)
    elif type(log_path) is pd.core.frame.DataFrame:
        df = log_path
    else:
        raise TypeError('The log type is not supported')

    # Collect number of metrics
    n_metrics = len(metrics)

    # Plots
    plt.figure(figsize=size)
    for ii, metric in enumerate(metrics):
        plt.subplot(1, n_metrics, ii + 1)
        plt.plot(df[metric], ls='--', c=colours[ii])
        plt.plot(df['val_' + metric], c=colours[ii])
        plt.grid()
        plt.title(metric)
        plt.xlabel('epoch')

        # Zoom y-axis if required
        if zoomed is True:
            if metric in ['acc', 'f1_score']:
                plt.ylim(0.9, 1.01)
            elif metric is 'loss':
                plt.ylim(-0.01, 0.5)
    plt.show()


def display_result(model_path, X_test, y_test, normalise=True, size=(20, 10)):
    """
    Displays prediction alongside labels and original image.

    :param model_path: String -- The path to the .h5 model

    :param X_test: array - the prepared original image - must be (480, 480, 3) 
        dimensions.

    :param y_test: array - the preparred labels - must be (480, 480, 1)
    
    :param normalise: Bool, optional - True wil normalise X_test. Set to false
        if this has already been done.

    :param size: tuple, optional - the size of the figure (all 3 images) to be 
        displayed.

    :return: None -- displays image.
    """

    from tensorflow.keras.models import load_model
    import model

    # Load model
    cnn = load_model(model_path, custom_objects={'f1_score': model.f1_score})

    # Reshape y_test
    y_test = y_test.reshape(1, 480, 480, 1)

    # Normalise X_test
    if normalise is True:
        X_test_orig = X_test.copy()
        X_test = X_test - np.mean(X_test)
        X_test = X_test / np.std(X_test)

    # Get predictions
    y_pred = cnn.predict(X_test.reshape(1, 480, 480, 3))

    # Convert predictions into single channel
    y_pred_sc = np.zeros((480, 480))
    for row in range(480):
        for col in range(480):
            y_pred_sc[row, col] = y_pred[0, row, col, :].argmax()

    # Display predictions
    plt.figure(figsize=size)
    plt.subplot(131)
    if normalise is True:
        io.imshow(X_test_orig)
    else:
        io.imshow(X_test)
    plt.subplot(132)
    plt.imshow(np.squeeze(y_test))
    plt.subplot(133)
    plt.imshow(y_pred_sc)
    plt.show()


def training_bands(log_dir,
                   log_format,
                   factor,
                   factor_vals,
                   iterations,
                   shape,
                   fsize=(20, 8),
                   colours=['red', 'blue', 'green', 'orange', 'purple'],
                   x_lim=None,
                   y_lim=None):

    plt.figure(figsize=fsize)

    for ii, fac in enumerate(factor_vals):
        plt.subplot(shape[0], shape[1], ii+1)

        df = pd.DataFrame([])
        model_itr = 0
        for itr in iterations:
            for log_file in os.listdir(log_dir):
                if log_file == log_format.format(fac, itr):
                    df_temp = pd.read_csv(os.path.join(log_dir, log_file))
                    df_temp['model'] = model_itr
                    model_itr += 1
                    df = pd.concat([df, df_temp])

        # Instantiate lists
        x = np.arange(len(df_temp))
        ymin = []
        ymax = []

        # Collect min and max scores for each epoch
        for ep in range(1000):
            df_temp = df[df.epoch == ep]
            ymin.append(df_temp.acc.min())
            ymax.append(df_temp.acc.max())

        # Display
        plt.fill_between(x, ymin, ymax, alpha=0.2, facecolor=colours[ii])
        plt.plot(x, ymin, c=colours[ii], alpha=1)
        plt.plot(x, ymax, c=colours[ii], alpha=1)
        for mod in range(10):
            plt.plot(
                df[df.model == mod].epoch,
                df[df.model == mod].acc,
                c=colours[ii],
                alpha=0.3
            )
        if x_lim != None:
            plt.xlim(x_lim[0], x_lim[1])
        if y_lim != None:
            plt.ylim(y_lim[0], y_lim[1])
        plt.grid()
        plt.title('{} = {}'.format(factor, fac))
