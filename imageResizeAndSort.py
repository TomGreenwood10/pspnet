"""
Resizes images from the semantic drone dataset (from kaggle) and puts them into
training and testing directories.

Can be easily adapted to other datasets but pay attention to the image 
directory structure.
"""

import os
import random
from multiprocessing import Pool
from skimage import io
from skimage import transform


# Original (large) images
ORIG_IMAGES_DIR = "images/semantic_drone_dataset/original_images"
ORIG_LABELS_DIR = "images/semantic_drone_dataset/label_images_semantic"

# New (small) images
TRAIN_TEST_SPLIT = 0.8
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
IMAGE_DIR = "images"
LABELS_DIR = "labels"


def process_image_label_pair(filename, train):
    image, label = load_image_label_pair(filename)
    image, label = resize_image_and_label(image, label)
    save_in_train_and_test_directories(image, label, filename, train)

    
def load_image_label_pair(filename):
    # Assumes image and label have same file name but with different extensions
    # as per the semantic drone dataset (from kaggel)
    filename = filename.strip(".jpg")
    full_image_path = os.path.join(ORIG_IMAGES_DIR, filename + ".jpg")
    full_label_path = os.path.join(ORIG_LABELS_DIR, filename + ".png")

    image = io.imread(full_image_path)
    label = io.imread(full_label_path)

    return image, label


def choose_training_image_indexes():
    random.seed(123)
    n_images = len(os.listdir(ORIG_IMAGES_DIR))
    train_idxs = random.sample(
        range(n_images), k=int(n_images * TRAIN_TEST_SPLIT))

    return train_idxs


def resize_image_and_label(image, label, size=400):
    # NOTE: dont import resize directly from skimage.transform as this will
    # clash with this function
    image = transform.resize(image=image, output_shape=(size, size, 3))
    label = transform.resize(image=label, output_shape=(size, size), 
        preserve_range=False, anti_aliasing=False, order=0)

    return image, label


def save_in_train_and_test_directories(image, label, filename, train):
    filename = filename.strip(".jpg")
    if train:
        image_save_filepath = os.path.join(
            TRAIN_DIR, IMAGE_DIR, filename + ".png")
        label_save_filepath = os.path.join(
            TRAIN_DIR, LABELS_DIR, filename + ".png")
    else:
        image_save_filepath = os.path.join(
            TEST_DIR, IMAGE_DIR, filename + ".png")
        label_save_filepath = os.path.join(
            TEST_DIR, LABELS_DIR, filename + ".png")

    io.imsave(image_save_filepath, image)
    io.imsave(label_save_filepath, label)


if __name__ == "__main__":
    
    train_idxs = choose_training_image_indexes()
    all_file_names = os.listdir(ORIG_IMAGES_DIR)

    args = []
    for idx, filename in enumerate(os.listdir(ORIG_IMAGES_DIR)):
        train = True if idx in train_idxs else False
        args.append((filename, train))

    with Pool() as pool:
        pool.starmap(process_image_label_pair, args)

