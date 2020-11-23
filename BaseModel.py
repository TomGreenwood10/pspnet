"""
Base model class for segmentation models.
"""

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from os import listdir
from os.path import join
from skimage.transform import resize


class SegmentationModel:
    """
    Intended to be inherited by architecture specific model classes that add a 
    build() method and any other specifics.
    """

    def __init__(self):
        self.net = None
        self.trained = False
        self.input_shape = (480, 480, 3)
        self.loss = None
        self.loss_weights = [0.2, 0.8]
        self.optimizer = None
        self.metrics = []
        self.model_name = 'new_model'
        self.log_name = 'new_log'
        self.compiled = False
        self.epochs = 200
        self.batch_size = 1  # todo: check if different batch size should be 
                             #  used for training vs prediction
        self.validation_split = 0.1
        self.history = None
        self.callbacks = [ModelCheckpoint(filepath=self.model_name + '.h5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='min',
                                          period=1),
                          CSVLogger(filename=self.log_name,
                                    append=False)]

    def set_attr(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def compile(self, **kwargs):
        self.set_attr(kwargs)
        self.net.compile(
            loss=self.loss,
            loss_weights=self.loss_weights,
            optimizer=self.optimizer,
            metrics=self.metrics)
        self.compiled = True

    def fit(self, x_train, y_train, **kwargs):
        self.set_attr(kwargs)
        if not self.compiled:
            self.compile()
        self.history = self.net.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.callbacks)

    def predict(self, x):
        y_pred = self.net.predict(x).ravel()
        return y_pred

    def test(self, dir_test_images, dir_labeled_images):
        pass


def get_pool_size(input_shape):
    # Index from end of shape array to avoid errors if the batch size is put in
    #  first. Input shape will be either (rows, columns, channels) or 
    #  (batch_size, rows, columns, channels)
    if input_shape[-3] != input_shape[-2]:
        raise ValueError(
            "image need to be square (equal number of rows and columns)")
    pool_size = 2
    while input_shape[-3] % pool_size != 0:
        pool_size += 1
    return pool_size
