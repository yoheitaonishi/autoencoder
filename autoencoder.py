import argparse
import glob
import logging
import os

import numpy as np
import cv2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import array_to_img
from keras.utils import plot_model
from PIL import Image

INPUT_IMAGE_SIZE = 128

def setup_logger():
    """
    Set up logger
    """
    logger = logging.getLogger("RUN INFORMATION")
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/log.log')
    return logger

def setup_argument_parser():
    """
    Set argument
    """
    parser = argparse.ArgumentParser() 
    parser.add_argument('--source-dir', help='source image directory', default='./work/images', required=True)
    parser.add_argument('--decode-dir', help='saving decoded image directory', default='./work/decode', required=True)
    parser.add_argument('--resize-dir', help='saving resized image directory', default = './work/resize', required=True)
    parser.add_argument('--batch-size', help='batch size', type=int, default=16)
    parser.add_argument('--epoch', help='epoch', type=int, default=50)
    parser.add_argument('--trained-weight', help='saving trained h5 file directory', default='./work/trained.5h')
    parser.add_argument('--log-dir', help='tensor board log directory')
    return parser

def get_image_list(source_dir):
    """
    Get image path
    """
    image_list = glob.glob(source_dir + '/*.jpg')
    return image_list


def load_image(image_list, resize_dir, logger=None):
    """
    Load image and convert packed numpy array
    """
    image_data_array = []

    if logger:
        logger.info('Loading Image...')

    for image_path in image_list:
        image_data = cv2.imread(image_path)
        image_data = cv2.resize(image_data, dsize=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))

        if not os.path.exists(resize_dir):
            os.makedirs(resize_dir)

        basename = os.path.basename(image_path)
        basename = basename.replace('jpg', 'png')
        resize_image_save_file = os.path.join(resize_dir, basename)
        cv2.imwrite(resize_image_save_file, image_data)

        image_data = np.asarray(image_data)
        image_data_array.append(image_data)

    image_data_array = np.array(image_data_array)
    image_data_array = image_data_array.astype('float32') / 255.
    image_data_array = np.reshape(image_data_array, (len(image_data_array), INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)) 

    if logger:
        logger.info('Finish Loading Image!')

    return image_data_array

def build_model():
    """
    Building model
    """
    input_img = Input(shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)) 
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (16, 16, 16) i.e. 4096-dimensional

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

def set_optimizer(model):
    """
    Setting optimizer and loss function
    """
    model.compile(optimizer='AdaDelta', loss='binary_crossentropy')

def train_autoencoder(model, image_data_array, batch_size, epoch, trained_weight, log_dir, logger=None):
    """
    Training model and save trained weight
    """
    if logger:
        logger.info('Training Model...')

    if log_dir is not None:
        model.fit(image_data_array, image_data_array,
                        epochs=epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(image_data_array, image_data_array),
                        callbacks=[TensorBoard(log_dir=log_dir)])
    else:
        model.fit(image_data_array, image_data_array,
                        epochs=epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(image_data_array, image_data_array))

    model.save_weights(trained_weight)

    if logger:
        logger.info('Finish Traing Model!')


def decode_image(model, image_data_array, image_path_list, decode_dir, logger=None):
    """
    Decode and save images
    """
    if logger:
        logger.info('Decoding and Saving Image...')

    decoded_image_array = model.predict(image_data_array)

    # save
    for (decoded, image_path) in zip(decoded_image_array, image_path_list):
        image_data = array_to_img(decoded)

        # saved path
        if not os.path.exists(decode_dir):
            os.makedirs(decode_dir)
        filename = os.path.basename(image_path)
        base, _ = os.path.splitext(filename)
        save_path = os.path.join(decode_dir, base + '.png')
        cv2.imwrite(save_path, image_data)

    if logger:
        logger.info('Finish Decoding and Saving Image!')

def autoencoder(source_dir, decode_dir, resize_dir, batch_size, epoch, trained_weight, log_dir, logger=None):
    """
    Run autoencoder
    """
    # data preparing
    image_path_list = get_image_list(source_dir)
    image_data_array = load_image(image_path_list, resize_dir, logger)

    # NN model preparing
    model = build_model()
    set_optimizer(model)

    # training
    train_autoencoder(model, image_data_array, batch_size, epoch, trained_weight, log_dir, logger)

    # output result
    decode_image(model, image_data_array, image_path_list, decode_dir, logger=None)


if __name__ == '__main__':
    # setup
    logger = setup_logger()

    parser = setup_argument_parser()
    args = parser.parse_args()

    # main process
    autoencoder(
        source_dir=args.source_dir, 
        decode_dir=args.decode_dir,
        resize_dir=args.resize_dir,
        batch_size=args.batch_size,
        epoch=args.epoch,
        trained_weight=args.trained_weight,
        log_dir=args.log_dir,
        logger=logger,
    )
