import argparse
import glob
import logging
import os

import numpy as np
import cv2
import random
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils import plot_model

INPUT_IMAGE_SIZE = 128
SALT_AND_PEPPER_NUMBER = 1000

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
    parser.add_argument('--output-prefix', help='source image file name', default='label_poodl', required=True)
    parser.add_argument('--output-ext', help='source image file extension', default='png', required=True)
    parser.add_argument('--batch-size', help='batch size', type=int, default=16)
    parser.add_argument('--epoch', help='epoch', type=int, default=50)
    parser.add_argument('--trained-weight', help='saving trained h5 file directory', default='./work/trained.h5')
    parser.add_argument('--initial-weight', help='load pretrain h5 file', default='./pretrained.h5')
    parser.add_argument('--log-dir', help='tensor board log directory')
    parser.add_argument('--count', type=int, help='number of preprocessing images', default=100)
    parser.add_argument('--salt-and-pepper-noise', type=bool, help='use salt-and-pepper noise for preprocessing', default=False)
    parser.add_argument('--salt-and-pepper-noise-rgb', type=int, help='rgb value of salt-and-pepper noise for preprocessing', default=0)
    parser.add_argument('--hsv-noise', type=bool, help='use hsv noise for preprocessing', default=False)
    parser.add_argument('--hue', type=int, help='maximum h parameter of hsv', default=0)
    parser.add_argument('--saturation', type=int, help='maximum s parameter of hsv', default=0)
    parser.add_argument('--value', type=int, help='maximum v parameter of hsv', default=0)
    return parser

def get_image_list(source_dir):
    """
    Get image path
    """
    image_list = glob.glob(source_dir + '/*.jpg')
    return image_list

def processing_salt_and_pepper_noise(image_data, salt_and_pepper_noise_rgb):
    row, col, ch = image_data.shape
    pts_x = np.random.randint(0, col-1 , SALT_AND_PEPPER_NUMBER)
    pts_y = np.random.randint(0, row-1 , SALT_AND_PEPPER_NUMBER)
    image_data[(pts_y, pts_x)] = (salt_and_pepper_noise_rgb, salt_and_pepper_noise_rgb, salt_and_pepper_noise_rgb)
    return image_data

def processing_hsv_noise(image_data, hue, saturation, value):
    hsv_image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
    hsv_image_shape = hsv_image_data.shape
    new_image = np.zeros(hsv_image_shape, np.uint8)

    random_h = round(random.random() * hue * 2 - hue)
    random_s = round(random.random() * saturation * 2 - saturation)
    random_v = round(random.random() * value * 2 - value)

    new_image = new_image + np.array([random_h, random_s, random_v], dtype=np.uint8)
    hsv_image_data = hsv_image_data + new_image

    image_data = cv2.cvtColor(hsv_image_data, cv2.COLOR_HSV2BGR)
    return image_data

def load_image(image_list, resize_dir, output_prefix, output_ext, count, salt_and_pepper_noise, salt_and_pepper_noise_rgb, hsv_noise, hue, saturation, value, logger=None):
    """
    Load image and convert packed numpy array
    """
    image_data_array = []

    if logger:
        logger.info('Loading Image...')

    for i in range(count):
        image_path = random.choice(image_list)
        image_data = cv2.imread(image_path)
        image_data = cv2.resize(image_data, dsize=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))

        # data preprocessing
        if salt_and_pepper_noise is True:
            image_data = processing_salt_and_pepper_noise(image_data=image_data, salt_and_pepper_noise_rgb=salt_and_pepper_noise_rgb)
        if hsv_noise is True:
            image_data = processing_hsv_noise(image_data=image_data, hue=hue, saturation=saturation, value=value)

        if not os.path.exists(resize_dir):
            os.makedirs(resize_dir)

        i_zero = str(i).zfill(8)
        file_name = output_prefix + '_' + i_zero + '.' + output_ext
        resize_image_save_file = os.path.join(resize_dir, file_name)
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

def decode_image(model, image_data_array, image_path_list, decode_dir, output_prefix, output_ext, logger=None):
    """
    Decode and save images
    """
    if logger:
        logger.info('Decoding and Saving Image...')

    decoded_image_array = model.predict(image_data_array)

    # save
    for i, decoded in enumerate(decoded_image_array):
        image_data = decoded * 255

        # saved path
        if not os.path.exists(decode_dir):
            os.makedirs(decode_dir)
        i_zero = str(i).zfill(8)
        file_name = output_prefix + '_' + i_zero + '.' + output_ext
        save_path = os.path.join(decode_dir, file_name)
        cv2.imwrite(save_path, image_data)

    if logger:
        logger.info('Finish Decoding and Saving Image!')

def autoencoder(source_dir, decode_dir, resize_dir, output_prefix, output_ext, batch_size, epoch, trained_weight, initial_weight, log_dir, count, salt_and_pepper_noise, salt_and_pepper_noise_rgb, hsv_noise, hue, saturation, value, logger=None):
    """
    Run autoencoder
    """
    # data preparing
    image_path_list = get_image_list(source_dir)
    image_data_array = load_image(
        image_list=image_path_list, 
        resize_dir=resize_dir, 
        output_prefix=output_prefix,
        output_ext=output_ext,
        count=count, 
        salt_and_pepper_noise=salt_and_pepper_noise, 
        salt_and_pepper_noise_rgb=salt_and_pepper_noise_rgb, 
        hsv_noise=hsv_noise, 
        hue=hue, 
        saturation=saturation, 
        value=value,
        logger=logger
    )

    # NN model preparing
    model = build_model()
    if initial_weight is not None:
        model.load_weights(initial_weight)
    set_optimizer(model)

    # training
    train_autoencoder(model, image_data_array, batch_size, epoch, trained_weight, log_dir, logger)

    # output result
    decode_image(model, image_data_array, image_path_list, decode_dir, output_prefix, output_ext, logger=None)

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
        output_prefix=args.output_prefix,
        output_ext=args.output_ext,
        batch_size=args.batch_size,
        epoch=args.epoch,
        trained_weight=args.trained_weight,
        initial_weight=args.initial_weight,
        log_dir=args.log_dir,
        count=args.count,
        salt_and_pepper_noise=args.salt_and_pepper_noise,
        salt_and_pepper_noise_rgb=args.salt_and_pepper_noise_rgb,
        hsv_noise=args.hsv_noise,
        hue=args.hue, 
        saturation=args.saturation, 
        value=args.value,
        logger=logger,
    )
