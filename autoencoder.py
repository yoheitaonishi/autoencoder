from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.preprocessing.image import array_to_img
import keras.backend as K
import os
import argparse
import numpy as np
import glob
from PIL import Image
import logging
from logging import getLogger, StreamHandler, Formatter

# Set up logger
logger = getLogger("RUN INFORMATION")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/log.log')

# Set argument
parser = argparse.ArgumentParser() 
parser.add_argument('--source-dir', help='images directory', default='./work/images')
parser.add_argument('--decode-dir', help='decode images directory', default='./work/images/decode')
parser.add_argument('--resize-dir', help='resize images directory', default = './work/images/resize')
parser.add_argument('--batch-size', help='batch size', type=int, default=128)
parser.add_argument('--epoch', help='epoch', type=int, default=50)
args = parser.parse_args()

# Directory path
IMAGE_FILE = args.source_dir + '/*.jpg'
DECODE_IMAGE_SAVE_PATH = args.decode_dir + '/'
RESIZE_IMAGE_SAVE_PATH = args.resize_dir + '/'

image_list = glob.glob(IMAGE_FILE)
image_data_array = []

logger.info('Loading Image...')

for image_path in image_list:
    image_data = Image.open(image_path)
    image_data = image_data.resize((128, 128))

    basename = os.path.basename(image_path)
    basename = basename.replace('jpg', 'png')
    resize_image_save_file = RESIZE_IMAGE_SAVE_PATH + basename
    image_data.save(resize_image_save_file)

    image_data = np.asarray(image_data)
    image_data_array.append(image_data)

image_data_array = np.array(image_data_array)
image_data_array = image_data_array.astype('float32') / 255.
image_data_array = np.reshape(image_data_array, (len(image_data_array), 128, 128, 3)) 

logger.info('Finish Loading Image!')

input_img = Input(shape=(128, 128, 3)) 
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
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

logger.info('Traing Model...')

autoencoder.fit(image_data_array, image_data_array,
                epochs=args.epoch,
                batch_size=args.batch_size,
                shuffle=True,
                validation_data=(image_data_array, image_data_array),
                callbacks=[TensorBoard(log_dir='./logs')])
autoencoder.save_weights("./train.h5")

logger.info('Finish Traing Model!')


logger.info('Decoding and Saving Image...')

decoded_images = autoencoder.predict(image_data_array)
for (decode_image, image_path) in zip(decoded_images, image_list):
    image_data = array_to_img(decode_image)
    basename = os.path.basename(image_path)
    basename = basename.replace('jpg', 'png')
    decode_image_save_file = DECODE_IMAGE_SAVE_PATH + basename
    image_data.save(decode_image_save_file)

logger.info('Finish Decoding and Saving Image!')
