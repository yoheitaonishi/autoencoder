from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import glob
from PIL import Image

# Get images and convert
IMAGE_FILE = './work/images/*.jpg'
image_list = glob.glob(IMAGE_FILE)
image_data_array = []
for image_path in image_list:
    image_data_list = Image.open(image_path)
    image_data_list = np.asarray(image_data_list.resize((128, 128)))
    image_data_array.append(image_data_list)

image_data_array = np.array(image_data_array)
image_data_array = np.reshape(image_data_array, (len(image_data_array), 128, 128, 3))  # adapt this if using `channels_first` image data format

input_img = Input(shape=(128, 128, 3))  # adapt this if using `channels_first` image data format

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

autoencoder.fit(image_data_array, image_data_array,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(image_data_array, image_data_array),
                callbacks=[TensorBoard(log_dir='./logs/log.log')])

