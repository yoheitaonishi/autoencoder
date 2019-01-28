from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
import numpy as np
import glob
from PIL import Image

#from keras.datasets import mnist
#(x_train, _), (x_test, __) = mnist.load_data()
#
#print(np.shape(x_train[0]))
#print(np.shape(x_train))
#print(type(x_train))


# Get images and convert

IMAGE_FILE = './work/images/*.jpg'
image_list = glob.glob(IMAGE_FILE)
image_data_list = np.asarray([np.asarray(Image.open(image_data)) for image_data in image_list])
print(np.shape(image_data_list[0]))
print(np.shape(image_data_list))
print(type(image_data_list))

# convert




#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 128, 128, 3))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 128, 128, 3))  # adapt this if using `channels_first` image data format

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
x = Conv2D(64, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')












"""
Get image file and convert jpg file to numpy array.
Resize and save image file. 
"""

"""
Create and training network
Save trained model
"""

"""
Get image file
"""

