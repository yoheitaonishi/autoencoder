import argparse
import glob
import os

import cv2
import numpy as np
import random
from tqdm import tqdm

def setup_argument_parser():
    """
    Set argument
    """
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input-dir', help='input images directory', default='work/input/', required=True)
    parser.add_argument('--include', help='input images file name', default='*.jpg')
    parser.add_argument('--output-dir', help='source image directory', default='work/output/', required=True)
    parser.add_argument('--output-prefix', help='source image file name', default='label_poodl', required=True)
    parser.add_argument('--output-ext', help='source image file extension', default='png', required=True)
    parser.add_argument('--size', help='crop size (width, height)', default='300, 300', required=True)
    parser.add_argument('--count', help='number of output images', type=int, default=100, required=True)
    parser.add_argument('--seed', help='random seed for deciding origin point(x,y)', type=int, default=1234567)
    return parser

def get_random_value(image_size, crop_size):
    max_range = image_size - crop_size
    value = random.random() * max_range
    value = round(value)
    return value

def get_random_origin_point(image_width, image_height, crop_width, crop_height):
    x = get_random_value(image_size=image_width, crop_size=crop_width)
    y = get_random_value(image_size=image_height, crop_size=crop_height)
    width = x + crop_width
    height = y + crop_height
    return x, y, width, height
 
def get_image_info(image_shape, size, seed):
    """
    Get image information after cropped (x, y, width, height)
    """
    size = size.split(',')
    size = [int(a) for a in size]

    image_width = image_shape[1]
    image_height = image_shape[0]
    crop_width = size[0]
    crop_height = size[1]

    x, y, width, height = get_random_origin_point(image_width=image_width, image_height=image_height, crop_width=crop_width, crop_height=crop_height)
    return x, y, width, height

def get_image_src(input_dir, include):
    """
    Get image path
    """
    image_list = glob.glob(os.path.join(input_dir, include))
    image_data_src = random.choice(image_list)
    return image_data_src

def crop_images(input_dir, include, output_dir, output_prefix, output_ext, count, size):
    """
    Crop images
    """
    for i in tqdm(range(count)):
        image_data_src = get_image_src(input_dir=input_dir, include=include)
        image_data = cv2.imread(image_data_src)
        image_shape = np.shape(image_data)
        x, y, width, height = get_image_info(image_shape=image_shape, size=size, seed=seed)
        image_data = image_data[y:height, x:width]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i_zero = str(i).zfill(8)
        file_name = output_prefix + '_' + i_zero + '.' + output_ext
        resize_image_save_file = os.path.join(output_dir, file_name)
        cv2.imwrite(resize_image_save_file, image_data)

if __name__ == '__main__':
    # setup
    parser = setup_argument_parser()
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)

    # main process
    crop_images(
        input_dir=args.input_dir,
        include=args.include,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        output_ext=args.output_ext,
        count=args.count,
        size=args.size,
    )
