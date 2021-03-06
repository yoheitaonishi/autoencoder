import argparse
import glob
import logging
import os

import cv2
from tqdm import tqdm

def setup_logger():
    """
    Set up logger
    """
    logger = logging.getLogger("RUN INFORMATION")
    logger.setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/log.log')
    return logger

def setup_argument_parser():
    """
    Set argument
    """
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input-dir', help='input images directory', default='work/input/', required=True)
    parser.add_argument('--include', help='input images file name', default='*.jpg')
    parser.add_argument('--output-dir', help='source image directory', default='work/output/', required=True)
    parser.add_argument('--area', help='crop area (x, y, width, height)', default='252,427,1050,800', required=True)
    return parser

def get_image_info(area):
    """
    Get image information after cropped (x, y, width, height)
    """
    area = area.split(',')
    area = [int(a) for a in area]

    x = area[0]
    y = area[1]
    width = x + area[2]
    height = y + area[3]
    return x, y, width, height

def get_image_list(input_dir, include):
    """
    Get image path
    """
    image_list = glob.glob(os.path.join(input_dir, include))
    return image_list

def crop_images(input_dir, include, output_dir, area, logger=None):
    """
    Crop images
    """
    image_list = get_image_list(input_dir, include)
    x, y, width, height = get_image_info(area=area)

    for image_path in tqdm(image_list):
        image_data = cv2.imread(image_path)

        row, col, ch = image_data.shape
        if logger and (height > row or width > col):
            logger.warning('Crop size is out of image size. File name is ' + image_path)

        image_data = image_data[y:height, x:width]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        basename = os.path.basename(image_path)
        basename = basename.replace('jpg', 'png')
        resize_image_save_file = os.path.join(output_dir, basename)
        cv2.imwrite(resize_image_save_file, image_data)

if __name__ == '__main__':
    # setup
    logger = setup_logger()
    parser = setup_argument_parser()
    args = parser.parse_args()

    # main process
    crop_images(
        input_dir=args.input_dir,
        include=args.include,
        output_dir=args.output_dir,
        area=args.area,
        logger=logger
    )
