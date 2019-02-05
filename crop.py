import argparse
import glob
import os

import cv2



parser = argparse.ArgumentParser() 
parser.add_argument('--input-dir', help='input images directory', default='work/input/', required=True)
parser.add_argument('--include', help='input images file name', default='*.jpg', required=True)
parser.add_argument('--output-dir', help='source image directory', default='work/output/', required=True)
parser.add_argument('--area', help='crop area (x, y, width, height)', default='252,427,1050,800', required=True)
args = parser.parse_args()


input_dir = args.input_dir
include = args.include
output_dir = args.output_dir
area = args.area

area = area.split(',')
area = [int(a) for a in area]

x = area[0]
y = area[1]
width = x + area[2]
height = y + area[3]

image_list = glob.glob(input_dir + include)

for image_path in image_list:
    image_data = cv2.imread(image_path)

    image_data = image_data[y:height, x:width]

    basename = os.path.basename(image_path)
    basename = basename.replace('jpg', 'png')
    resize_image_save_file = os.path.join(output_dir, basename)
    cv2.imwrite(resize_image_save_file, image_data)


