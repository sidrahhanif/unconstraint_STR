import glob
import shutil
import os
import json
from PIL import Image
import numpy as np
line = False
words = True

if line == True:
    test_image = json.load(open('/home/tug85766/Trace+CC/data_processing/online_coordinate_data/'
                                 'MAX_stroke_vNORMAL_TRAINING_TESTFull/test_online_coords.json','r'))
    train_image = json.load(open('/home/tug85766/Trace+CC/data_processing/online_coordinate_data/'
                                 'MAX_stroke_vNORMAL_TRAINING_TESTFull/train_online_coords.json','r'))

    dst_path = '/home/tug85766/Trace/data_processing/prepare_IAM_Lines/Images_test_trace/'
    scr_path = '/home/tug85766/Trace/data_processing/'

    for i in range(len(test_image)):
        image = scr_path + test_image[i]['full_img_path']
        name = image.split('/')[9].split('.')[0]
        im = Image.open(image)
        im.save(dst_path + name + '.png')

    print('Done Test!')

    for i in range(len(train_image)):
        image = scr_path + train_image[i]['full_img_path']
        name = image.split('/')[9].split('.')[0]
        im = Image.open(image)
        im.save(dst_path + name + '.png')

    print('Done Train!')


if words == True:
    path = ''
    words_image = os.listdir(path)
    for i in range(len(words_image)):
        image = scr_path + words_image[i]['full_img_path']
        name = image.split('/')[9].split('.')[0]
        im = Image.open(image)
        im.save(dst_path + name + '.png')

    print('Done Test!')