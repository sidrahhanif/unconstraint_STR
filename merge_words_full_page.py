import os
import cv2
import numpy as np
import torch

import shutil


image_name = '00e8afc1-27de-4b56-8b6e-0211613a33de-trans'


home_path = os.path.dirname(os.path.realpath(__file__))

recovered_STR_path = home_path + '/example_weights/results/imgs/current/eval/'
yolo_words_path = home_path + '/word_detection/runs/detect/exp14/crops/word/'
yolo_txt_path = home_path + '/word_detection/runs/detect/exp14/labels/' + image_name + '.txt'
yolo_path = home_path + '/word_detection/'
test_STR_path = home_path + '/word_detection/test_STR_images/jerry_samples/'
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y

with open(yolo_txt_path) as f:
    contents = f.readlines()

original_image_path = home_path + '/word_detection/runs/detect/exp14/'
                                  #'/home/tug85766/Text_Detection/yolov5/test_STR_images/images/'
original_image = original_image_path + image_name + '.png'
original_image = cv2.imread(original_image)
w,h = original_image.shape[1], original_image.shape[0]
resultant_image = np.ones((original_image.shape[0], original_image.shape[1], original_image.shape[2]), dtype=np.uint8)*255

for line in contents:
    line = line.split(' ')
    sub_image_name_original = test_STR_path + image_name + '.jpg'
    sub_image_name = line[0].split('/')[-1].split('.')[0]
    sub_image_name = recovered_STR_path + 'reconstruction_' + sub_image_name + '_nan.png'
    sub_image_original = cv2.imread(sub_image_name_original)
    sub_image = cv2.imread(sub_image_name)
    original_bbox = [(float(i)) for i in line[2:6]]

    x1, y1, x2, y2 = original_bbox #xywhn2xyxy(original_bbox , w =w, h= h)
    bbox_w, bbox_h = int(x2)-int(x1), int(y2)-int(y1)
    ### 0 width, 1 height
    fx = bbox_h/61

    back_to_original_bbox = cv2.resize(sub_image, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    if int(y2)- int(y1) != back_to_original_bbox.shape[0] or int(x2)- int(x1) != back_to_original_bbox.shape[1]:
        dim  =(int(x2)-int(x1), int(y2)-int(y1))
        back_to_original_bbox = cv2.resize(back_to_original_bbox, dim, interpolation=cv2.INTER_CUBIC)
    resultant_image[int(y1):int(y2), int(x1):int(x2),:] = back_to_original_bbox
#resultant_image = np.concatenate((sub_image_original, resultant_image), axis =1)
cv2.imwrite('results_' +image_name+ '.jpg', resultant_image)
#shutil.rmtree(yolo_path + 'runs/detect/exp/')

