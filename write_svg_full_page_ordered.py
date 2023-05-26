from svgwrite import Drawing
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt

from operator import itemgetter
'''
The instruction to run the code is as follows:

1) Please provide the name of the image in the variable "image_name".
2) We can change the width in variable line_width, default value is set to 3.
3) The output .svg file is saved as "example_full_page.svg" in the same directory as the script.
'''

def bounding_box_sorting(boxes):
    num_boxes = boxes.shape[0]
    # sort from top to bottom and left to right
    sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    # print('::::::::::::::::::::::::::testing')

    # check if the next neighgour box x coordinates is greater then the current box x coordinates if not swap them.
    # repeat the swaping process to a threshold iteration and also select the threshold
    threshold_value_y = 10
    for i in range(5):
      for i in range(num_boxes - 1):
          if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < threshold_value_y and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
              tmp = _boxes[i]
              _boxes[i] = _boxes[i + 1]
              _boxes[i + 1] = tmp
    return _boxes

image_name = '18'
home_path = os.path.dirname(os.path.realpath(__file__))
recovered_STR_path = home_path + '/example_weights/results/imgs/current/eval/'
yolo_words_path = home_path + '/word_detection/runs/detect/exp/crops/word/'
yolo_txt_path = home_path + '/word_detection/runs/detect/exp/labels/' + image_name + '.txt'
yolo_path = home_path + '/word_detection/'
test_STR_path = home_path + '/word_detection/test_STR_images/images/'

with open(yolo_txt_path) as f:
    contents = f.readlines()
original_image_path = home_path + '/word_detection/runs/detect/exp/'
                                  #'/home/tug85766/Text_Detection/yolov5/test_STR_images/images/'
original_image = original_image_path + image_name + '.jpg'
original_image = cv2.imread(original_image)
w, h = original_image.shape[1], original_image.shape[0]
Line = []
bounding_boxes = []
### TODO: Order the contents of the lines here
for k, line in enumerate(contents):
    line = line.split(' ')
    sub_image_name_original = test_STR_path + image_name + '.jpg'
    sub_image_name_line = line[0].split('/')[-1].split('.')[0]
    sub_image_name = recovered_STR_path + 'reconstruction_' + sub_image_name_line + '_nan.png'
    sub_image_original = cv2.imread(sub_image_name_original)
    sub_image = cv2.imread(sub_image_name)
    original_bbox = [(float(i)) for i in line[2:6]]

    bounding_boxes.append([[original_bbox[0], original_bbox[1]],[original_bbox[2], original_bbox[1]],
                                                                       [original_bbox[2], original_bbox[3]],[original_bbox[0], original_bbox[3]], k])
    #xywhn2xyxy(original_bbox , w =w, h= h)
points = list(bounding_box_sorting(np.asarray(bounding_boxes)))
#indices, L_sorted = zip(*sorted(enumerate(bounding_boxes), key=lambda k: [k[1], k[0]]))
"""
indexes = []
for each_line in points:
    for e in each_line:
        indexes.append(int(e))
"""
contents1 = [contents[i[4]] for i in points]
########## TODO: ordering of lines stop here
for line in contents1:
    line = line.split(' ')
    sub_image_name_original = test_STR_path + image_name + '.jpg'
    sub_image_name_line = line[0].split('/')[-1].split('.')[0]
    sub_image_name = recovered_STR_path + 'reconstruction_' + sub_image_name_line + '_nan.png'
    sub_image_original = cv2.imread(sub_image_name_original)
    sub_image = cv2.imread(sub_image_name)
    original_bbox = [(float(i)) for i in line[2:6]]

    x1, y1, x2, y2 = original_bbox #xywhn2xyxy(original_bbox , w =w, h= h)
    bbox_w, bbox_h = int(x2)-int(x1), int(y2)-int(y1)

    Coordinates = np.load(recovered_STR_path + sub_image_name_line + '_nan.npy', allow_pickle = True).tolist()
    count = 0
    for i in range(len(Coordinates)):
        Coordinates[i][0] = Coordinates[i][0] * bbox_h +x1
        Coordinates[i][1] = bbox_h-(Coordinates[i][1] * bbox_h) +y1

        if Coordinates[i][2] == 1:
            count += 1
    p, p1 = [], []
    k = 0
    for i in range(count):
        if Coordinates[k][2] == 1:
            p = [Coordinates[k][0], Coordinates[k][1]]

            if Coordinates[k+1][2] == 1:
                p1 = [[Coordinates[k][0]+1, Coordinates[k][1]+1]]
                #Line.append([p,p1])
                p1, p =[], []
                k += 1
            else:
                k += 1
                while (Coordinates[k][2] != 1):
                    p1.append([Coordinates[k][0], Coordinates[k][1]])
                    k +=1
                    if k == len(Coordinates):
                        break
                Line.append([p, p1])
                p1, p = [], []
w = w
h = h

line_width = 1
w_str = "{}pt".format(w)
h_str = "{}pt".format(h)
fn = 'example_full_page.svg'

dwg = Drawing(filename=fn,
              size=(w_str, h_str),
              viewBox=("0 0 {} {}".format(w, h)))
paths = Line
for path in paths:
    #print(path)
    #print(len(path))
    #print(path[0][0], path[0][1])
    if (len(path) > 1):
        str_listM = []
        str_listM.append("M {},{}".format(path[0][0], path[0][1]))
        str_listC = []
        for e in path[1]:
            #print(e)
            if str_listC == []:
                str_listC.append(" L {},{}".format(e[0], e[1]))
            else:
                str_listC.append(" {},{}".format(e[0], e[1]))
        s = ''.join(str_listC)
        s = str_listM[0] + s
        dwg.add(dwg.path(s).stroke(color="black", width=line_width).fill("none"))
        dwg.save()
dwg.save()

#shutil.rmtree(yolo_path + 'runs/detect/exp/')