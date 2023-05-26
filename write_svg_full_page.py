from svgwrite import Drawing
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt

'''
The instruction to run the code is as follows:

1) Please provide the name of the image in the variable "image_name".
2) We can change the width in variable line_width, default value is set to 3.
3) The output .svg file is saved as "example_full_page.svg" in the same directory as the script.
'''



image_name = '00fb96a6-171e-4fdb-b63c-a6ee74d0dbb2-trans'
             #'0cca7f95-1c93-4106-81f4-8bd6b545510b-trans'
    #'00e8afc1-27de-4b56-8b6e-0211613a33de-trans'# '0fb8be54-2431-462d-b430-04fbd334de36-trans'#
home_path = os.path.dirname(os.path.realpath(__file__))
recovered_STR_path = home_path + '/example_weights/results/imgs/current/eval/'
yolo_words_path = home_path + '/word_detection/runs/detect/exp18/crops/word/'
yolo_txt_path = home_path + '/word_detection/runs/detect/exp18/labels/' + image_name + '.txt'
yolo_path = home_path + '/word_detection/'
test_STR_path = home_path + '/word_detection/test_STR_images/images_ryolo/'

with open(yolo_txt_path) as f:
    contents = f.readlines()
original_image_path = home_path + '/word_detection/runs/detect/exp18/'
                                  #'/home/tug85766/Text_Detection/yolov5/test_STR_images/images/'
original_image = original_image_path + image_name + '.png'
original_image = cv2.imread(original_image)
w, h = original_image.shape[1], original_image.shape[0]
Line = []
for line in contents:
    line = line.split(' ')
    sub_image_name_original = test_STR_path + image_name + '.png'
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
            print(len(Coordinates))
            print(k)
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
                    if k== len(Coordinates):
                        break
                Line.append([p, p1])
                p1, p = [], []
w = w
h = h

line_width = 3
w_str = "{}pt".format(w)
h_str = "{}pt".format(h)
fn = image_name + str(10)+ '.svg'

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

#shutil.rmtree(yolo_path + 'runs/detect/exp/')