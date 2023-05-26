import numpy as np
import cv2
from matplotlib import image
from matplotlib import pyplot as plt

img_path = '/dev/shm/strokes_recovery/Signed_data_writing/results on Good_sorted_images_wo_black_images/0ad9b523-707b-4582-9291-017a3777df3e-trans.png'
label_path = '/dev/shm/strokes_recovery/Signed_data_writing/results on Good_sorted_images_wo_black_images/labels/0ad9b523-707b-4582-9291-017a3777df3e-trans.txt'
data = image.imread(img_path)
with open(label_path) as f:
    contents = f.readlines()

    for each_row in contents:
        row = each_row.split('\n')[0].split(' ')
        point = []
        #point1, point2,point3,point4
        point.append([int(float(row[5])), int(float(row[6]))])

        point.append([int(float(row[1])), int(float(row[2]))])
        point.append([int(float(row[3])), int(float(row[4]))])
        point.append([int(float(row[7])), int(float(row[8]))])
        A = cv2.minAreaRect(np.asarray(point))
        print(A)

        # to draw a point on co-ordinate (200,300)
        plt.plot(point[0][0], point[0][1], marker='o', color="red", markersize=2)
        plt.plot(point[1][0], point[1][1], marker='x', color="green", markersize=2)
        plt.plot(point[2][0], point[2][1], marker='o', color="blue", markersize=2)
        plt.plot(point[3][0], point[3][1], marker='^', color="yellow", markersize=2)
plt.imshow(data)
plt.savefig('books_read.png')

a =1

