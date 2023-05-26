### Format: class x_center y_center width height
### x_center, w / image_w || y_center, h / image_h ||

### find image width and height

import os
import cv2
import shutil

gnhk_image_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/only_train_img/'
gnhk_label_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/train_gt_east/'
gnhk_write_label_yolov5 = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/yolov5/labels/train/'
gnhk_write_img_yolov5 = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/yolov5/images/train/'
class_ = 0
image_file_path = os.listdir(gnhk_image_path)
def images_labels():
    for i, name in enumerate(image_file_path):
        img_name = name
        name = name.split('.')[0]
        name_text_file = gnhk_label_path + name + '.txt'
        ### read text file
        img = cv2.imread(gnhk_image_path + img_name)
        img_width, img_height = img.shape[0], img.shape[1]
        write_label_name_yolov5 = gnhk_write_label_yolov5 + name + '.txt'
        with open(name_text_file) as f:
            contents = f.readlines()
            with open(write_label_name_yolov5, 'a') as fwrite:
                for each_row in contents:
                    row = each_row.split(',')
                    x1 = min(int(row[0]), int(row[2]), int(row[4]), int(row[6]))
                    y1 = min(int(row[1]), int(row[3]), int(row[5]), int(row[7]))
                    x2 = max(int(row[0]), int(row[2]), int(row[4]), int(row[6]))
                    y2 = max(int(row[1]), int(row[3]), int(row[5]), int(row[7]))
                    #x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[4]), int(row[5])

                    x_center, y_center = (x1+x2)/2, (y1+y2)/2
                    width, height = x2-x1, y2-y1
                    x_center, width = round(x_center/img_height, 6), round(width/img_height,6)
                    y_center,  height = round(y_center / img_width,6) , round(height / img_width,6)
                    line = [class_, x_center, y_center, width, height]
                    line = map(str, line)
                    # Join the items together with commas
                    line = " ".join(line)
                    # Write to the file
                    fwrite.write(line + '\n')


        fwrite.close()
        shutil.copy(gnhk_image_path + img_name, gnhk_write_img_yolov5 + img_name)

def file_names_text_file():
    #write format = ./images/val2017/000000182611.jpg
    txt_file_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/yolov5_2048/'
    with open(txt_file_path + 'train.txt', 'a') as fwrite:
        for i, name in enumerate(image_file_path):
            img_name = name



            line = './images/train/'+ img_name

            # Join the items together with commas

            # Write to the file
            fwrite.write(line + '\n')

        fwrite.close()

#images_labels()
file_names_text_file()


