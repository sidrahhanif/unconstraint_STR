### Copy images to yolov5s_craft_separate
### scale labels and save
### make train/val txt files
import os
import cv2
import shutil

#### Source
source_path = '/raid/tmp/Text_detection/CRAFT_fasterrcnn/'
train_rs_plus_as = source_path + 'train_rs_2048/'
test_rs_plus_as = source_path + 'test_rs_2048/'
GT_test_CRAFT = source_path + 'test_rs_GT_2048/'
GT_train_CRAFT = source_path + 'train_rs_GT_2048/'
### Destination
dest_path =  '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/yolov5s_RS_CRAFT_2048/'
dest_train_rs_plus_as = dest_path + 'images/' + 'train/'
dest_test_rs_plus_as = dest_path + 'images/' + 'val/'
dest_GT_train_CRAFT = dest_path + 'labels/' + 'train/'
dest_GT_test_CRAFT = dest_path + 'labels/' + 'val/'
image_file_path = os.listdir(train_rs_plus_as)

class_ = 0
def images_labels():
    for i, name in enumerate(image_file_path):
        img_name = name
        name = name.split('.')[0].split('_mask')[0]
        name_text_file = GT_train_CRAFT + name + '.txt'
        ### read text file
        img = cv2.imread(train_rs_plus_as + img_name) ####
        img_width, img_height = img.shape[0], img.shape[1]
        write_label_name_yolov5 = dest_GT_train_CRAFT + name + '.txt' ####
        with open(name_text_file) as f:
            contents = f.readlines()
            with open(write_label_name_yolov5, 'a') as fwrite:
                for each_row in contents:
                    row = each_row.split(',')

                    x1 = min(int(row[0]), int(row[2]), int(row[4]), int(row[6]))
                    y1 = min(int(row[1]), int(row[3]), int(row[5]), int(row[7]))
                    x2 = max(int(row[0]), int(row[2]), int(row[4]), int(row[6]))
                    y2 = max(int(row[1]), int(row[3]), int(row[5]), int(row[7]))
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
        shutil.copy(train_rs_plus_as + img_name, dest_train_rs_plus_as + name + '.jpg')
def file_names_text_file():
    #write format = ./images/val2017/000000182611.jpg
    txt_file_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_correct_GT/yolov5s_RS_CRAFT_2048/'
    with open(txt_file_path + 'train.txt', 'a') as fwrite:
        for i, name in enumerate(image_file_path):


            name = name.split('.')[0].split('_mask')[0] + '.jpg'
            img_name = name



            line = './images/train/'+ img_name

            # Join the items together with commas

            # Write to the file
            fwrite.write(line + '\n')

        fwrite.close()
file_names_text_file()
images_labels()

