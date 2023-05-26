### Copy images to yolov5s_craft_separate
### scale labels and save
### make train/val txt files
import os
import cv2
import shutil
import glob, json
from pathlib import Path
import shutil
#### Source
source_path = '/raid/tmp/Text_detection/GNHK/Datasets/original_GNHK_datasets/Images_GT/'
train_path = source_path + 'train/'
test_path = source_path + 'test/'
mode = 'train/'

### Destination
dest_path =  '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all/'
dest_train = dest_path + 'images/' + 'train/'
dest_test = dest_path + 'images/' + 'val/'
dest_GT_train = dest_path + 'labels/' + 'train/'
dest_GT_test = dest_path + 'labels/' + 'val/'
isExist = os.path.exists(dest_train)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(dest_train)
isExist = os.path.exists(dest_test)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(dest_test)
isExist = os.path.exists(dest_GT_train)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(dest_GT_train)
isExist = os.path.exists(dest_GT_test)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(dest_GT_test)

GT_file_path_train = glob.glob(train_path + '/*.json')
GT_file_path_test = glob.glob(test_path + '/*.json')
class_ = 0
def images_labels(path, GT_file_names, dest_GT, dest_img):
    for i, name in enumerate(GT_file_names):
        GT_name = name
        name = name.split('/')[-1].split('.')[0]
        name_image_file = path + name + '.jpg'
        ### read text file
        img = cv2.imread(name_image_file) ####
        img_width, img_height = img.shape[0], img.shape[1]
        write_label_name_yolov5 = dest_GT + name + '.txt' ###
        GT = json.load(open(GT_name, 'r'))

        with open(write_label_name_yolov5, 'a') as fwrite:
            for j in range(len(GT)):
                line = list(GT[j]['polygon'].values())
                x1 = max(min(line[0], line[2], line[4], line[6]), 0)
                y1 = max(min(line[1], line[3], line[5], line[7]), 0)
                x2 = max(line[0], line[2], line[4], line[6])
                y2 = max(line[1], line[3], line[5], line[7])
                #x, y, w, h = x1, y1, x2 - x1, y2 - y1
                x_center, y_center = (x1+x2)/2, (y1+y2)/2
                width, height = x2-x1, y2-y1
                x_center, width = round(x_center/img_height, 6), round(width/img_height, 6)
                y_center,  height = round(y_center / img_width, 6) , round(height / img_width, 6)
                line = [class_, x_center, y_center, width, height]
                line = map(str, line)
                # Join the items together with commas
                line = " ".join(line)
                # Write to the file
                fwrite.write(line + '\n')
        fwrite.close()
        shutil.copy(path + name + '.jpg', dest_img + name + '.jpg')
def file_names_text_file(GT_file_path, mode):
    #write format = ./images/val2017/000000182611.jpg
    txt_file_path = dest_path
    with open(txt_file_path + mode + '.txt', 'a') as fwrite:
        for i, name in enumerate(GT_file_path):
            name = name.split('/')[-1].split('.')[0] + '.jpg'
            img_name = name
            line = './images/' + mode + '/' + img_name
            fwrite.write(line + '\n')
        fwrite.close()

#images_labels(train_path, GT_file_path_train, dest_GT_train, dest_train)
#images_labels(test_path, GT_file_path_test, dest_GT_test, dest_test)

#file_names_text_file(GT_file_path_train, 'train')
#file_names_text_file(GT_file_path_test, 'val')



#
def copy_images_RS_large_med_small(large_med_small_path , mode):
    source_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all/images/val/'
    des_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all_rs_as_original_size/images/' + mode + '/'
    for n in large_med_small_path:
        mask_file_name = n.split('/')[-1].split('.')[0] + '.jpg'
        shutil.copy2(source_path + mask_file_name, des_path + mask_file_name)

mode = 'small'
large_med_small_path = glob.glob('/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all_rs_original_size/images/' + mode + '/*.jpg')
copy_images_RS_large_med_small(large_med_small_path , mode)

def copy_label_large_med_small(large_med_small_path , mode):
    source_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all/labels/val/'
    des_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all_rs_as_original_size/labels/' + mode + '/'
    for n in large_med_small_path:
        label_file_name = n.split('/')[-1].split('.')[0] + '.txt'
        shutil.copy2(source_path + label_file_name, des_path + label_file_name)


#large_med_small_path = glob.glob('/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all/images/' + mode + '/*.jpg')
copy_label_large_med_small(large_med_small_path , mode)

def file_names_text_file_large_med_small(large_med_small_path, mode):
    #write format = ./images/val2017/000000182611.jpg
    txt_file_path = '/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all/'
    with open(txt_file_path + mode + '.txt', 'a') as fwrite:
        for i, name in enumerate(large_med_small_path):
            name = name.split('/')[-1].split('.')[0] + '.jpg'
            img_name = name
            line = './images/' +mode + '/' + img_name
            fwrite.write(line + '\n')
        fwrite.close()

large_med_small_path = glob.glob('/raid/tmp/Text_detection/GNHK/Datasets/gnhk_all/images/' + mode + '/*.jpg')
file_names_text_file_large_med_small(large_med_small_path, mode)