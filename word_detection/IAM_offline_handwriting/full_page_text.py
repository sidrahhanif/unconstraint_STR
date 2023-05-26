import glob
import sys

import html
from html.parser import HTMLParser
import xml.etree.ElementTree
from os import listdir
from os.path import isfile, join
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from PIL import Image, ImageDraw
matplotlib.use('Tkagg')
lines_images_path = '/home/tug85766/Trace/data_processing/prepare_IAM_Lines/lines/r06/r06-143/'
bounding_box_xml_path = '/home/tug85766/Trace/data_processing/prepare_IAM_Lines/xml/'


bounding_box_xml_list = glob.glob(bounding_box_xml_path + '*.xml')
for xml_name in bounding_box_xml_list:
    root = xml.etree.ElementTree.parse(xml_file).getroot()
    namespace = get_namespace(root)

    handwritten_part = root.find('handwritten-part')

    lbys = []
    dys = []
    line_gts = {}
    word_gts = {}
    img_path = '/home/tug85766/Trace/data_processing/prepare_IAM_Lines/lines/'
    for line in handwritten_part:
        image_name = line.attrib['id']
        if image_name == 'n02-127-04':
            a = 1
        print(image_name + '\n')
        GT_all = []
        word_label = []
        folder_path = image_name.split('-')[0] + '/' + '-'.join(image_name.split('-')[0:2]) + '/'
        img = img_path + folder_path + image_name + '.png'
        min_x = 1000000
        min_y = 1000000
        for num, word in enumerate(line):

            if word.tag == 'word':
                for k in range(len(word)):
                    cmp = word[k]
                    if int(cmp.attrib['x']) < min_x:
                        min_x = int(cmp.attrib['x'])
                    if int(cmp.attrib['y']) < min_y:
                        min_y = int(cmp.attrib['y'])
        x_offset, y_offset = min_x, min_y
        for num, word in enumerate(line):

            if word.tag == 'word':
                bbox = []
                for k in range(len(word)):
                    cmp = word[k]

                    x, y, w, h = int(cmp.attrib['x']) - x_offset, int(cmp.attrib['y']) - y_offset, int(
                        cmp.attrib['width']), int(cmp.attrib['height'])
                    bbox.append([x, y, w, h])
                if bbox == []:
                    continue
                GT_all.append(combineBoundingBox(bbox))
                word_label.append(word.attrib['text'])