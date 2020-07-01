# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import json
from collections import OrderedDict
from text_detection import imgproc

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None, num = 0):
        img = np.array(img)
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        if filename == "clip_1":
            res_file = OrderedDict()
        else:
            jstring = open("./labels.json", "r").read()
            res_file = json.loads(jstring, object_pairs_hook=OrderedDict)

        res_img_file = dirname + filename + '.png'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        res_file[filename + ".png"] = OrderedDict()
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            absolute_coord = "[" + str(poly[0]) + " " + str(poly[1]) + " " + str(poly[2]) + " " + str(poly[3]) + " " \
                             + str(poly[4]) + " " + str(poly[5]) + " " + str(poly[6]) + " " + str(poly[7]) + "]"
            res_file[filename+".png"]["textbox_{:d}".format(i)] = {"absolute_coord": absolute_coord, "contents": None}
        jstring = json.dumps(res_file, indent=4)
        f = open("./labels.json", "w")
        f.write(jstring)
        f.close()
