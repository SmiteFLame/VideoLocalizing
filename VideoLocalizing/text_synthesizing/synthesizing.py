"""
SRNet - Editing Text in the Wild
Data prediction.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import tensorflow as tf
from text_synthesizing.model import SRNet
import numpy as np
import os
from skimage.measure import compare_ssim
import pygame
from pygame import freetype
from text_synthesizing import cfg
from text_synthesizing.utils import *
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
import json
import collections
import math
import cv2

def mse(x, y):
    return np.linalg.norm(x - y)

def averaging(img1, img2):
    img1_average = np.average(np.average(img1, axis=0), axis=0).astype(int)
    img2_average = np.average(np.average(img2, axis=0), axis=0).astype(int)
    diff1 = abs(img1_average[0] - img2_average[0])
    diff2 = abs(img1_average[1] - img2_average[1])
    diff3 = abs(img1_average[2] - img2_average[2])
    if diff1 < 5 and diff2 < 5 and diff3 < 5:
        np.clip((img1 - (img1_average - img2_average)),0, 255, out=img1)
    return img1

def comparing(img1, img2, label1, label2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if label1["contents"] != label2["contents"]:
        return 0
    abs_coord1 = np.array([int(i) for i in label1["absolute_coord"][1:-1].split(" ")])
    abs_coord2 = np.array([int(i) for i in label2["absolute_coord"][1:-1].split(" ")])
    for i in range (8):
        if abs(abs_coord1[i]-abs_coord2[i]) > 15:
            return 0
    if (img1.shape[0]* img1.shape[1]) < (img2.shape[0]* img2.shape[1]):
        height = img1.shape[0]
        width = img1.shape[1]
    else:
        height = img2.shape[0]
        width = img2.shape[1]
    #d_img1 = cv2.resize(img1, (int(height/10), int(width/10)), interpolation=cv2.INTER_AREA)
    #d_img2 = cv2.resize(img2, (int(height/10), int(width/10)), interpolation=cv2.INTER_AREA)
    d_img1 = cv2.resize(img1, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    d_img2 = cv2.resize(img2, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    
    mse_score = mse(d_img1, d_img2)
    ssim_score = compare_ssim(d_img1, d_img2, full=True)

    return ssim_score[0]

def cropping(img, data, j):
    line = data["textbox_{:d}".format(j)]["absolute_coord"]
    line = [int(i) for i in line[1:-1].split(" ")]
    line = np.array(line)
    cnt = line.reshape(-1, 2)
    rect = cv2.minAreaRect(cnt)
    center, size = rect[0], rect[1]
    size0 = abs(int(math.sqrt((int(line[3]) - int(line[1])) ** 2 + (int(line[2]) - int(line[0])) ** 2)))
    size1 = abs(int(math.sqrt((int(line[5]) - int(line[3])) ** 2 + (int(line[4]) - int(line[2])) ** 2)))
    angle = math.atan((int(line[3]) - int(line[1])) / (int(line[2]) - int(line[0]))) * 180 / math.pi
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    new_size = (size0, size1)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, new_size, center)
    return img_crop

def masking(i_s, data):
    fontpath = "./gulim.ttc"
    boarding = 5
    font_size = int((i_s.shape[0] + int(i_s.shape[1] / len(data["contents"]))) / 2) - boarding
    font = freetype.Font(fontpath)
    font.antialiased = True
    font.origin = True
    font.size = font_size
    i_t = Image.new("RGB", (i_s.shape[1], i_s.shape[0]), (127, 127, 127))
    pre_remain = None
    padding = 0.1
    shape = (i_t.size[1], i_t.size[0])
    if padding < 1:
        border = int(min(shape) * padding)
    else:
        border = int(padding)
    target_shape = tuple(np.array(shape) - 2 * border)
    while True:
        rect = font.get_rect(data["translated_text"])
        res_shape = tuple(np.array(rect[1:3]))
        remain = np.min(np.array(target_shape) - np.array(res_shape))
        if pre_remain is not None:
            m = pre_remain * remain
            if m <= 0:
                if m < 0 and remain < 0:
                    font_size -= 1
                if m == 0 and remain != 0:
                    if remain < 0:
                        font_size -= 1
                    elif remain > 0:
                        font_size += 1
                break
        if remain < 0:
            if font_size == 2:
                break
            font_size -= 1
        else:
            font_size += 1
        pre_remain = remain
        font.size = font_size
    res_font = ImageFont.truetype(fontpath, font_size)
    draw = ImageDraw.Draw(i_t)
    w, h = draw.textsize(data["translated_text"], font=res_font)
    draw = ImageDraw.Draw(i_t)
    draw.text(((i_s.shape[1] - w) / 2, (i_s.shape[0] - h) / 2), data["translated_text"], font=ImageFont.truetype(fontpath, font_size), fill=(0, 0, 0))
    return i_t

def combining(img, o_f, data, j):
    img = Image.fromarray(np.uint8(img)).convert('RGBA')
    o_f = Image.fromarray(np.uint8(o_f)).convert('RGBA')
    line = data["textbox_{:d}".format(j)]["absolute_coord"]
    line = [int(i) for i in line[1:-1].split(" ")]
    line = np.array(line)

    thres = 0
    im_a = Image.new("L", o_f.size, 0)
    draw = ImageDraw.Draw(im_a)
    draw.rectangle((thres, thres, o_f.size[0]-thres, o_f.size[1]-thres), fill=255)
    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(3))
    o_f.putalpha(im_a_blur)

    sx, sy = o_f.size
    angle = -math.atan((int(line[3]) - int(line[1])) / (int(line[2]) - int(line[0]))) * 180 / math.pi
    o_f = o_f.rotate(angle, expand=1)
    loc_x = int(line[0])
    loc_y = int(line[1])
    if angle < 0:
        loc_x = int(loc_x - math.cos(np.deg2rad(90 - abs(angle))) * sy)
    if angle > 0:
        loc_y = int(loc_y - math.cos(np.deg2rad(90 - abs(angle))) * sx)
    sx, sy = o_f.size
    img.paste(o_f, (loc_x, loc_y, loc_x + sx, loc_y + sy), o_f)
    return img

def synthesizing(img_path, modi_path, label):
        # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # define model
    print_log('model compiling start.', content_color = PrintColor['yellow'])
    model = SRNet(shape = cfg.data_shape, name = 'predict')
    print_log('model compiled.', content_color = PrintColor['yellow'])
    pygame.init()
    freetype.init()
    with model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())

            # load pretrained weights
            print_log('weight loading start.', content_color = PrintColor['yellow'])
            saver.restore(sess, cfg.predict_ckpt_path)
            print_log('weight loaded.', content_color = PrintColor['yellow'])
            # predict
            print_log('predicting start.', content_color = PrintColor['yellow'])
            with open(label, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            listlen = len(os.listdir(img_path))
            for i in range(1, listlen):
                print("Synthesizing Test image {:d}/{:d}".format(i, listlen))
                img = cv2.imread(img_path+"clip_{:d}.png".format(i))
                for j in range(0, len(json_data["clip_{:d}.png".format(i)])):
                    if json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["IsTranslate?"] == True:
                        try :
                            i_s = cropping(img, json_data["clip_{:d}.png".format(i)],j)
                            i_t = np.array(masking(i_s, json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]))
                            ssim_score = 0
                            if i > 1:
                                rjt = 0
                                t_img = cv2.imread(img_path+"clip_{:d}.png".format(i-1))
                                for jt in range(0, len(json_data["clip_{:d}.png".format(i-1)])):
                                    it_s = cropping(t_img, json_data["clip_{:d}.png".format(i-1)],jt)
                                    t_ssim_score = comparing(i_s, it_s, json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)], json_data["clip_{:d}.png".format(i-1)]["textbox_{:d}".format(jt)])
                                    #print("clip_{:d}.png, textbox_{:d} Vs clip_{:d}.png, textbox_{:d}: ".format(i, j, i-1, jt), t_ssim_score)
                                    if t_ssim_score > ssim_score:
                                        ssim_score = t_ssim_score
                                        rjt = jt
                            if ssim_score > 0.75 and i>1:
                                fixed_img_num = json_data["clip_{:d}.png".format(i-1)]["textbox_{:d}".format(rjt)]["fixed_img_num"]
                                fixed_txb_num = json_data["clip_{:d}.png".format(i-1)]["textbox_{:d}".format(rjt)]["fixed_txb_num"]
                                fixed_t_img = cv2.imread(img_path+"clip_{:d}.png".format(fixed_img_num))
                                fixed_i_t= cropping(fixed_t_img, json_data["clip_{:d}.png".format(fixed_img_num)],fixed_txb_num)
                                ssim_score = comparing(i_s, fixed_i_t, json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)], json_data["clip_{:d}.png".format(fixed_img_num)]["textbox_{:d}".format(fixed_txb_num)])
                                if ssim_score > 0.75:
                                    json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["fixed_img_num"] = fixed_img_num
                                    json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["fixed_txb_num"] = fixed_txb_num
                                    m_img = cv2.imread(modi_path+"clip_{:d}.png".format(fixed_img_num))
                                    o_f = cropping(m_img, json_data["clip_{:d}.png".format(fixed_img_num)],fixed_txb_num)
                                    o_f = cv2.resize(o_f, (int(i_s.shape[1]), int(i_s.shape[0])), interpolation=cv2.INTER_AREA)
                                else:
                                    _, _, _, o_f = model.predict(sess, i_t, i_s)
                                    json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["fixed_img_num"] = i
                                    json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["fixed_txb_num"] = j
                            else:
                                _, _, _, o_f = model.predict(sess, i_t, i_s)
                                json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["fixed_img_num"] = i
                                json_data["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["fixed_txb_num"] = j
                            #o_f = averaging(o_f, i_s)
                            img = combining(img, o_f, json_data["clip_{:d}.png".format(i)], j)
                            img = np.array(img.convert("RGB"))
                        except:
                            print("clip_{:d}.png".format(i), "textbox_{:d}".format(j), "Error")
                cv2.imwrite(modi_path+"clip_{:d}.png".format(i), img)
            print_log('predicting finished.', content_color = PrintColor['yellow'])