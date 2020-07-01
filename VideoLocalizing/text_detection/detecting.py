import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import json
import glob
import cv2
import numpy as np
import math

from text_detection import craft_utils
from text_detection import imgproc
from text_detection.craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

""" For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def detection(imagefolder, label):

    # load net
    net = CRAFT()     # initialize
    trained_model = "models/detection_model.pth"
    print('Loading weights from checkpoint (' + trained_model  + ')')
    net.load_state_dict(copyStateDict(torch.load(trained_model )))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None

    t = time.time()

    # load data
    listlen = len(os.listdir(imagefolder))
    print(glob.glob(imagefolder))
    file_data = OrderedDict()

    for k in range(1, listlen):
        image_path = imagefolder + "clip_{}.png".format(k)
        print("detection Test image {:d}/{:d}".format(k+1, listlen))
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, False, refine_net)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        clipname = "clip_{}.png".format(k)
        file_data[clipname] = {}
        for i, box in enumerate(bboxes):
            absolute_coord = "[" + str(int(box[0][0])) + " " + str(int(box[0][1])) + " " + str(int(box[1][0])) + " " +\
                             str(int(box[1][1])) + " " + str(int(box[2][0])) + " " + str(int(box[2][1])) + " " +\
                             str(int(box[3][0])) + " " + str(int(box[3][1])) + "]"
            file_data[clipname]["textbox_{}".format(i)] = {}
            file_data[clipname]["textbox_{}".format(i)]['absolute_coord'] = absolute_coord

    with open(label, 'w', encoding="UTF-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")
