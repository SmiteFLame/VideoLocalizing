import torch
from torch.autograd import Variable
import text_recognition.utils as utils
import text_recognition.dataset as dataset
from PIL import Image
import numpy as np
import cv2
import math
from collections import OrderedDict
from text_recognition.moran import MORAN
import os
import json


def recognition(imagefolder, label):
    global MORAN

    credential_path = "VideoLocalizing-fdee194f2156.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
    model_path = 'models/recognition_model.pth'
    cuda_flag = False
    if torch.cuda.is_available():
        cuda_flag = True
        MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=cuda_flag)
        MORAN = MORAN.cuda()
    else:
        MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.intTensor',
                      CUDA=cuda_flag)

    if cuda_flag:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        MORAN_state_dict_rename[name] = v
    MORAN.load_state_dict(MORAN_state_dict_rename)

    for p in MORAN.parameters():
        p.requires_grad = False
    MORAN.eval()

    print("recognition")

    converter = utils.strLabelConverterForAttention(alphabet, ':')
    transformer = dataset.resizeNormalize((100, 32))

    listlen = len(os.listdir(imagefolder))

    with open(label) as json_file:
        f = json.load(json_file)

    for k in range(1, listlen):
        print("Recognition Test image {:d}/{:d}".format(k, listlen))
        img = "clip_{}.png".format(k)
        imgL = Image.open(imagefolder + img).convert('L')
        lines = f[img]

        # 텍스트는 기존 박스 방식 이 훨씬 좋으므로 합치기 전 박스에서 실행을 한다.
        for line in lines:
            if line != "imageSimilarity":
                textbox = line
                image = cropping(imgL, f[img][line]["absolute_coord"])
                image = transformer(image)
                if cuda_flag:
                    image = image.cuda()

                image = image.view(1, *image.size())
                image = Variable(image)
                text = torch.LongTensor(1 * 5)
                length = torch.IntTensor(1)
                text = Variable(text)
                length = Variable(length)

                max_iter = 20
                t, l = converter.encode('0' * max_iter)
                utils.loadData(text, t)
                utils.loadData(length, l)
                output = MORAN(image, length, text, text, test=True, debug=True)

                preds, preds_reverse = output[0]
                demo = output[1]

                _, preds = preds.max(1)
                _, preds_reverse = preds_reverse.max(1)

                sim_preds = converter.decode(preds.data, length.data)
                sim_preds = sim_preds.strip().split('$')[0]

                # 번역

                translated_text = sim_preds
                f[img][textbox]["contents"] = sim_preds


    with open(label, 'w', encoding="UTF-8") as make_file:
        json.dump(f, make_file, ensure_ascii=False, indent="\t")

def cropping(img, line):
    img = np.array(img)
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
    return Image.fromarray(img_crop).convert('L')