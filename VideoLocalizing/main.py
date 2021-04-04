import os
import sys
import shutil
import cv2
import time
import numpy as np

from text_synthesizing import synthesizing
from text_detection import detecting
from text_recognition import recognition
from utils import CombineCrop
from utils import ImageSimilarity

ori_img = os.getcwd() + "/images/"
modi_img = os.getcwd() + "/modification_images/"
label = os.getcwd() + "/labels.json"
label2 = os.getcwd() + "/labels_s5.json"
videoroot = os.getcwd() + "/data/Sample5.mp4"

for file in os.scandir(ori_img):
    os.remove(file.path)
for file in os.scandir(modi_img):
    os.remove(file.path)
if os.path.isfile(label):
    os.remove(label)


os.system('ffmpeg -ss {:d} -i "{:s}" -r {:d} -f image2 .\images\clip_%d.png'.format(0, videoroot, 30))

S2time = time.time()
print("detection")
detecting.detection(ori_img, label)
S3time = time.time()
print("recognition")
recognition.recognition(ori_img, label)
S4time = time.time()
print("CombineBox")
CombineCrop.CombineBox(label)
S5time = time.time()
print("BoxSimilarity")
count = ImageSimilarity.BoxSimilarity(label)
S6time = time.time()
print("synthesizing")
synthesizing.synthesizing(ori_img, modi_img, label2)
S7time = time.time()
print("Detection : ", round(S3time - S2time, 1), "s")
print("Recognition : ", round(S4time - S3time, 1), "s")
print("CombineBox : ", round(S5time - S4time, 3), "s")
print("Image Similarity : ", round(S6time - S5time, 1), "s")
#print("Translation Count : ", count)
print("Synthesizing : ", round(S7time - S6time, 1), "s")
print("Full time : ", round(S7time - S2time, 1), "s")

os.system(r'ffmpeg -r {:d} -f image2 -i .\modification_images\clip_%d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p "{:s}"'.format(30, modi_img+"output.mp4"))
os.system('ffmpeg -i "{:s}" -i "{:s}" -c:v copy -c:a aac "{:s}"'.format(modi_img+"output.mp4", videoroot, os.path.splitext(videoroot)[0] + "_result.mp4"))

#for file in os.scandir(ori_img):
#    os.remove(file.path)
#for file in os.scandir(modi_img):
#    os.remove(file.path)
#os.remove(label)
