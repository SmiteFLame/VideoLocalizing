# VideoLocalizing
A program that automatically translates text in a video.

## Introduction
We made it with [Dong-Hyeok Yang](https://github.com/smiteflame), [Min-ho Kim](https://github.com/JadeBright), Young-Joon Seo, Ji-Yeoop Na

We used a deep learning model to create an automatic translation system.

It was difficult to develop a deep learning model with a master's degree, so I made a program using a different model.

The process of merging the text box size and the image similarity test algorithm were developed and added directly.

## Requirements
```
- python == 3.6
- CUDA 10.x
- cuDNN 7.6.5
- tensorflow-gpu==1.14.0
- pytorch
- opencv-python == 3.4.2.17
- PyQt5==5.14.2
- scikit-image==0.14.2
- scipy==1.1.0
- lmdb==0.98
- imutils==0.5.3
- google-cloud-translate==2.0.1
  (If you want, change translation API)
```

## Additional Installations
```
ffmpng
google-cloud-translate JSON file
k-lite-codec

```

## Translation
Added Google translation system to "image similarity" file. If you want to use a different translation system, you can change it.

And each parameter is set at the top of each program, so it can be changed and used.

## Download Model
You can download [model](https://drive.google.com/drive/folders/1GULPGHU9DUq-HH5kVK2a6hcxmigddstV?usp=sharing) about detection, recognition and synthesizing.

Input model in folder named model. If not, create a "model" file.

## Reference
Detection [CRAFT](https://github.com/clovaai/CRAFT-pytorch)

Recognition [MORAN](https://github.com/Canjie-Luo/MORAN_v2)

Synthesizing [SRNET](https://github.com/youdao-ai/SRNet)



## Result
![image](https://github.com/SmiteFLame/VideoLocalizing/blob/master/img/image/clip_1.png)
![image](https://github.com/SmiteFLame/VideoLocalizing/blob/master/img/mod/clip_1.png)

![image](https://github.com/SmiteFLame/VideoLocalizing/blob/master/img/image/clip_2.png)
![image](https://github.com/SmiteFLame/VideoLocalizing/blob/master/img/mod/clip_2.png)

![image](https://github.com/SmiteFLame/VideoLocalizing/blob/master/img/image/clip_3.png)
![image](https://github.com/SmiteFLame/VideoLocalizing/blob/master/img/mod/clip_3.png)
