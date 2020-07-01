import cv2
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os
import json
import time
import operator

from collections import OrderedDict
from google.cloud import translate_v2 as translate

def LS(s1, s2):
    if len(s1) < len(s2):
        return LS(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return (len(s1) + len(s2) - previous_row[-1]) / (len(s1) + len(s2))


def BoxSimilarity(label):
    with open(label, encoding='utf-8') as json_file:
        f = json.load(json_file)
    checkLength = 10
    count = 0
    simPer = 0.6
    checkFrameCount = 5
    client = translate.Client()

    # 모든 Clip 확인
    for i in range(1, len(f) + 1):
        print("Test image {:d}/{:d}".format(i, len(f) - 1))
        # Clip 안의 모든 textbox 확인
        for j in range(len(f["clip_{:d}.png".format(i)])):
            # 이미 전 clip에서 유사도 측정이 완료되서 번역이 완료된 것은 패스
            if f["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]["Check"] == False:
                # 같은 이미지 유사도 imageList에 저장
                imageList = [[i, j]]
                # absolute_coord 사용
                line = [int(c) for c in f["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]['absolute_coord'][1:-1].split(' ')]
                txt1 = f["clip_{:d}.png".format(i)]["textbox_{:d}".format(j)]['contents']
                # 다음 프레임부터 비슷한 프레임이 있으면 모든 프레임 검사
                for k in range(i + 1, len(f)):
                    check = False
                    # 각 프레임 마다 txtbox 검사
                    for l in range(len(f["clip_{:d}.png".format(k)])):
                        # 이미 전 clip에서 유사도 측정이 완료되서 번역이 완료된 것은 패스
                        if f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["Check"] == False:
                            line2 = [int(c) for c in f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]['absolute_coord'][1:-1].split(' ')]
                            # 범위 안에 드는지 조건 확인
                            if line2[0] - checkLength < line[0] < line2[0] + checkLength and line2[1] - checkLength < \
                                line[1] < line2[1] + checkLength and line2[2] - checkLength < line[2] < line2[2] + \
                                checkLength and line2[5] - checkLength < line[5] < line2[5] + checkLength:
                                
                                txt2 = f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]['contents']
                                # 유사도가 존재하면 다음 이미지도 실행을 한다.
                                if LS(txt1, txt2) > simPer:
                                    check = True
                                    txt1 = txt2
                                    # 다음 프레임부터는 최근 프레임의 txtbox를 사용하기 위해서 수정한다.
                                    line = line2
                                    imageList.append([k, l])
                                    break

                    ## 다음 프레임에서 유사도를 찾지 못하더라도 한번 더 다음 프레임 검사를 진행한다.
                    if check == False:
                        for k2 in range(k + 1, min(k + checkFrameCount, len(f) - 2)):
                            for l in range(len(f["clip_{:d}.png".format(k2)])):
                                if f["clip_{:d}.png".format(k2)]["textbox_{:d}".format(l)]["Check"] == False:
                                    line2 = [int(c) for c in f["clip_{:d}.png".format(k2)]["textbox_{:d}".format(l)]['absolute_coord'][1:-1].split(' ')]
                                    # 범위 안에 드는지 조건 확인
                                    if line2[0] - checkLength < line[0] < line2[0] + checkLength and line2[
                                        1] - checkLength < \
                                            line[1] < line2[1] + checkLength and line2[2] - checkLength < line[2] < \
                                            line2[2] + \
                                            checkLength and line2[5] - checkLength < line[5] < line2[5] + checkLength:
                                        # 유사도 측정                                     
                                        txt2 = f["clip_{:d}.png".format(k2)]["textbox_{:d}".format(l)]['contents']

                                        # 추가로 검색한 프레임에 유사도가 있는 박스를 찾으면 실행
                                        if LS(txt1, txt2) > simPer:
                                            check = True
                                            line = line2
                                            txt1 = txt2
                                            imageList.append([k2, l])
                                            for k3 in range(k, k2):
                                                p = len(f["clip_{:d}.png".format(k3)])
                                                imageList.append([k3, p])
                                                # 박스 는 이전박스와 다음 박스의 중간 사이즈로 진행, 그리고 다음 의 텍스트를 가져와서 임의대로 삽입
                                                absolute_coord = "[" + str(int((line[0] + line2[0])/2)) + " " + \
                                                             str(int((line[1] + line2[1])/2)) + " " + \
                                                             str(int((line[2] + line2[2])/2)) + " "+ \
                                                             str(int((line[3] + line2[3])/2)) + " "+ \
                                                             str(int((line[4] + line2[4])/2)) + " "+ \
                                                             str(int((line[5] + line2[5])/2)) + " "+ \
                                                             str(int((line[6] + line2[6])/2)) + " "+ \
                                                             str(int((line[7] + line2[7])/2)) + "]"
                                                f["clip_{:d}.png".format(k3)]["textbox_{:d}".format(p)] = {}
                                                f["clip_{:d}.png".format(k3)]["textbox_{:d}".format(p)]['absolute_coord'] = absolute_coord
                                                f["clip_{:d}.png".format(k3)]["textbox_{:d}".format(p)]['contents'] = f["clip_{:d}.png".format(k2)]["textbox_{:d}".format(l)]['contents']
                                                f["clip_{:d}.png".format(k3)]["textbox_{:d}".format(p)]["Check"] = True
                                                break
                        # 한번 더 검사를 진행해서 찾았을 경우
                        if check:
                            k = k + 2
                        # 한번 더 검사를 진행해서 찾지 못했을 경우
                        else:
                            break

                ## 이미지 유사도 전부 확인 번역하는 과정


                TextList = {}
                # 비슷한 이미지의 모든 Contetns를 검사해서 Dic에 저장한다.
                for k, l in imageList:
                    if not f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["contents"] in TextList:
                        TextList[f["clip_{:d}.png".format(k )]["textbox_{:d}".format(l)]["contents"]] = 1
                    else:
                        TextList[f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["contents"]] += 1

                # 가장 많이 나온 값을 가져오기기
                besttext = ""
                besttextnum = 0
                for text in TextList:
                    if TextList[text] > besttextnum:
                        besttext = text
                        besttextnum = TextList[text]

                besttext = besttext[:-1]
                # 가장 많이 나온 besttext를 번역한다
                result = client.translate(besttext, target_language='ko')

                # 구글 번역기를 사용한 횟수를 체크
                count += 1

                # 비슷한 모든 이미지의 텍스트를 하나로 수정을 한다.
                # 번역을 진행했으므로 Check를 True로 해서 추가로 번역하는 일을 방지한다.
                # 번역을 진행해도 영어인 경우에는 이미지 수정을 하지 않는다 -> Synthesizing에서 사용
                for k, l in imageList:
                    f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]['contents'] = besttext
                    f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["translated_text"] = result['translatedText']
                    f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["Check"] = True
                    if besttext == result['translatedText']:
                        f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["IsTranslate?"] = False
                    else:
                        f["clip_{:d}.png".format(k)]["textbox_{:d}".format(l)]["IsTranslate?"] = True


    # 마지막 labels 저장하기
    with open(label, 'w', encoding="UTF-8") as make_file:
        json.dump(f, make_file, ensure_ascii=False, indent="\t")

    # 번역을 한 횟수만큼 리턴
    return count