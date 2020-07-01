import json
from collections import OrderedDict

def CombineBox(label):
    with open(label, encoding='utf-8') as json_file:
        f = json.load(json_file)

    file_data = OrderedDict()
    for file in f:
        print("Combine", file)
        file_data[file] = {}
        list = []
        for line in f[file]:
            list.append([int(i) for i in f[file][line]['absolute_coord'][1:-1].split(' ')])
        ndir = {i : i for i in range(len(list))}
        CheckOri = len(list)
        Check = len(list)
        Check2 = 0

        while Check != Check2:
            Check2 = Check
            Check = 0
            for i in range(len(list)):
                if ndir[i] == i:
                    Check += 1
                    paddingX = int((max(int(list[i][2]), int(list[i][4])) - min(int(list[i][0]), int(list[i][6]))) / 10)
                    startX = min(int(list[i][0]), int(list[i][6])) - paddingX
                    endX = max(int(list[i][2]), int(list[i][4])) + paddingX
                    startY = min(int(list[i][1]), int(list[i][3]))
                    endY = max(int(list[i][5]), int(list[i][7]))
                    for j in range(len(list)):
                        if ndir[j] == j and i != j:
                            paddingX2 = int((max(int(list[j][2]), int(list[j][4])) - min(int(list[j][0]), int(list[j][6]))) / 10)
                            startX2 = min(int(list[j][0]), int(list[j][6])) - paddingX2
                            endX2 = max(int(list[j][2]), int(list[j][4])) + paddingX2
                            startY2 = min(int(list[j][1]), int(list[j][3]))
                            endY2 = max(int(list[j][5]), int(list[j][7]))
                            for k in range(len(list)):
                                if ndir[k] == j and j != k:
                                    startX2 = min(startX2, int(list[k][0]), int(list[k][6]))
                                    endX2 = max(endX2, int(list[k][2]), int(list[k][4]))
                                    startY2 = min(startY2, int(list[k][1]), int(list[k][3]))
                                    endY2 = min(endY2, int(list[k][5]), int(list[j][7]))

                            if startX2 < endX and abs(startY - startY2) < 20 and endX2 > startX and abs(endY - endY2) < 20:
                                if startX < startX2:
                                    list[i][0] = startX
                                    list[i][6] = startX
                                    ndir[j] = i
                                else:
                                    ndir[i] = j
                                if startY < startY2:
                                    list[i][1] = startY
                                    list[i][3] = startY
                                if endX > endX2:
                                    list[i][2] = endX
                                    list[i][4] = endX
                                if endY > endY2:
                                    list[i][5] = endY
                                    list[i][7] = endY
        boxnum = 0
        for x in range(CheckOri):
            if ndir[x] == x:
                num = []
                num.append(x)
                for y in range(CheckOri):
                    if ndir[y] == x and x != y:
                        num.append(y)
                MaX = 0
                MiX = 9223372036854775807
                MaY = 0
                MiY = 9223372036854775807
                Oricontents = ""

                for y in num:
               
                    MaX = max([MaX, int(list[y][0]), int(list[y][2]), int(list[y][4]), int(list[y][6])])
                    MiX = min([MiX, int(list[y][0]), int(list[y][2]), int(list[y][4]), int(list[y][6])])
                    MaY = max([MaY, int(list[y][1]), int(list[y][3]), int(list[y][5]), int(list[y][7])])
                    MiY = min([MiY, int(list[y][1]), int(list[y][3]), int(list[y][5]), int(list[y][7])])
                    Oricontents += f[file]["textbox_{}".format(y)]['contents'] + " "

             
                file_data[file]["textbox_{}".format(boxnum)] = {}
                file_data[file]["textbox_{}".format(boxnum)]["absolute_coord"] = "[" + str(MiX) + " " + str(MiY) + " " \
                 + str(MaX) + " " + str(MiY) + " " + str(MaX) + " " + str(MaY) + " " + str(MiX) + " " + str(MaY) + "]"
                file_data[file]["textbox_{}".format(boxnum)]['contents'] = Oricontents
                file_data[file]["textbox_{}".format(boxnum)]['Check'] = False
                boxnum += 1

    with open(label, 'w', encoding="UTF-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")


if __name__ == "__main__":
    print("main")