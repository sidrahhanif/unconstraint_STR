from svgwrite import Drawing
import numpy as np


Coordinates = np.load('/home/tug85766/Trace/example_weights/results/imgs/current/eval/11_0,0.npy', allow_pickle = True).tolist()
count =0
for i in range(len(Coordinates)):
    Coordinates[i][0] = Coordinates[i][0] * 61
    Coordinates[i][1] = 61 - (Coordinates[i][1] * 61)
    if Coordinates[i][2] == 1:
        count += 1
p, p1, Line = [], [], []
k = 0
for i in range(count):
    if Coordinates[k][2] == 1:
        p = [Coordinates[k][0], Coordinates[k][1]]

        if Coordinates[k+1][2] == 1:
            p1 = [[Coordinates[k][0]+1, Coordinates[k][1]+1]]
            #Line.append([p,p1])
            p1, p =[], []
            k += 1
        else:
            k += 1
            while (Coordinates[k][2] != 1):
                p1.append([Coordinates[k][0], Coordinates[k][1]])
                k +=1
                if k == len(Coordinates):
                    break
            Line.append([p, p1])
            p1, p = [], []
w = 600
h = 61

line_width = 3
w_str = "{}pt".format(w)
h_str = "{}pt".format(h)
fn = 'example.svg'

dwg = Drawing(filename=fn,
              size=(w_str, h_str),
              viewBox=("0 0 {} {}".format(w, h)))
paths = Line
for path in paths:
    print(path)
    print(len(path))
    print(path[0][0], path[0][1])
    if (len(path) > 1):
        str_listM = []
        str_listM.append("M {},{}".format(path[0][0], path[0][1]))
        str_listC = []
        for e in path[1]:
            print(e)
            if str_listC == []:
                str_listC.append(" L {},{}".format(e[0], e[1]))
            else:
                str_listC.append(" {},{}".format(e[0], e[1]))
        s = ''.join(str_listC)
        s = str_listM[0] + s
        dwg.add(dwg.path(s).stroke(color="rgb(0%,0%,0%)", width=line_width).fill("none"))

dwg.save()