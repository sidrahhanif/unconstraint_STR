import os

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles
path_list =  GetFileFromThisRootDir("/home/tup21205/yolov5_obb-master/yolov5_obb-master/dataset/word/test_gt_east")
#print(len(path_list))
for path in path_list:
    filename = custombasename(path)
    fd = open(path, 'r')
    lines = fd.readlines()
    new_lines = []
    for line in lines:
        splitline= line.split(',')
        new_line = splitline[:6]+[splitline[6][1:]]+[splitline[7]]+['word','1']
        new_line = ' '.join(new_line)
        new_lines.append(new_line)
    fd.close()
    all_new_lines = '\n'.join(new_lines)
    f= open("/home/tup21205/yolov5_obb-master/yolov5_obb-master/dataset/word/Test/labelTxt/"+filename+".txt","w+")
    f.write(all_new_lines)
    f.close()
    print('done!')
