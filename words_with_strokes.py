import numpy as np
import json

from PIL import Image
import os
#################################################

def copy_files_Trace_test(path,dst_dir):
    import glob
    import shutil
    # import os

    print('Moving to destination')
    count = 0
    for name in glob.glob(path + '/*/*/*'):
        if name.endswith(".png"):
            count += 1

            shutil.copy(name, dst_dir)
            print('\t', name)
    print(count)

path = "/home/tug85766/Trace/data_processing/prepare_IAM_Lines/words"
dst_dir = "/home/tug85766/Trace/data_processing/prepare_IAM_Lines/Images_words"

copy_files_Trace_test(path, dst_dir)

