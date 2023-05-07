import os
import shutil

path = "../out"
for dir in os.listdir(path):
    dir_path = os.path.join(path, dir)
    flag = False
    for p, dir_list, file_list in os.walk(dir_path):  
        for file_name in file_list:  
            if "DONE" in file_name:
                flag = True
    if flag is False:         
        shutil.rmtree(dir_path)