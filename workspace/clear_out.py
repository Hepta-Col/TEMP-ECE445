import os
import shutil

path = "../out"
for dir in os.listdir(path):
    shutil.rmtree(os.path.join(path, dir))