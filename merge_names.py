import glob
import random

files = glob.glob ('data/custom/images/*.jpg')
files += glob.glob ('data/custom/images/*.JPG')
train=int(len(files)*4)//5
random.shuffle (files)
#int val=len(files)-train
files_train=files[:train]
files_val=files[train:]

with open ('data/custom/train.txt', 'w') as in_files:
    for eachfile in files_train:
        in_files.write(eachfile+'\n')

with open ('data/custom/val.txt', 'w') as in_files:
    for eachfile in files_val:
        in_files.write(eachfile+'\n')
