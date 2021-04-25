#!/usr/bin/env python
# coding: utf-8

# In[1]:

import math
import os
import glob
import cv2
import random
import numpy as np
from tqdm import tqdm


# In[2]:


def brightness(img, value):
    #value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img
def rotation(img, angle):
    #angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def rotation_matrix(x, y, angle):
    rad = math.radians(angle)
    rx = x * math.cos(rad) - y * math.sin(rad)
    ry = x * math.sin(rad) + y * math.cos(rad)
    return rx, ry

star = 'Sagittarius'
source_Image_Folder = 'org/Image/' + star + '/'
source_Label_Folder = 'org/Label/' + star + '/'
target_Image_Folder = 'new/Image/' + star + '/'
target_Label_Folder = 'new/Label/' + star + '/'

### rotate * 3
num_angle_aug = 3
for imgpath in tqdm(glob.glob(source_Image_Folder + '*')):
    name = imgpath.split('/', 4)[3].split('.')[0]
    anglist = []
    for i in range(num_angle_aug):
        angle = 10
        angle = int(random.uniform(-angle, angle))
        while angle == 0 or angle in anglist:
            angle = 10
            angle = int(random.uniform(-angle, angle))
        anglist.append(angle)
    anglist.append(0)
    for angle in anglist:
        flag = True
        path = source_Label_Folder + name + '.txt'
        infile = open(path, 'r')
        for line in infile:
            data = line.split(' ')
            x, y = float(data[1]) - 0.5, float(data[2]) - 0.5
            x, y = rotation_matrix(x, y, angle * (-1))
            x, y = x + 0.5, y + 0.5
            if x > 1 or x < 0 or y > 1 or y < 0:
                flag = False
                break
            outfile = open(target_Label_Folder + name + '_rotation' + str(angle) + '.txt', 'w')
            data[1], data[2] = str(x)[:8], str(y)[:8]
            outfile.write(' '.join(data))
            outfile.close()
        infile.close()
        if flag:
            img = cv2.imread(imgpath)
            img_r = rotation(img, angle)
            cv2.imwrite(target_Image_Folder + name + '_rotation' + str(angle) + '.jpg', img_r)

### brightness * 4
num_brightness_aug = 4
for imgpath in tqdm(glob.glob(target_Image_Folder + '*')):
    img = cv2.imread(imgpath)
    name = imgpath.split('/', 4)[3].split('.')[0]
    valist = []
    for i in range(num_brightness_aug):
        if i % 2 == 0:
            low, high = 0.4, 0.8
        else:
            low, high = 1.2, 1.6
        value = random.uniform(low, high)
        valist.append(value)
    for value in valist:
        img_r = brightness(img, value)
        cv2.imwrite(target_Image_Folder + name + '_brightness' + str(value)[:5] + '.jpg', img_r)
        path = target_Label_Folder + name + '.txt'
        infile = open(path, 'r')
        for line in infile:
            outfile = open(target_Label_Folder + name + '_brightness' + str(value)[:5] + '.txt', 'w')
            outfile.write(line)
            outfile.close()
        infile.close()

### horizontal_flip & vertical_flip
for imgpath in tqdm(glob.glob(target_Image_Folder + '*')):
    img = cv2.imread(imgpath)
    name =  imgpath.split('/', 4)[3].split('.')[0]
    img_r = horizontal_flip(img, True)
    cv2.imwrite(target_Image_Folder + name + '_horizontal' + '.jpg', img_r)
    img_r = vertical_flip(img, True)
    cv2.imwrite(target_Image_Folder + name + '_vertical' + '.jpg', img_r)
    path = target_Label_Folder + name + '.txt'
    infile = open(path, 'r')
    for line in infile:
        data = line.split(' ')
        x = float(data[1]) - 0.5
        x = x * (-1)
        x = x + 0.5
        outfile = open(target_Label_Folder + name + '_horizontal' + '.txt', 'w')
        data[1] = str(x)
        outfile.write(' '.join(data))
        outfile.close()
        data = line.split(' ')
        y = float(data[2]) - 0.5
        y = y * (-1)
        y = y + 0.5
        outfile = open(target_Label_Folder + name + '_vertical' + '.txt', 'w')
        data[2] = str(y)[:8]
        outfile.write(' '.join(data))
        outfile.close()
    infile.close()


#img = rotation(img, 30)    
#img = vertical_flip(img, True)
#img = horizontal_flip(img, True)
#img = brightness(img, 0.5, 1.5)


# In[20]:


# k=0
# for path in glob.glob("./string_450/*.jpg"):
#     img = cv2.imread(path)
#     #print (os.path.split(path)[1])
    
#     #rotation
#     img_r = rotation(img, 30)    
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    
#     #vertical flip
#     img_r = vertical_flip(img, True)   
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    
#     #horizontal flip
#     img_r = horizontal_flip(img, True)
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    
#     #brightness
#     img_r = brightness(img, 0.5, 1.5)   
#     cv2.imwrite('./string/string_'+str(k)+'.jpg',img_r)
#     k+=1
    

