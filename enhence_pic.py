# -*- coding:gb2312 -*-
import cv2
from math import *
import numpy as np
import os
def rotate_img(img_path, degree):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    # print(matRotation)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    image = cv2.resize(imgRotation, (112, 112))
    # if (degree <= 0) & (degree >= -20):
    #     a = 112/(1+2*tan(radians(abs(degree))))
    #     b = (112-a)/2
    #     image = image[int(b):int(a+b), int(b):int(a+b)]
    # elif (degree > 0) & (degree <= 20):
    #     b = 112/(tan(radians(degree))+1/(tan(radians(degree)))+2)
    #     a = 112 - 2*b
    #     image = image[int(b):int(a+b), int(b):int(a+b)]
    return image

def main():
    train_data_dir = "H:/why_workspace/ReCTS/img_dir5/"
    count = 0
    for root, dirs, files in os.walk(train_data_dir):
        if count == 0:
            count += 1
            continue
        print(count)
        #print(root[31:35])
        os.mkdir("H:/why_workspace/ReCTS/img_dir5_enhence/" + str(root[32:36]))
        for file in files:
            for degree in range(-20, 21):
                output_path = "H:/why_workspace/ReCTS/img_dir5_enhence/"+str(root[32:36])+'/'+os.path.splitext(file)[0]+'_'+str(degree)+'.png'
                img = rotate_img(os.path.join(root, file), degree)
                print(output_path)
                cv2.imwrite(output_path, img)

if __name__ == '__main__':
    main()
