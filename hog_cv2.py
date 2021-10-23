import cv2
import os
import numpy as np

def repair_name(word):
    return word[0:len(word)-4]

def read_annotation(file):
    bounding_boxes_actual = []
    f1 = open(file)
    lines = f1.readlines()
    lines = lines[8:]
    while(len(lines)!=0):
        line_imp = lines[2]
        words = line_imp.split()
        x_min = int(words[12][1:len(words[12])-1])
        x_max = int(words[15][1:len(words[15])-1])
        y_min = int(words[13][:len(words[13])-1])
        y_max = int(words[16][:len(words[16])-1])
        bounding_boxes_actual.append([x_min,y_min,x_max,y_max])
        lines = lines[5:]
    
    return bounding_boxes_actual
    
file_names = [repair_name(name) for name in os.listdir("PennFudanPed/PNGImages") if os.path.isfile(os.path.join("PennFudanPed/PNGImages", name))]
no_images = len(file_names)    
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
total = 0    

for i in range(0,no_images):
    img = cv2.imread("PennFudanPed/PNGImages/" + file_names[i] +".png")
    result = read_annotation("PennFudanPed/Annotation/"+ file_names[i] +".txt")
    (rects, weights) = hog.detectMultiScale(img)
    total = total + len(rects)
    for (x,y,w,h) in rects:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite("PennFudanPed/output/img"+file_names[i]+".png",img)
    
print(total)
