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

def miou_calculate(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_top = max(bb1[1],bb2[1])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

def non_max_supression(boxes,overlapThreshold):
    if (len(boxes)==0):
        return []
    final_boxes = []
    list_box = sorted(boxes,key =lambda x: x[2]*x[3] ,reverse=True)
    list_box = np.array(list_box)
    
    list_indices = []
    for i in range(0,len(list_box)):
        list_indices.append(True)
    
    for i in range(0,len(list_box)):
        for j in range(i+1,len(list_box)):
            if miou_calculate(list_box[i],list_box[j]) > overlapThreshold:
               #Remove block j from list of allowed positive
                list_indices[j] = False
    
    for i in range(0,len(list_box)):
        if list_indices[i] :
            final_boxes.append(list_box[i])
    
    final_boxes = np.array(final_boxes)
    return final_boxes


file_names = [repair_name(name) for name in os.listdir("PennFudanPed/PNGImages") if os.path.isfile(os.path.join("PennFudanPed/PNGImages", name))]
no_images = len(file_names)    
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.save("weights_cv2_model")
total = 0    

for i in range(0,no_images):
    img = cv2.imread("PennFudanPed/PNGImages/" + file_names[i] +".png")
    result = read_annotation("PennFudanPed/Annotation/"+ file_names[i] +".txt")
    (rects, weights) = hog.detectMultiScale(img)
    rect_fin = non_max_supression(rects,0.2)
    
    total = total + len(rect_fin)
    for (x,y,w,h) in rect_fin:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite("PennFudanPed/output/img"+file_names[i]+".png",img)
    
print(total)
