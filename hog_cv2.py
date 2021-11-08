import cv2
import os
import numpy as np
import json

def repair_name(word):
    return word[0:len(word)-4]

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
    list_box = sorted(boxes,key =lambda x: (x[2]+1)*(x[3]+1) ,reverse=True)
    list_box = np.array(list_box)
    
    list_indices = []
    for i in range(0,len(list_box)):
        list_indices.append(True)
    
    for i in range(0,len(list_box)):
        if list_indices[i] == False :
            continue
        for j in range(i+1,len(list_box)):
            if miou_calculate(list_box[i],list_box[j]) > overlapThreshold:
               #Remove block j from list of allowed positive
                list_indices[j] = False
    
    for i in range(0,len(list_box)):
        if list_indices[i] :
            final_boxes.append(list_box[i])
    
    final_boxes = np.array(final_boxes)
    return final_boxes


file_names = []
image_to_id = {}
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hog.save("weights_cv2_model")
total = 0    

file1 = open("pedestrian_detection/PennFudanPed_val.json")
data1 = json.load(file1)
for i in range(0,len(data1["images"])):
    file_names.append(data1["images"][i]["file_name"])
    image_to_id[data1["images"][i]["file_name"]]= data1["images"][i]["id"]

no_images = len(file_names)    
dict_list = []


for i in range(0,no_images):
    img = cv2.imread(file_names[i])
    (rects, weights) = hog.detectMultiScale(img)
    rect_fin = non_max_supression(rects,0.2)

    for ((x,y,w,h),weigh) in zip(rects,weights):
        if (x,y,w,h) not in rect_fin:
            continue
        dict_store = {}
        dict_store["bbox"] = [int(x),int(y),int(w),int(h)]
        dict_store["score"] = float(weigh)
        
        #May have to change
        dict_store["image_id"] = image_to_id[file_names[i]]
        dict_store["category_id"] = 1
        

        dict_list.append(dict_store)
        total = total + 1
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        

with open('output_file.json', 'w') as fout:
    json.dump(dict_list , fout)
    
