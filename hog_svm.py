import cv2
import os
import numpy as np
import json
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn import svm
import time
import joblib
import sys
import argparse


def repair_name(word):
    idx = len(word)-1
    while(word[idx] != '/'):
        idx -= 1
    idx += 1
    return word[idx:]

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
        bounding_boxes_actual.append([x_min,y_min,x_max-x_min,y_max-y_min])
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
    list_box = sorted(boxes,key =lambda x: ((x[2]*x[3])+x[4]) ,reverse=True)
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



def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


def scale_down(values, scale, count):
    val = 1.0
    for i in range(0,count):
        val = val * scale
    values[0] = values[0]/val
    values[1] = values[1]/val
    values[2] = values[2]/val
    values[3] = values[3]/val
    return values

def scale_up(values, scale, count):
    val = 1.0
    for i in range(0,count):
        val = val * scale
    values[0] = values[0]*val
    values[1] = values[1]*val
    values[2] = values[2]*val
    values[3] = values[3]*val
    return values


orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
(winW, winH) = (80,240)
windowSize = (winW,winH)
stepSize = 20
no_layers = 2



file_names = []
image_to_id = {}
list_bbox = {}
file_names.sort()
total = 0

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='.', help="path to dataset root directory")
parser.add_argument("--train", type=str, default='PennFudanPed_train.json', help="path to test json")
parser.add_argument("--test", type=str, default='PennFudanPed_val.json', help="path to test json")
parser.add_argument("--out", type=str, default='scikit_svm_output_file.json', help="path to output json")
parser.add_argument("--image_box", type=str, default='PennFudanPed/output', help="path to save ")
parser.add_argument("--model", default="./weights", help="path to trained SVM model")
args = parser.parse_args()

file1 = open(args.train)
data1 = json.load(file1)
for i in range(0,len(data1["images"])):
    file_names.append(data1["images"][i]["file_name"])
    image_to_id[data1["images"][i]["file_name"]]= data1["images"][i]["id"]
    list_bbox[data1["images"][i]["id"]]= []

for i in range(0,len(data1["annotations"])):
    list_bbox[data1["annotations"][i]["image_id"]].append(data1["annotations"][i]["bbox"])
    

positive_labels = []
negative_labels = []


for i in range(0,no_layers):
    positive_labels.append([])
    negative_labels.append([])


start = time.time()

for i in range(0,len(file_names)):
    img = cv2.imread(os.path.join(args.root,file_names[i]))
    result = list_bbox[image_to_id[file_names[i]]]
    
    count = 0
    for resize in pyramid_gaussian(img,max_layer=no_layers-1 ,downscale=1.1):
        for (x,y,window) in sliding_window(resize, stepSize=20, windowSize=(winW,winH)):
            store_res = [ miou_calculate(scale_down(rect1,1.2,count),[x,y,winW,winH]) for rect1 in result ]
            if window.shape != (winH,winW,3):
                continue
            if max(store_res) > 0.20:
                hog_val = hog(window,orientations=9,pixels_per_cell=(16,16),cells_per_block=(4,4))
                positive_labels[count].append(hog_val)
            else:
                hog_val = hog(window,orientations=9,pixels_per_cell=(16,16),cells_per_block=(4,4))
                negative_labels[count].append(hog_val)
        count = count + 1
    
y_train = []
x_train = []

for i in range(0,no_layers):
    y_train.append([])
    
for i in range(0,no_layers):
    for j in range(0,len(positive_labels[i])):
        y_train[i].append(1)
    for j in range(0,len(negative_labels[i])):
        y_train[i].append(0)

for i in range(0,no_layers):
    positive_labels[i] = np.array(positive_labels[i])
    negative_labels[i] = np.array(negative_labels[i])
    y_train[i] = np.array(y_train[i])
    if positive_labels[i].shape[0] != 0:
        x_train.append(np.concatenate((positive_labels[i],negative_labels[i])))
    else:
        x_train.append(negative_labels[i])      
clf = []
for i in range(0,no_layers):
    clf.append(svm.LinearSVC())
    clf[i].fit(x_train[i],y_train[i])
    joblib.dump(clf, os.path.join(args.model,f"svm_scikit_{i}"))
    
file_names_2 = []
file2 = open(args.test)
data2 = json.load(file2)
for i in range(0,len(data2["images"])):
    file_names_2.append(data2["images"][i]["file_name"])
    image_to_id[data2["images"][i]["file_name"]]= data2["images"][i]["id"]
 
dict_list = []
for i in range(0,len(file_names_2)):
    list_boxes = []
    img = cv2.imread(os.path.join(args.root,file_names_2[i]))
    store_img = img
    
    for resize in pyramid_gaussian(img,max_layer=no_layers-1 ,downscale=1.1):
        count = 0
        for (x,y,window) in sliding_window(resize, stepSize=20, windowSize=(winW,winH)):
            if window.shape != (winH,winW,3):
                continue
            hog_val = hog(window,orientations=9,pixels_per_cell=(16,16),cells_per_block=(4,4)).reshape(1,-1)
            val1 = np.dot(clf[count].coef_ , hog_val.T) + clf[count].intercept_
            idx = clf[count].predict(hog_val)
            if idx == 1 :
                store_arr = scale_up([x,y,winW,winH],1.2,count)
                store_arr.append(val1[0][0])
                list_boxes.append(store_arr)
        count = count+ 1
    list_boxes = non_max_supression(list_boxes,0.05)
    for (x,y,w,h,weigh) in list_boxes:
        dict_store = {}
        dict_store["bbox"] = [int(x),int(y),int(w),int(h)]
        dict_store["score"] = float(weigh)
        
        #May have to change
        dict_store["image_id"] = image_to_id[file_names_2[i]]
        dict_store["category_id"] = 1
        

        dict_list.append(dict_store)
        total = total + 1
        store_img = cv2.rectangle(store_img, (int(x),int(y)), (int(x)+winW,int(y)+winH), (255,0,0), 2)
    cv2.imwrite(os.path.join(args.image_box,repair_name(file_names_2[i])),img)
            

with open(args.out, 'w') as fout:
    json.dump(dict_list , fout)
