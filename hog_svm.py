import cv2
import os
import numpy as np
import json
from skimage.feature import hog
from sklearn import svm
import time
import joblib

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
    list_box = sorted(boxes,key =lambda x: (x[4]) ,reverse=True)
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


def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
(winW, winH)= (64,128)
windowSize=(winW,winH)
stepSize=20

file_names = [repair_name(name) for name in os.listdir("PennFudanPed/PNGImages") if os.path.isfile(os.path.join("PennFudanPed/PNGImages", name))]
no_images = len(file_names) 
total = 0


positive_labels = []
negative_labels = []


start = time.time()

for i in range(0,no_images):
    img = cv2.imread("PennFudanPed/PNGImages/" + file_names[i] +".png")
    store_img = img
    result = read_annotation("PennFudanPed/Annotation/"+ file_names[i] +".txt")
    
    for (x,y,window) in sliding_window(img, stepSize=20, windowSize=(winW,winH)):
        store_img = cv2.rectangle(store_img,(x,y),(x+winW,y+winH),(255,0,0),2)
        store_res = [ miou_calculate(rect1,[x,y,winW,winH]) for rect1 in result ]
        if window.shape != (winH,winW,3):
            continue
        if max(store_res) > 0.25:
            hog_val = hog(window,orientations=9,pixels_per_cell=(16,16),cells_per_block=(4,4))
            positive_labels.append(hog_val)
        else:
            hog_val = hog(window,orientations=9,pixels_per_cell=(16,16),cells_per_block=(4,4))
            negative_labels.append(hog_val)
     
print("No of positive labels = :",len(positive_labels))
print("No of negative labels = :",len(negative_labels))

y_train = []
for i in range(0,len(positive_labels)):
    y_train.append(1)
for i in range(0,len(negative_labels)):
    y_train.append(0)


positive_labels = np.array(positive_labels)
negative_labels = np.array(negative_labels)
y_train = np.array(y_train)
x_train = np.concatenate((positive_labels,negative_labels))

clf = svm.LinearSVC()
clf.fit(x_train,y_train)
joblib.dump(clf, "svm_scikit")

dict_list = []
for i in range(0,no_images):
    list_boxes = []
    img = cv2.imread("PennFudanPed/PNGImages/" + file_names[i] +".png")
    store_img = img
    result = read_annotation("PennFudanPed/Annotation/"+ file_names[i] +".txt")
    
    for (x,y,window) in sliding_window(img, stepSize=20, windowSize=(winW,winH)):
        store_res = [ miou_calculate(rect1,[x,y,winW,winH]) for rect1 in result ]
        if window.shape != (winH,winW,3):
            continue
        hog_val = hog(window,orientations=9,pixels_per_cell=(16,16),cells_per_block=(4,4)).reshape(1,-1)
        val1 = np.dot(clf.coef_ ,  hog_val.T) + clf.intercept_
        idx = clf.predict(hog_val)
        if idx == 1 :
            list_boxes.append([x,y,winW,winH,val1[0][0]])
    list_boxes = non_max_supression(list_boxes,0.10)
    for (x,y,w,h,weigh) in list_boxes:
        dict_store = {}
        dict_store["bbox"] = [int(x),int(y),int(w),int(h)]
        dict_store["score"] = float(weigh)
        
        #May have to change
        dict_store["image_id"] = file_names[i]
        dict_store["category_id"] = 1
        

        dict_list.append(dict_store)
        total = total + 1
        store_img = cv2.rectangle(store_img, (int(x),int(y)), (int(x)+winW,int(y)+winH), (255,0,0), 2)
    cv2.imwrite("PennFudanPed/output/img"+file_names[i]+".png",img)
            

with open('output_file.json', 'w') as fout:
    json.dump(dict_list , fout)
