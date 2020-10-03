import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
import json 

def detect_match(algorithm,query_img,train_img,min_match_count):
    query_image = query_img
    train_image = train_img
    key_pts1,des1 = algorithm(query_image,None)
    key_pts2,des2 = algorithm(train_image,None)
    msed=np.inf
    if not (isinstance(des1,np.float32)&isinstance(des2,np.float32)):
        des1 = np.float32(des1)
        des2 = np.float32(des2)
    
    flann_idx = 1
    index_params = dict(algorithm = flann_idx, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = [m for m,n in matches if m.distance < 0.7*n.distance]
    
    if len(good)>min_match_count:
        src_pts = np.float32([ key_pts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ key_pts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        if M is None:
            return [0],[0],[0],[0],np.inf
        matchesMask = mask.ravel().tolist()
        h,w = query_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        msed = np.mean([np.sqrt(np.sum(diff)) for diff in (np.power(pts-dst,2))]/(np.sqrt(h**2+w**2)))
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), min_match_count) )
        matchesMask = [0]
    
    return key_pts1, key_pts2, good, matchesMask, msed


sift = cv2.SIFT_create().detectAndCompute
golden_image = cv2.imread('img1/IMG_20200927_101730.jpg',0) 
imgs = os.listdir('img1')
metrics = []

for img in imgs:
    image_i = cv2.imread('img1/{}'.format(img),0)  
    do = time.monotonic()
    _, __, match, inlier ,mse = detect_match(algorithm=sift,query_img=image_i,train_img=golden_image,
                                                min_match_count=5)
    done = time.monotonic()
    if match == 0:
        match = np.inf
    result = {
            "image_name": img,
            "count_of_good": np.sum(inlier)/(len(match)),
            "MSE_distance": mse,
            "duration_of_computing": done - do
        }
    print(result,',',sep='')
    metrics.append(result)
with open('metrics_Jyliya.json', 'w') as outfile:
    json.dump({"metrics":metrics}, outfile)  