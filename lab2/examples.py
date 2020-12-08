import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time 
def detect_match(algorithm,query_img,train_img,min_match_count,verbose=False):
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
        #print(M,mask)
        matchesMask = mask.ravel().tolist()
        h,w = query_image.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        msed = np.mean([np.sqrt(np.sum(diff)) for diff in (np.power(pts-dst,2))]/(np.sqrt(h**2+w**2)))
        #cv2.polylines(train_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        if verbose==True:
            print( "Not enough matches are found - {}/{}".format(len(good), min_match_count) )
        matchesMask = [0]
    
    return key_pts1, key_pts2, good, matchesMask, msed

def plot_match(query_img, train_img, key_pts1, key_pts2, good, matchesMask):
    draw_params = dict(matchColor = (255.0,255.0,255.0), # draw matches in white color
                   singlePointColor = (255.0,0,0),
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(query_img, key_pts1, train_img, key_pts2, good, None, **draw_params)
    plt.imshow(img3, 'gray'),plt.show()

img1 = cv2.imread('data/20200925_233107.jpg',0) 
img2 = cv2.imread('data/20200925_233551.jpg',0) 
do = time.monotonic()
key_pts1, key_pts2, good, matchesMask, mse = detect_match(algorithm=cv2.SIFT_create().detectAndCompute,query_img=img1,train_img=img2,
                                                     min_match_count=10)
match = good
inlier = matchesMask
done = time.monotonic()
if match == 0:
    match = np.inf
result = {
        "image_name": img1,
        "count_of_good": np.sum(inlier)/(len(match)),
        "MSE_distance": mse,
        "duration_of_computing": done - do
    }
print(result,',',sep='')

img3=plot_match(query_img=img1, key_pts1=key_pts1,train_img=img2,  key_pts2=key_pts2, good=good, matchesMask=matchesMask)
plt.show(img3)

