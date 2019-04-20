# Author: Bing Shi
# For Image
# 2019.1.3
# Nanjing

import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
#from copy import deepcopy

def siftdetcompmat(fn1, fn2):
    MIN_MATCH_COUNT = 10
    RATIO = 0.5
    fnm1, ext = os.path.splitext(fn1)
    fnm2, ext = os.path.splitext(fn2)
    
    #--1 Read image-- 
    img1 = cv2.imread(fn1,0)               # queryImage
    img2 = cv2.imread(fn2,0)               # trainImage
    # gray= deepcopy(img1);  print(gray.shape);  # gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    #--2 Extract features--
    # Initiate SIFT detector    
    sift = cv2.xfeatures2d.SIFT_create()           # sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print('G1:',end=' '); print(des1.shape);  print('G2:',end=' '); print(des2.shape)
    # print(type(des1));  print(type(des1[0]));  print(des1[:3])
    # gray = cv2.drawKeypoints(gray, kp1, gray, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #--3 Match--
    #--3.1--
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print('Matches: %d'%len(matches))
    #--3.2--
    # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    
    #--4 Filter out--
    #--4.1--
    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.      
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
    #--4.2--
    # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in xrange(len(matches))]        # corresponding to matches
    # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         matchesMask[i]=[1, 0]    
    #--4.3--
    # store all the good matches as per Lowe's ratio test.       # delect outliers
    print('Ratio = %f'%RATIO)
    good = []
    for m,n in matches:
        if m.distance < RATIO * n.distance:
            good.append(m) 
    print('good matching: %d'%len(good))           #print(good)  
    if len(good)==0:
        return
    # for m in good:  print("%d %d,"%(m.queryIdx, m.trainIdx));
    # Homography   
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()               # corresponding to good
        h, w = img1.shape                                       # h, w, d = img1.shape for color image
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None       
    print('Homography: %d'%sum(matchesMask))   # print(matchesMask)
        
    #--5 Draw matches--    
    #--5.1--
    # cv2.drawMatchesKnn expects list of lists as matches.  
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2)  
    #--5.2--
    # draw_params = dict(matchColor = (0,255,0),
    #               singlePointColor = (255,0,0),
    #               matchesMask = matchesMask,
    #               flags = 0)                                                        # draw all
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params) 
    #--5.3--      
    draw_params = dict(matchColor = (0,255,0),        # draw matches in green color  
                       singlePointColor = None,
                       matchesMask = matchesMask,         
                       flags = 2)                                    # draw only inliers
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray')
    plt.show()
        
    #--6 Save results--
    src = [[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] for m in good ] 
    srcarr = np.asarray(src)
    np.savetxt(fnm1+'.kps1.txt', srcarr, fmt='%f %f', newline='\r\n')
    desind1 = [m.queryIdx for m in good ]
    des12 = des1[desind1]
    desi1 = des12.astype(int)
    np.savetxt(fnm1+'.des1.txt', desi1, fmt='%d '*128, newline='\r\n') 
    
    dst = [[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] for m in good ]
    dstarr = np.asarray(dst)
    np.savetxt(fnm2+'.kps1.txt', dstarr, fmt='%f %f', newline='\r\n')
    desind2 = [m.trainIdx for m in good ]
    des22 = des2[desind2]    
    desi2 = des22.astype(int)
    np.savetxt(fnm2+'.des1.txt', desi2, fmt='%d '*128, newline='\r\n') 

    mm=np.asarray(matchesMask)
    srcarr1 = srcarr[mm==1]
    desi3 = desi1[mm==1]
    np.savetxt(fnm1+'.kps.txt', srcarr1, fmt='%f %f', newline='\r\n')
    np.savetxt(fnm1+'.des.txt', desi3, fmt='%d '*128, newline='\r\n') 
    dstarr1 = dstarr[mm==1]
    desi4 = desi2[mm==1]
    np.savetxt(fnm2+'.kps.txt', dstarr1, fmt='%f %f', newline='\r\n')
    np.savetxt(fnm2+'.des.txt', desi4, fmt='%d '*128, newline='\r\n') 
    
    #--7 Label points--
    dummy = np.zeros((1,1))
    kp11 = [kp1[m.queryIdx] for m in good ] 
    print('G1: %d'%len(kp11))
    img4 = cv2.drawKeypoints(img1, kp11, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.close()
    plt.imshow(img4, 'gray')
    for (i, p) in enumerate(srcarr1):
        #if i==2 or i==4: continue
        plt.text(p[0], p[1], '%i'%i, fontsize=8, color = "r")
    plt.show()
    
    kp22 = [kp2[m.trainIdx] for m in good ] 
    print('G2: %d'%len(kp22))
    img5 = cv2.drawKeypoints(img2, kp22, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.close()
    plt.imshow(img5, 'gray')
    for (i, p) in enumerate(dstarr1):
        plt.text(p[0], p[1], '%i'%i, fontsize=8, color = "r")
    plt.show()
    
# End

def surfdetcompmat(img, img2):
    # Create SURF object. specify params. set Hessian Threshold to 400, it is better to have a value 300-500.
    # for reduceing keypoints. take 50000
    surf = cv2.xfeatures2d.SURF_create(50000)               
    # cases where orientation is not a problem,  All the orientations are shown in same direction. It is more faster.
    surf.setUpright(True)  
    # get 128-dim descriptors
    surf.extended = True   
    kp, des = surf.detectAndCompute(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    plt.imshow(img2)
    plt.show()
# End

if __name__ == "__main__":
    siftdetcompmat(sys.argv[1], sys.argv[2])
