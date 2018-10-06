import sys
import numpy as np
import matplotlib 
matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

sys.path.append("./TensorFlow-MNIST")
print(sys.path)

from MNISTTester import MNISTTester
import os
	
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = script_dir+'/Tensorflow-MNIST/mnist/data/'
model_path = script_dir+'/Tensorflow-MNIST/models/mnist-cnn'

# Camera 객체를 생성 후 사이즈르 320 X 240 으로 조정.
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

mnist = MNISTTester(
            model_path=model_path,
            data_path=data_path)

while(1):
    # camera에서 frame capture.
    ret, frame = cap.read()
    flag_result = 0

    if ret:
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 32:
        print("check!")
        crops = []
        n1 = 0
        n2 = 0
        
        # image = cv2.imread("2plus2.png")
        cv2.imshow("frame", frame)
        cv2.imwrite("c_1_frame.jpg", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.waitKey(0)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # grayscale
        cv2.imshow("gray", gray)
        cv2.imwrite("c_2_gray.jpg", gray)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.waitKey(0)

        img2 = cv2.equalizeHist(gray)

        cv2.imshow("histogram equalized", img2)
        cv2.imwrite("c_equal.jpg",img2)

#        img2 = cv2.GaussianBlur(gray,(5,5),0)
#        cv2.imshow("denoise", img2)
#        cv2.imwrite("c_denoise.jpg",img2)
#        cv2.waitKey(0)

        ret3, thresh = cv2.threshold(img2, 7, 255, cv2.THRESH_BINARY_INV)
        img_thresh = cv2.resize(thresh,(300,240),interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img_thresh", img_thresh)
        cv2.imwrite("c_3_otsu.jpg", img_thresh)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.waitKey(0)

#        gray = cv2.resize(img2,(300,240),interpolation=cv2.INTER_CUBIC)
#        cv2.imshow("resize gray", gray)
#        cv2.imwrite("c_4_resize_gray.jpg", gray)
#        if cv2.waitKey(1) & 0xFF == 27:
#            break
#        cv2.waitKey(0)


#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#        opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
#        cv2.imshow("opening",opening)
#        cv2.imwrite("c_5_opening.jpg", opening)
#        cv2.waitKey(0)
 
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        dilated = cv2.dilate(img_thresh,kernel,iterations = 3) # dilate
        cv2.imshow("dilated", dilated)
        cv2.imwrite("c_5_dilated.jpg", dilated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.waitKey(0)
        f, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

        # get contours
        # for each contour found, draw a rectangle around it on original image
        idx = 0
        for contour in contours:
            # get rectangle bounding contour

            [x,y,w,h] = cv2.boundingRect(contour)
            print("boundingRect : ", x,y,w,h)
            # discard areas that are too large

            if h>300 and w>300:

                continue
            # discard areas that are too small

            if h<40 or w<40:

                continue

            idx=idx+1
            #inverse_threshold = 255-img_thresh
            #crops.append(cv2.resize(inverse_threshold[y:y+h,x:x+w], (28, 28), interpolation=cv2.INTER_CUBIC))

            inverse_dilated = 255-dilated
            crops.append(cv2.resize(inverse_dilated[y:y+h,x:x+w], (28, 28), interpolation=cv2.INTER_CUBIC))

        testidx = 0 
        for image in crops:
            print(testidx)
            testidx=testidx+1
            cv2.imshow("crop image",image) 
            cv2.imwrite("c_6_crop_"+str(testidx)+".jpg", image)
            cv2.waitKey(0)

        print("crops : ")
        print(len(crops))
        sorted(crops, key=lambda crop: abs((sum([crop[i][1] for i in range(len(crop))])/len(crop))-(300/2)))
        testidx = 0 
        for image in crops:
            print(testidx)
            testidx=testidx+1
            cv2.imshow("crop image",image) 
            cv2.waitKey(0)
            
        if(len(crops)>=3):
            n1 = mnist.predict_2(crops[1])
            n2 = mnist.predict_2(crops[2])
            
            print("sum of n1+n2 is ",n1+n2)
       	    flag_result = 1
            
        if(len(crops)==2):
            n1 = mnist.predict_2(crops[1])
            n2 = mnist.predict_2(crops[2])

            print("sum of n1+n2 is ",n1+n2)
            flag_result = 1

    if(flag_result==1): break 
            
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
