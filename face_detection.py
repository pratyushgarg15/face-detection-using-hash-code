# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:51:58 2019

@author: HP
"""

import cv2 as cv
import numpy as np
feed = cv.VideoCapture(0)

clf = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

while True:
    check, img = feed.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(grey)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    print(img.shape)
    cv.imshow('test',img)
    key = cv.waitKey(1)
    if key== ord('q'):
        break
    
feed.release()
cv.destroyAllWindows()    
    