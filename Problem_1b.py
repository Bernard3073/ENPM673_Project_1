#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:37:48 2021

@author: bernard
"""

import numpy as np 
import cv2 

def main():
    cap = cv2.VideoCapture('/home/bernard/ENPM673/Project_1/Tag1.mp4')
    if (cap.isOpened()==False):
        print("Error")
    while cap.isOpened(): 
        ret, frame = cap.read()
        
        if ret == True:
            scale_percent = 60 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 400, 420)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            squares = []
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.1*cnt_len, True)
                if len(cnt) == 4:
                    if 2000 < cv2.contourArea(cnt) < 17500:
                        squares.append(cnt)

            cv2.drawContours(frame, contours, -1, (0,0,255), thickness = 2)
            cv2.drawContours(frame, squares, -1, (255,128,0), thickness = 2)
            
            # print(hierarchy)
            # for i in range(len(contours)):
            #     if hierarchy[0][i][3] == -1:
                    
            
            cv2.imshow('Feed', frame) 
            if cv2.waitKey(10) & 0xFF == ord("q"): 
                break 
        else:
            break
    cap.release() 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()