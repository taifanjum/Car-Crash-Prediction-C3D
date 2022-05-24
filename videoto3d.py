# -*- coding: utf-8 -*-
"""
Created on Thu May  5 02:25:11 2022

@author: Taifanjum
"""
import numpy as np
import cv2

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=False, skip=False):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)] #Creates an array that would indicate the indexes of the frames were are interested on
            # e.g. is skip is true and depth is 16, array would look like [0, 3, 6, 9, 12, ...]
        else:
            #if skip is false
            frames = [x for x in range(self.depth)] # eg. if depth is 16, Array would look like [0,1,2,3,4,5...] 
            
        framearray = []

        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i]) #Gets the frames according to the frames array. Eg. if array is [0,3,6] it would get the 0, 3rd and 6th frames
            ret, frame = cap.read() #read the frames
            frame = cv2.resize(frame, (self.height, self.width)) #resize the frames according to desired height and width 
            if color:
                framearray.append(frame) #if color is true add the frames as is
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) #if color is false convert images to gray scale

        cap.release()
        return np.array(framearray)

    def get_UCF_classname(self, filename): #Get the labels from the filenames 
        filename = filename.split('_')
        filename = filename[1].split('.')
        return filename[0]