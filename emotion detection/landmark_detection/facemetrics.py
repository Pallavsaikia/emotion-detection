# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:56:39 2018

@author: paul
"""

import math

def face_metric(landmarks):
    metric=[]
    ##########################################distance between upper and lower lips##################
    for i in range(0,67):
        for j in range(i+1,68) :
            a= math.sqrt(((landmarks[i][0]-landmarks[j][0])**2)+((landmarks[i][1]-landmarks[j][1])**2))
            metric.append(a)
    ##################
    
    
    
    
    
    
    return metric




        