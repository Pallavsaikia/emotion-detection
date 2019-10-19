"""
This is starting page
It contains all the link
"""

import cv2  ##opencv
import dlib ##face detection library
from imutils.face_utils import FaceAligner##for aligning face
import numpy##for numpy arrays
import pandas as pd
from sklearn.preprocessing import LabelBinarizer ##for encoding
from keras.models import load_model##load model
from threading import Thread

import socket


import landmark_detection.facemetrics as fm
##packages loaded for face detection
import detect_face.face as face

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




#################################class objects##########################################################

detector = dlib.get_frontal_face_detector()
##location of facial landmark datafile
path='landmark_detection/shape_predictor_68_face_landmarks.dat'
###facial landmark detector
predictor=dlib.shape_predictor(path)
##face alignment
facealign = FaceAligner(predictor, desiredFaceWidth=500,desiredFaceHeight=500)
#face model to predict whose faces are detected
face_model=load_model('model/face_recognition.h5')
##names of people to be detected
face_names=pd.read_csv('model/savednames.csv')
Y=face_names.iloc[:,0:1].values##names of faces
encoder = LabelBinarizer()##transform to localbinarizer
y = encoder.fit_transform(Y)
ip=""



path='C:/xampp/htdocs/music/'




##############################main section##############################################################
def main_prog(frame):##queue and parallel process id
    ################variables###############


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame,faceloc,status=face.face_locate(frame,gray,detector)##locates face; status=if face is present; faceloc = object of location of face; returns frame with face pointed
    if status !=0:##if face is detected
        emotion=get_landmarks(frame,faceloc)
        return emotion
    else:
        return "no face detected"

       

#####################################################################################################
#########################################main function#################################################
def get_landmarks(frame,faceloc):
        cv2.imwrite('temporary_context.jpg',frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)##converts to grayscale
        for j, d in enumerate(faceloc):##loops through faces in the frame
            ####################################align face############################################
            detected_rect=dlib.rectangle(d.left()-30, d.top()-30, d.right()+30, d.bottom()+30)##rectangle with the location of the faces
            faceAligned = facealign.align(frame, gray, detected_rect)##aligns the face
            ###########################################################################################
            
            #############################################get face matrix##################################
            grayed_aligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)##converts the aligned faces
            not_needed1,aligned_face_loc,not_needed2=face.face_locate(faceAligned,grayed_aligned,detector)##gets the location of the aligned face
            for m, n in enumerate(aligned_face_loc):##loops through aligned faces
                landmarks=numpy.matrix([[p.x,p.y] for p in predictor(grayed_aligned,dlib.rectangle(n.left(),n.top(),n.right(),n.bottom())).parts()])##returns face landmark numpy array
                ##displays location of face landmarks
            #############################################################################################    
                lm = numpy.squeeze(numpy.asarray(landmarks))
                face_metrics=fm.face_metric(lm)
                face_metric=numpy.asarray(face_metrics)
                a = numpy.asmatrix(face_metric)
            ##########################normalise face matrix###############################
                detected_face=predict_face(a,face_model)##name of the face detected
                
        return detected_face
 
####################################################################################################################
    
############################################predict face landmarks###########################################
def predict_face(face_metrics,face_model):

    prediction=face_model.predict(face_metrics)##predict face
    prediction=(prediction>0.6)##confidence
    if prediction.any() == True:
        return (encoder.inverse_transform(prediction)[0])
    else:
        return ('unknown')
################################################################################################################
##################for socket###########



#data1=[]
#while True:
#        data = client.recv(4096)
#        if not data:
#            break
#        else:
#            print(data)
#            print(type(data))
##            data1.append(data)
#
#
#print(data1)
#data1=bytearray(data1)
def send(emotion,ip):
    clients=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clients.connect((ip,9080))
    path_read_from=path+emotion+'/'
    if emotion != "unknown" and emotion!="no face detected":
        listing=os.listdir(path_read_from)
        mylist=[emotion]
        
        for file in listing:
            mylist.append(file)
        emotion=','.join(str(e) for e in mylist)
        print(emotion)
    
    clients.send(emotion.encode())
    clients.close()
    


def start():
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    hostname = socket.gethostname()    
    IPAddr = socket.gethostbyname(hostname) 
    sock.bind((IPAddr,8000))
    
    while True:   
        print("listening")
        sock.listen(1)
    
        (client,(ip,port))=sock.accept()
        print(ip+' has connected')
        with open('tst.jpg', 'wb') as img:
            while True:
                data = client.recv(4096)
                if not data:
                    break
                else:
                    img.write(data)
            img.close()
        img=cv2.imread("tst.jpg")
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
         
        angle90 = 90
        
        M = cv2.getRotationMatrix2D(center, angle90, 1.0)  
        rotated90 = cv2.warpAffine(img, M, (h, w)) 
        emotion=main_prog(rotated90)
        
        try:
            process=Thread(target=send, args=[emotion,ip])
            process.daemon=True
            process.start()
        except:
            print("error")

        
    sock.close()
    


start()
##main_prog()
