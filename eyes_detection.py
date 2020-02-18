import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def Opencv(directory,result_dir):
    #Setup the enviorment by linking to the Haar Cascades Models

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    img = cv2.imread(directory)

    # Resize the image to save space and be more manageable.
    # We do this by calculating the ratio of the new image to the old image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    w_size = 0
    h_size = 0
    #Perform the resizing and store the resized image in variable resized

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    if (str.split(os.path.split(dir.split('.')[0])[-1], '_')[-1]!='0'):
        resizde1 = cv2.resize 

    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    #Identify the face and eye using the haar-based classifiers.
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = resized[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
    x, y, w, h = faces[0]
    
    if eyes.shape[0] <= 1 :
        w_size = 0
        h_size = 0
    else:
        w_size = int(2/5 * w)
        h_size = int(1/50 * h)
        
        p_pan=pd.DataFrame(eyes).iloc[:,1:]
        p_pan=pd.DataFrame(eyes).T
        p_pan=abs(p_pan.corr(method='pearson'))

        #print(p_pan)
        filter_pan = p_pan[(p_pan<1) & (p_pan>0.6)]
        cal_pan=filter_pan.sum() > 0.6
        cal_pan.index[cal_pan]
        
        #print(cal_pan)
        eyes=eyes[cal_pan.index[cal_pan],:]
        
        #y값을 비교해서 boundury 안에 있으면 x값이 가장작은 행선택
        if((eyes[0,1] > eyes[1,1]-5) | (eyes[0,1] < eyes[1,1] +5)):
            eyes = [eyes[np.argmin(eyes[:,0]),:]]
        
        #아니면 y값이 가장작은 행 선택.
        else:
            eyes=[eyes[np.argmin(eyes[:,1]),:]]
            
    
        for (ex,ey,ew,eh) in eyes :
            img=cv2.rectangle(roi_color,(ex-7,ey-7),(ex+ew+w_size,ey+eh+h_size),(0,0,0),0)
    #Display the bounding box for the face and eyes
    
    #print(eyes[0])
        #plt.imshow(img[ey-4 : ey+eh+h_size, ex-5: ex+ew+w_size])
        cv2.imshow('img',resized)
        cv2.waitKey(0)
        #cv2.imwrite(os.path.join(result_dir, os.path.split(directory)[-1]), img[ey-5 : ey+eh+h_size, ex-5: ex+ew+w_size])
Opencv(dir, r'C:\Users\YongTaek\Desktop\시도\detect\test8_0.test.jpg')



'''###########################여기부터 코드 다시시작#############################'''
dir = r'C:\Users\YongTaek\Desktop\test8.jpg'
dir1 = r'C:\Users\YongTaek\Desktop\try\origin\test8_1.jpg'

dir.split('.')[0]
os.path.split(dir.split('.')[0])[-1]
str.split(os.path.split(dir.split('.')[0])[-1], '_')[-1]





#Setup the enviorment by linking to the Haar Cascades Models

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


img = cv2.imread(dir)
plt.imshow(img)
plt.imshow(img[0:256, 0:256])
# Resize the image to save space and be more manageable.
# We do this by calculating the ratio of the new image to the old image
r = 500.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))
w_size = 0
h_size = 0
#Perform the resizing and store the resized image in variable resized
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
resized.shape
image



    if (str.split(os.path.split(dir.split('.')[0])[-1], '_')[-1]!='0'):
        resizde1 = cv2.resize 




dir = r'C:\Users\YongTaek\Desktop\test8.jpg'
img = cv2.imread(directory)


def Opencv(directory=dir, img=img, result_dir):
    #Setup the enviorment by linking to the Haar Cascades Models

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    dic={}
    for i in range(14):
        dic['img'+str(i)] = img[0:256, 256*i:256*(i+1)]

    img = dic['img0']
    # Resize the image to save space and be more manageable.
    # We do this by calculating the ratio of the new image to the old image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    w_size = 0
    h_size = 0
    #Perform the resizing and store the resized image in variable resized

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    #Identify the face and eye using the haar-based classifiers.
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = resized[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
    x, y, w, h = faces[0]
    
    if eyes.shape[0] <= 1 :
        w_size = 0
        h_size = 0
    else:
        w_size = int(2/5 * w)
        h_size = int(1/50 * h)
        
        p_pan=pd.DataFrame(eyes).iloc[:,1:]
        p_pan=pd.DataFrame(eyes).T
        p_pan=abs(p_pan.corr(method='pearson'))

        #print(p_pan)
        filter_pan = p_pan[(p_pan<1) & (p_pan>0.6)]
        cal_pan=filter_pan.sum() > 0.6
        cal_pan.index[cal_pan]
        
        #print(cal_pan)
        eyes=eyes[cal_pan.index[cal_pan],:]
        
        #y값을 비교해서 boundury 안에 있으면 x값이 가장작은 행선택
        if((eyes[0,1] > eyes[1,1]-5) | (eyes[0,1] < eyes[1,1] +5)):
            eyes = [eyes[np.argmin(eyes[:,0]),:]]
        
        #아니면 y값이 가장작은 행 선택.
        else:
            eyes=[eyes[np.argmin(eyes[:,1]),:]]
            
    
        for (ex,ey,ew,eh) in eyes :
            img=cv2.rectangle(roi_color,(ex-7,ey-7),(ex+ew+w_size,ey+eh+h_size),(0,0,0),0)
            #Display the bounding box for the face and eyes
    
            #print(eyes[0])
            #plt.imshow(img[ey-4 : ey+eh+h_size, ex-5: ex+ew+w_size])
            cv2.imshow('img',resized)
            cv2.waitKey(0)
            #cv2.imwrite(os.path.join(result_dir, os.path.split(directory)[-1]), img[ey-5 : ey+eh+h_size, ex-5: ex+ew+w_size])

dir1 = r'C:\Users\YongTaek\Desktop\try\resolution\resol_test8_0.jpg'
dir2 = r'C:\Users\YongTaek\Desktop\try\resolution\resol_test8_1.jpg'

img1 = cv2.imread(dir1)
img2 = cv2.imread(dir2)
np.concatenate((img1, img2), axis=1).shape