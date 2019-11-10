import cv2 
import os
# Read the video from specified path 
num=0
currentframe = 0
for file in os.listdir("videos2"):
    num=str(num+1)
    str1="data"+num
    num=int(num)
    cam = cv2.VideoCapture(os.path.join("videos2",file))
    if not os.path.exists(str1):
        os.makedirs(str1)
    ret,frame = cam.read()
     

        
    while ret:
        ret,frame = cam.read()
        if ret:

            name = './'+str1+'/frame'+str(currentframe)+'.jpg'
            print('Creating...' + name)  
            cv2.imwrite(name, frame) 

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 