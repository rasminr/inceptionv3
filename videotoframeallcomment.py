import cv2#to import opencv functions  
import os#operating system commands
# Read the video from specified path 
num=0#number of folder
currentframe = 0#number of frames
for file in os.listdir("videos2"):
    num=str(num+1)#convert interger to string
    str1="data"+num#concatenate 2 strings
    num=int(num)#convert string to integer
    cam = cv2.VideoCapture(os.path.join("videos2",file))#captures video file to object- cam
    if not os.path.exists(str1):
        os.makedirs(str1)#if there is no folder with the name str1
        
    ret,frame = cam.read()
     

        
    while ret:
        ret,frame = cam.read()
        if ret:

            name = './'+str1+'/frame'+str(currentframe)+'.jpg'
            print('Creating...' + name)  
            cv2.imwrite(name, frame) 

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe=currentframe+1
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 