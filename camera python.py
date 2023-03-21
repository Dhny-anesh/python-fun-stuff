# importing the python open cv library
import cv2

import os
# intialize the webcam and pass a constant which is 0
cam = cv2.VideoCapture(0)

# title of the app
cv2.namedWindow('python webcam screenshot app')

# let's assume the number of images gotten is 0
img_counter = 0

# while loop
while True:
    # intializing the frame, ret
    ret, frame = cam.read()
    # if statement
    if not ret:
        print('failed to grab frame')
        break
    # the frame will show with the title of test
    cv2.imshow('test', frame)
    #to get continuous live video feed from my laptops webcam
    k  = cv2.waitKey(1)
    # if the escape key is been pressed, the app will stop
    if k%256 == 27:
        print('escape hit, closing the app')
        break
        cv2.imwrite(img_name, frame)
        print('screenshot taken')
        # the number of images automaticallly increases by 1
        img_counter += 1

# release the camera
cam.release()

# stops the camera window
cam.destoryAllWindows()