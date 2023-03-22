import cv2
img = cv2.imread('backpropagation.png')
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gaussianimg = cv2.GaussianBlur(grayImg,(21,21),0)
cv2.imwrite("GaussianBlur.jpg",gaussianimg)
