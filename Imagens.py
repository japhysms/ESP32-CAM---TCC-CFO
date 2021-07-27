import cv2
import numpy as np


im1 = cv2.imread("imf1.jpg")
# cv2.imshow("IM1", im1)
im2 = cv2.imread("imf2.jpg")
# cv2.imshow("IM2", im2)
im3 = cv2.imread("imf3.jpg")
# cv2.imshow("IM3", im3)
im4 = cv2.imread("imf4.jpg")
# cv2.imshow("IM4", im4)
im1 = cv2.bilateralFilter(im1, 9, 75, 75)
im2 = cv2.bilateralFilter(im2, 9, 75, 75)
im3 = cv2.bilateralFilter(im3, 9, 75, 75)
im4 = cv2.bilateralFilter(im4, 9, 75, 75)
        

hsvim1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
hsvim2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
hsvim3 = cv2.cvtColor(im3, cv2.COLOR_BGR2HSV)
hsvim4 = cv2.cvtColor(im4, cv2.COLOR_BGR2HSV)  
l_bound = np.array([0,130,150])
u_bound = np.array([35,250,255])
kernel = np.ones((5,5), np.uint8)
mask1 = cv2.inRange(hsvim1, l_bound, u_bound)
mask2 = cv2.inRange(hsvim2, l_bound, u_bound)
mask3 = cv2.inRange(hsvim3, l_bound, u_bound)
mask4 = cv2.inRange(hsvim4, l_bound, u_bound)
#img_erosion = cv2.erode(mask4, kernel, iterations=1)
#img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
# res = cv2.bitwise_and(frame, frame, mask= mask)
res_con1 = cv2.bitwise_and(im1, im1, mask= mask1)
res_con2 = cv2.bitwise_and(im2, im2, mask= mask2)
res_con3 = cv2.bitwise_and(im3, im3, mask= mask3)
res_con4 = cv2.bitwise_and(im4, im4, mask= mask4)

_,thr1 = cv2.threshold(mask1, 100, 255, 0)
contours1,_ = cv2.findContours(thr1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im1, contours1, -1, (0,255,0), 3)

_,thr2 = cv2.threshold(mask2, 100, 255, 0)
contours2,_ = cv2.findContours(thr2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im2, contours2, -1, (0,255,0), 3)

_,thr3 = cv2.threshold(mask3, 100, 255, 0)
contours3,_ = cv2.findContours(thr3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im3, contours3, -1, (0,255,0), 3)

_,thr4 = cv2.threshold(mask4, 100, 255, 0)
contours4,_ = cv2.findContours(thr4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(im4, contours4, -1, (0,255,0), 3)
    
# gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray, 100, 200)
    
# _,thr = cv2.threshold(mask, 100, 255, 0)
# contours,_ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(res_con, contours, -1, (255,0,0), 3)

# cv2.imshow('Original Frame', frame)
cv2.imwrite('MascaraIMf1.jpg',mask1)
cv2.imwrite('MascaraIMf2.jpg',mask2)
cv2.imwrite('MascaraIMf3.jpg',mask3)
cv2.imwrite('MascaraIMf4.jpg',mask4)

cv2.imwrite('Resf1.jpg',im1)
cv2.imwrite('Resf2.jpg',im2)
cv2.imwrite('Resf3.jpg',im3)
cv2.imwrite('Resf4.jpg',im4)

cv2.waitKey(0) 
#cv2.imshow('Mask', mask4)
# cv2.imshow('Res', res)
# cv2.imshow('Res_con', res_con)
# cv2.imshow('canny', canny)
