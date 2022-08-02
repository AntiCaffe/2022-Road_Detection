import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color)


    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img, lines, color=[255,0,0], thickness = 3):
    line1 = lines[0]
    line2 = lines[0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if line1[0][0] < x1 or line1[0][2] < x1:
                line1 = line
            if line2[0][0] > x1 or line2[0][2] > x1:
                line2 = line

    cv2.line(img,(line1[0][0],line1[0][1]),(line1[0][2],line1[0][3]),color,thickness)
    cv2.line(img, (line2[0][0], line2[0][1]), (line2[0][2], line2[0][3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,origin):
    lines =     cv2.HoughLinesP(img, rho, theta, threshold,np.array([]),
                                minLineLength = min_line_len,
                                maxLineGap=max_line_gap)
    original = origin
    ##line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(original,lines)
    return original



img = mpimg.imread('1.jpg')

# plt.figure(figsize=(10,8))
# print('type',type(img),'dimensions',img.shape)
# plt.imshow(img)
# plt.show()

gray = grayscale(img)
# plt.figure(figsize=(10,8))
# plt.imshow(gray,cmap='gray')
# plt.show()

kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)
# plt.figure(figsize=(10,8))
# plt.imshow(blur_gray, cmap='gray')
# plt.show()

low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold, high_threshold)

# plt.figure(figsize=(10,8))
# plt.imshow(edges, cmap='gray')
# plt.show()

imshape = img.shape
vertices = np.array([[(0,imshape[0]),
                      (180,250),
                      (300,250),
                      (imshape[1]-100,imshape[0])]], dtype=np.int32)
mask = region_of_interest(edges,vertices)

# plt.figure(figsize=(10,8))
# plt.imshow(mask, cmap='gray')
# plt.show()

rho = 1
theta = np.pi/180
threshold = 10
min_line_len = 100
max_line_gap = 50

lines = hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap,img)

plt.figure(figsize=(10,8))
plt.imshow(lines, cmap='gray')
plt.show()