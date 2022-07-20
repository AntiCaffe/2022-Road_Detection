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


img = mpimg.imread('1.jpg')

plt.figure(figsize=(10,8))
print('type',type(img),'dimensions',img.shape)
plt.imshow(img)
plt.show()

gray = grayscale(img)
plt.figure(figsize=(10,8))
plt.imshow(gray,cmap='gray')
plt.show()

kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)
plt.figure(figsize=(10,8))
plt.imshow(blur_gray, cmap='gray')
plt.show()

low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges, cmap='gray')
plt.show()

imshape = img.shape
vertices = np.array([[(0,imshape[0]),
                      (180,250),
                      (300,250),
                      (imshape[1]-100,imshape[0])]], dtype=np.int32)
mask = region_of_interest(edges,vertices)

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()


