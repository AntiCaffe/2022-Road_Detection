# OpenCV, HoughLine을 이용한 이미지 차선인식
주어진 이미지를 흑백으로 처리, 필터를 씌운 후 
HoughLine을 이용해 차선을 검출한다.
## 1.이미지 출력
```python
img = mpimg.imread('1.jpg')

plt.figure(figsize=(10,8))
print('type',type(img),'dimensions',img.shape)
plt.imshow(img)
plt.show()
```

## 2. 이미지 Gray
```python
gray = grayscale(img)
plt.figure(figsize=(10,8))
plt.imshow(gray,cmap='gray')
plt.show()
```

## 3. 가우시안 필터
```python
kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)
plt.figure(figsize=(10,8))
plt.imshow(blur_gray, cmap='gray')
plt.show()
```

## 4. Canny
```python
low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges, cmap='gray')
plt.show()
```

## 5.ROI(Region of Interest)
```python
imshape = img.shape
vertices = np.array([[(0,imshape[0]),
                      (180,250),
                      (300,250),
                      (imshape[1]-100,imshape[0])]], dtype=np.int32)
mask = region_of_interest(edges,vertices)

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()
```

## 6.Houghline
```python
rho = 3
theta = np.pi/180
threshold = 10
min_line_len = 100
max_line_gap = 50

lines = hough_lines(mask,rho,theta,threshold,min_line_len,max_line_gap)

plt.figure(figsize=(10,8))
plt.imshow(lines, cmap='gray')
plt.show()
```
