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
![1](https://user-images.githubusercontent.com/97824834/180408007-681fc473-4485-462e-9727-9f0af4f39c3a.png)


## 2. 이미지 Gray
```python
gray = grayscale(img)
plt.figure(figsize=(10,8))
plt.imshow(gray,cmap='gray')
plt.show()
```
![2](https://user-images.githubusercontent.com/97824834/180408055-c9d42359-1f42-46bb-98a5-a74330adfdd1.png)


## 3. 가우시안 필터
```python
kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)
plt.figure(figsize=(10,8))
plt.imshow(blur_gray, cmap='gray')
plt.show()
```![3](https://user-images.githubusercontent.com/97824834/180408076-cbfe3fdf-284d-4691-b85c-e113a5e8a514.png)


## 4. Canny
```python
low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges, cmap='gray')
plt.show()
```
![4](https://user-images.githubusercontent.com/97824834/180408113-b93c83d2-5fae-415b-8b3a-cf1065596579.png)


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
![5](https://user-images.githubusercontent.com/97824834/180408163-a58929d5-fb6a-413f-ae6b-368cadbae62e.png)


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
![6](https://user-images.githubusercontent.com/97824834/180408184-a69c1093-bf9b-4623-a0e2-2af860b79bb4.png)


