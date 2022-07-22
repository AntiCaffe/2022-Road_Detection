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
