# Getting Started

In this notebook, we will see how to use OpenCV-Python and some basic operations of OpenCV.

---

# Import OpenCV-Python and other packages


```python
# For python2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```


```python
import numpy as np
import cv2 # OpenCV-Python
%matplotlib inline
import matplotlib.pyplot as plt

print("OpenCV-Python Version %s" % cv2.__version__)
```

    OpenCV-Python Version 4.1.2
    


```python
!wget https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/messi.jpg
```

    --2021-03-12 07:37:10--  https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/messi.jpg
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.111.133, 185.199.108.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.111.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 72974 (71K) [image/jpeg]
    Saving to: ‘messi.jpg’
    
    messi.jpg           100%[===================>]  71.26K  --.-KB/s    in 0.01s   
    
    2021-03-12 07:37:10 (5.11 MB/s) - ‘messi.jpg’ saved [72974/72974]
    
    

# Open/display an image


```python
img = cv2.imread('./messi.jpg', cv2.IMREAD_COLOR)

# If the image path is wrong, the resulting img will be none
if img is None:
    print('Open Error')
else:
    print('Image Loaded')
```

    Image Loaded
    


```python
# Check the resulting img
print(type(img))
print(img.shape) # H, W, C
print(img.dtype)
print(img[:2, :2, :])  # Right-upper-most few pixels of the image
```

    <class 'numpy.ndarray'>
    (342, 548, 3)
    uint8
    [[[39 43 44]
      [42 46 47]]
    
     [[37 40 44]
      [42 45 49]]]
    


```python
# display an image using matplotlib
# plt.imshow(img) # => The color of this line is wrong
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7fa44a6c0b90>




    
![png](output_7_1.png)
    


# Basic operations on Images
## 1. Draw an object

- `cv2.line(image, startPoint, endPoint, rgb, thickness)`
- `cv2.rectangle(image, topLeft, bottomRight, rgb, thickness)`
- `cv2.circle(image, center, radius, rgb, thickness)`
- `cv2.ellipse(image, center, axes, angle, startAngle, endAngle, rgb, thickness)`


```python
# Create a black image
img2 = np.zeros((512,512,3), np.uint8)
plt.imshow(img2)
```




    <matplotlib.image.AxesImage at 0x7fa447b35f50>




    
![png](output_9_1.png)
    



```python
# Draw a line using cv2.line(image, startPoint, endPoint, rgb, thickness)
cv2.line(img2, (0,0), (511,511), (255,0,0), 5)
# => Diagonal red line with thickness of 5 px

# Draw a rectangle using cv2.rectangle(image, topLeft, bottomRight, rgb, thickness)
cv2.rectangle(img2, (384,0), (510,128), (0,255,0), 3)
# => Green rectangle with thickness of 3 px

# Draw a circle using cv2.circle(image, center, radius, rgb, thickness)
cv2.circle(img2, (447,63), 63, (0,0,255), -1)
# => Blue filled circle(note that the thickness is -1)

# Draw a ellipse using cv2.ellipse(image, center, axes, angle, startAngle, endAngle, rgb, thickness)
cv2.ellipse(img2, (256,256), (100,50), -45, 0, 180, (255,0,0), -1)
# => Red wide down-half ellipse

plt.imshow(img2)
```




    <matplotlib.image.AxesImage at 0x7fa4497d0590>




    
![png](output_10_1.png)
    



```python
# Draw a line using cv2.polylines(image, points, isClosed, rgb, thickness, lineType, shift)
pts = np.array([[10,10],[150,200],[300,150],[200,50]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img2,[pts],True,(0,255,255),3)
# => Cyan closed quadrangle 

print(pts)
plt.imshow(img2)
```

    [[[ 10  10]]
    
     [[150 200]]
    
     [[300 150]]
    
     [[200  50]]]
    




    <matplotlib.image.AxesImage at 0x7fa4479f1610>




    
![png](output_11_2.png)
    



```python
# Put some text using cv2.putText(image, text, bottomLeft, fontType, fontScale, rgb, thickness, lineType)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img2, 'OpenCV', (10,500), font, 4, (255,255,255), 5, cv2.LINE_AA)
# => White 'OpenCV' text at the bottom

plt.imshow(img2)
```




    <matplotlib.image.AxesImage at 0x7fa4479d4690>




    
![png](output_12_1.png)
    


---

## 2. Modify pixels & ROI

- You can access/modify a single pixel or ROI using Numpy indexing.
- Just like matrix indexing, `img[a, b]` refer to `a`-th row and `b`-th column.


```python
# Access a pixel value(BGR order)
img[50, 235]
```




    array([29, 24, 25], dtype=uint8)




```python
# Change pixel values
for i in range(5):
    for j in range(5):
        img[50+i, 235+j] = (0, 255, 0)
# => Green dot above messi's head

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7fa447940f50>




    
![png](output_15_1.png)
    



```python
# ROI is obtained using Numpy indexing 
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

# img[50:55, 235:240] = (0, 255, 0)  # The for-loop in the code block above is equavalent to this line. 

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7fa4478add90>




    
![png](output_16_1.png)
    


# Exercises
## 1. Try to create the bounding box and label using drawing functions available in OpenCV

<img src="https://github.com/bckim92/iab_practice_example/blob/master/images/prac_1.png?raw=1" width="400">


```python
img = cv2.imread('./messi.jpg', cv2.IMREAD_COLOR)
new_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

# Your code here
cv2.rectangle(new_img, (200, 65), (270,140), (255,0,0), 3)
cv2.putText(new_img, 'Messi', (190,50), font, 1, (255,255,255), 2, cv2.LINE_AA)
cv2.rectangle(new_img, (330, 280), (390,340), (255,0,0), 3)
cv2.putText(new_img, 'ball', (330,270), font, 1, (255,255,255), 2, cv2.LINE_AA)

plt.imshow(new_img)
```




    <matplotlib.image.AxesImage at 0x7fa446e82b10>




    
![png](output_18_1.png)
    


---

### Reference

Please see the following official tutorials for more detailed explanation.

 - [Basic Operations on Images — OpenCV documentation](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html)
 - [Drawing Functions in OpenCV — OpenCV documentation](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html)
