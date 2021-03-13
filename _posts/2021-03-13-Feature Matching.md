# [OpenCV-Python Tutorial] Feature Matching

In this notebook, we will see how to extract SIFT(Scale-Invariant Feature Transform) and match SIFT features of two images with OpenCV-Python.

---


```python
!pip install opencv-python==3.4.2.17
!pip install opencv-contrib-python==3.4.2.17
!wget https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/beaver.png
!wget https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/box.png
!wget https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/box_in_scene.png
```

    Requirement already satisfied: opencv-python==3.4.2.17 in /usr/local/lib/python3.7/dist-packages (3.4.2.17)
    Requirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from opencv-python==3.4.2.17) (1.19.5)
    Collecting opencv-contrib-python==3.4.2.17
    [?25l  Downloading https://files.pythonhosted.org/packages/12/32/8d32d40cd35e61c80cb112ef5e8dbdcfbb06124f36a765df98517a12e753/opencv_contrib_python-3.4.2.17-cp37-cp37m-manylinux1_x86_64.whl (30.6MB)
    [K     |████████████████████████████████| 30.6MB 130kB/s 
    [?25hRequirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from opencv-contrib-python==3.4.2.17) (1.19.5)
    Installing collected packages: opencv-contrib-python
      Found existing installation: opencv-contrib-python 4.1.2.30
        Uninstalling opencv-contrib-python-4.1.2.30:
          Successfully uninstalled opencv-contrib-python-4.1.2.30
    Successfully installed opencv-contrib-python-3.4.2.17
    --2021-03-12 08:26:18--  https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/beaver.png
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.111.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 30625 (30K) [image/png]
    Saving to: ‘beaver.png.2’
    
    beaver.png.2        100%[===================>]  29.91K  --.-KB/s    in 0.003s  
    
    2021-03-12 08:26:18 (10.2 MB/s) - ‘beaver.png.2’ saved [30625/30625]
    
    --2021-03-12 08:26:18--  https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/box.png
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.109.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 50728 (50K) [image/png]
    Saving to: ‘box.png.2’
    
    box.png.2           100%[===================>]  49.54K  --.-KB/s    in 0.01s   
    
    2021-03-12 08:26:18 (3.91 MB/s) - ‘box.png.2’ saved [50728/50728]
    
    --2021-03-12 08:26:18--  https://raw.githubusercontent.com/bckim92/iab_practice_example/master/images/box_in_scene.png
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.111.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 122490 (120K) [image/png]
    Saving to: ‘box_in_scene.png.2’
    
    box_in_scene.png.2  100%[===================>] 119.62K  --.-KB/s    in 0.03s   
    
    2021-03-12 08:26:18 (4.53 MB/s) - ‘box_in_scene.png.2’ saved [122490/122490]
    
    


```python
# For python2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2 # OpenCV-Python
%matplotlib inline
import matplotlib.pyplot as plt
import time
```


```python
# Load an image
beaver = cv2.imread('./beaver.png')
plt.imshow(cv2.cvtColor(beaver, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7fd22106c590>




    
![png](output_3_1.png)
    



```python
# Members of cv2.xfeatures2d
dir(cv2.xfeatures2d)
```




    ['BoostDesc_create',
     'BriefDescriptorExtractor_create',
     'DAISY_NRM_FULL',
     'DAISY_NRM_NONE',
     'DAISY_NRM_PARTIAL',
     'DAISY_NRM_SIFT',
     'DAISY_create',
     'FREAK_NB_ORIENPAIRS',
     'FREAK_NB_PAIRS',
     'FREAK_NB_SCALES',
     'FREAK_create',
     'HarrisLaplaceFeatureDetector_create',
     'LATCH_create',
     'LUCID_create',
     'PCTSIGNATURES_GAUSSIAN',
     'PCTSIGNATURES_HEURISTIC',
     'PCTSIGNATURES_L0_25',
     'PCTSIGNATURES_L0_5',
     'PCTSIGNATURES_L1',
     'PCTSIGNATURES_L2',
     'PCTSIGNATURES_L2SQUARED',
     'PCTSIGNATURES_L5',
     'PCTSIGNATURES_L_INFINITY',
     'PCTSIGNATURES_MINUS',
     'PCTSIGNATURES_NORMAL',
     'PCTSIGNATURES_REGULAR',
     'PCTSIGNATURES_UNIFORM',
     'PCTSignaturesSQFD_create',
     'PCTSignatures_GAUSSIAN',
     'PCTSignatures_HEURISTIC',
     'PCTSignatures_L0_25',
     'PCTSignatures_L0_5',
     'PCTSignatures_L1',
     'PCTSignatures_L2',
     'PCTSignatures_L2SQUARED',
     'PCTSignatures_L5',
     'PCTSignatures_L_INFINITY',
     'PCTSignatures_MINUS',
     'PCTSignatures_NORMAL',
     'PCTSignatures_REGULAR',
     'PCTSignatures_UNIFORM',
     'PCTSignatures_create',
     'PCTSignatures_drawSignature',
     'PCTSignatures_generateInitPoints',
     'SIFT_create',
     'SURF_create',
     'StarDetector_create',
     'VGG_create',
     '__doc__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'matchGMS']



# Extract SIFT features from an image

## 1. SIFT detector


```python
# Convert image color(BGR->Grayscale)
gray = cv2.cvtColor(beaver, cv2.COLOR_BGR2GRAY)
# You can convert the image when calling cv2.imread()
# gray = cv2.imread('./beaver.png', cv2.IMREAD_GRAYSCALE)

print(str(beaver.shape) + " => " + str(gray.shape))
plt.imshow(gray, cmap='gray')
```

    (211, 300, 3) => (211, 300)
    




    <matplotlib.image.AxesImage at 0x7fd21f318c50>




    
![png](output_7_2.png)
    



```python
# SIFT feature detector/descriptor
sift = cv2.xfeatures2d.SIFT_create()
```


```python
# SIFT feature detection
start_time = time.time()
kp = sift.detect(gray, None) # 2nd pos argument is a mask indicating a part of image to be searched in
#kp = sift.detect(beaver, None) # 2nd pos argument is a mask indicating a part of image to be searched in
print('Elapsed time: %.6fs' % (time.time() - start_time))
```

    Elapsed time: 0.032046s
    


```python
# Display the SIFT features
beaver_sift = cv2.drawKeypoints(beaver, kp, None)
plt.imshow(cv2.cvtColor(beaver_sift, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7fd21ea9eb10>




    
![png](output_10_1.png)
    



```python
# Display the rich SIFT features
beaver_sift2 = cv2.drawKeypoints(beaver, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(beaver_sift2, cv2.COLOR_BGR2RGB))
```




    <matplotlib.image.AxesImage at 0x7fd21ea90850>




    
![png](output_11_1.png)
    



```python
# Inspect the keypoints
print(type(kp))
print(len(kp))
```

    <class 'list'>
    144
    


```python
print(type(kp[0]))
print(dir(kp[0]))
```

    <class 'cv2.KeyPoint'>
    ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'angle', 'class_id', 'convert', 'octave', 'overlap', 'pt', 'response', 'size']
    


```python
# A keypoint's property
# kp is sorted by scale of the keypoints
print(kp[-1].angle) # Orientation
print(kp[-1].class_id)
print(kp[-1].octave)
print(kp[-1].pt) # Position
print(kp[-1].response)
print(kp[-1].size) # Scale
```

    317.97381591796875
    -1
    9437951
    (283.1337890625, 167.98963928222656)
    0.042956698685884476
    2.5780200958251953
    

## 2. Extract SIFT descriptor


```python
# Extract SIFT feature from the (gray) image and detected keypoints
start_time = time.time()
kp, des = sift.compute(gray, kp)
print('Elapsed time: %.6fs' % (time.time() - start_time))

# SIFT keypoints and descriptors at the same time
# start_time = time.time()
# kp, des = sift.detectAndCompute(gray, None)
# print('Elapsed time: %.6fs' % (time.time() - start_time))
```

    Elapsed time: 0.030295s
    


```python
# Inspect the descriptors
print(type(des))
print(des.shape)
print(des.dtype)
```

    <class 'numpy.ndarray'>
    (144, 128)
    float32
    


```python
print(len(des[0, :]))
print(des[0, :])
```

    128
    [ 57.  42.  30.  40.  49.   1.   0.  20.  37.  35.  16.  29.  43.   2.
       6.  50.  19.  20.  14.   7.  11.  28.  37. 109.  65.   0.   0.   1.
       2.   7.  28. 150.  27.  40.  58.  50.  13.   0.   0.   9. 150.  34.
      24.  23.   5.   0.   8. 134.  50.   8.   3.   0.   0.   0.  51. 150.
       3.   0.   0.   0.   0.   1.  28.  95.  24.   5.  14.  31.  21.  16.
       9.  15. 150.  65.   3.   5.   2.   1.   7.  53. 150.  28.   0.   0.
       0.   0.   2.  45.   4.   0.   0.   0.   0.   1.   2.   7.  21.   2.
       1.   4.   9.  22.  22.  28. 150.  11.   0.   0.   0.   0.  11.  86.
     150.   7.   0.   0.   0.   0.   0.  31.   3.   0.   0.   0.   0.   0.
       0.   2.]
    

---

# Feature Matching

## 1. SIFT Feature Matching


```python
# Open and show images
img1 = cv2.imread('./box.png')
img2 = cv2.imread('./box_in_scene.png')

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
```


```python
# SIFT feature extracting
sift = cv2.xfeatures2d.SIFT_create()
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

start_time = time.time()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print('Elapsed time: %.6fs' % (time.time() - start_time))

print('Image 1 - %d feature detected' % des1.shape[0])
print('Image 2 - %d feature detected' % des2.shape[0])
```


```python
# BFMatcher(Brute Force Matcher) with defalut setting
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)
print('%d matches' % len(matches))
```


```python
# Inspect matcher results
print(type(matches))
print(len(matches))
print(type(matches[0]))
print(len(matches[0]))  # Number of match candidate = k
```


```python
print(type(matches[0][0]))
print(dir(matches[0][0]))
```


```python
print(matches[0][0].distance)
print(matches[0][0].queryIdx)
print(matches[0][0].trainIdx)
print(matches[0][0].imgIdx)
print(matches[0][1].distance)
print(matches[0][1].queryIdx)
print(matches[0][1].trainIdx)
print(matches[0][1].imgIdx)
```


```python
# Apply ratio test as in David Rowe's paper
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
print('%d matches' % len(good_matches))
```


```python
# Display matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
```
