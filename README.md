<p align="center">
    <h1 align="center">Edge-Based Color Constancy</h1>
</p>

## Introduction
A partial python reproduction of the journal paper "[Edge-Based Color Constancy](https://ieeexplore.ieee.org/document/4287009)", including Shades of Grey, Grey-Edge, and 2nd order Grey-Edge.

## Quickstart
### Prerequisite
1. python3
2. dependence package
   1. numpy
   2. scipy
   3. opencv-python
   
### Get started
```
from IllumEst_EdgeBased_Scipy import EBCCIMG
import numpy as np
import cv2

# import RAW IMAGE (float, uint8, uint16...)
img_RAW = cv2.imread('./input/Canon1DsMkIII_0199.PNG',1)
img_RAW = cv2.cvtColor(img_RAW, cv2.COLOR_BGR2RGB)

# different illuminant estimation methods
WP_GW = EBCCIMG(img_RAW,mode='GW')
WP_mRGB = EBCCIMG(img_RAW,mode='maxRGB')
WP_SoG = EBCCIMG(img_RAW,mode='SoG')
WP_GGW = EBCCIMG(img_RAW,mode='GGW', p=9, sigma=9)
WP_GE1 = EBCCIMG(img_RAW,mode='GE1', p=7, sigma=4)
WP_GE2 = EBCCIMG(img_RAW,mode='GE2', p=7, sigma=5)
```

`illumEst_EdgeBased_Scipy.py` is mainly based on `scipy`. More details are shown in it. All estimated illuminants are L2 normalized.

### Data
1. In the folder `./input`, `Canon1DsMkIII_0199.PNG` is one demosaicked raw image without white balance from [NUS-8](https://yorkucvil.github.io/projects/public_html/illuminant/illuminant.html).
2. `Canon1DsMkIII_0199.json` is the metadata extracted by [dcraw](https://www.dechifro.org/dcraw/dcraw.c). 

## Visualization
These images are done with white balance and gamma correction in raw color space. The leftmost applies `daylight multiplier` in `Canon1DsMkIII_0199.json`. The others shows the variation of different illuminant estimation methods in Ref. [1].
![](outcomparison.png)

## Reference
[1] J. van de Weijer, T. Gevers and A. Gijsenij, "Edge-Based Color Constancy," in IEEE Transactions on Image Processing, vol. 16, no. 9, pp. 2207-2214, Sept. 2007.