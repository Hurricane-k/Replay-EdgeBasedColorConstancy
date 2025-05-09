import numpy as np
from scipy import ndimage
from scipy.spatial.distance import minkowski

"""Reference
J.V.D. Weijer, T. Gevers, and A. Gijsenji, "Edge-Based Color Constancy," IEEE Trans. Img. Processing. 16(9). 2007
"""

def EBCCIMG(img_input,mode='GW',sigma=7,order=None,p=4):
    """edged-based color constancy (low-level statisitc-based color constancy)

    Args:
        img_input (_np.ndarray_): _2D or 3D_
        mode (str, optional): _the mode of color constancy_. Defaults to 'GW'.
            ['GW','maxRGB','SoG','GGW','GE1','GE2','ME']
            GE: Grey World
            maxRGB: max RGB
            SoG: Shade of Grey
            GGW: General Grey World
            GE1: 1st-order Grey Edge
            GE2: 2nd-order Grey Edge
            ME: Max Edge
        sigma (int, optional): _sigma of Gaussian Filter (scipy.ndimage.gaussian_filter)_. Defaults to 7.
        order (_type_, optional): _the order of derivative (only support 1st and 2nd order)_. Defaults to None.
        p (int, optional): _minkowski_. Defaults to 4.

        sigma, order, p derived from Table 1. from "Edge-Based Color Constancy"

        Another Reference:
        I. Erba, M. Buzzelli, and R. Schettini, "RGB color constancy using multispectral pixel information," JOSA A. 41(2) 2024.
            GW:  n = 0, p = 1,      sigma = 0
            WP:  n = 0, p = np.inf, sigma = 0
            SoG: n = 0, p = 4,      sigma = 0
            GGW: n = 0, p = 9,      sigma = 9
            GE1: n = 1, p = 1,      sigma = 6
            GE2: n = 2, p = 1,      sigma = 6

    Returns:
        _IllumEst_: illuminant estimation based on edge-based color constancy
    """


    if mode == 'GW': #Grey World 
        IllumEst = np.mean(img_input,axis=(0,1))
    elif mode == 'maxRGB': # maxRGB
        IllumEst = np.max(img_input,axis=(0,1))
    elif mode == 'SoG': # Shade of Grey
        IllumEst = MinkowskiIMG(img_input, p=p)
    elif mode == 'GGW': # General Gray World
        img_filtered = ScaleIMG(img_input, sigma = sigma)
        IllumEst = MinkowskiIMG(np.abs(img_filtered), p = p)
    elif mode == 'GE1': # Grey World
        img_filtered = ScaleIMG(img_input, sigma = sigma)
        img_deriv = DerivativeIMG(img_filtered, order=1)
        IllumEst = MinkowskiIMG(img_deriv, p = p)
    elif mode == 'ME': # Max Edge
        img_filtered = ScaleIMG(img_input, sigma = sigma)
        img_deriv = DerivativeIMG(img_filtered, order=1)
        IllumEst = np.max(img_deriv,axis=(0,1))
    elif mode == 'GE2': # 2nd order Grey World
        img_filtered = ScaleIMG(img_input, sigma = sigma)
        img_deriv = DerivativeIMG(img_filtered, order=2)
        IllumEst = MinkowskiIMG(img_deriv, p = p)
    else:
        TypeError('Invalid input for input params')

    IllumEst_norm = IllumEst/np.linalg.norm(IllumEst)

    return IllumEst_norm

def MinkowskiIMG(img_input, p=2):
    """ RGB white point based on MinkowskiIMG

    Args:
        img_input (_np.ndarray_): 2D or 3D
        p (int, optional): _Minkowski norm (p)_. Defaults to 2.

    Returns:
        _illuminant estimation_: _description_
    """

    if img_input.ndim == 2:
        u = img_input.reshape(-1).astype(np.float64)
        v = np.zeros_like(u,dtype=np.float64)
        dist_mink = minkowski(u,v,p)
    elif img_input.ndim == 3:
        dist_mink = np.zeros(img_input.shape[2],dtype=np.float64)
        for i in range(img_input.shape[2]):
            u = img_input[...,i].reshape(-1).astype(np.float64)
            v = np.zeros_like(u,dtype=np.float64)
            dist_mink[i] = minkowski(u,v,p)

    return dist_mink


def ScaleIMG(img_input, sigma = 7):
    """ gaussian filter for scaling

    Args:
        img_input (_np.ndarray_): 2D or 3D
        sigma (int, optional): sigma of Gaussian Filter. Defaults to 7.

    Returns:
        img_filtered: img after filter
    """

    assert sigma>0, "the parameter sigma must be a positive float \nRefers to scipy.ndimage.gaussian_filter"

    if img_input.ndim == 2:
        img_filtered = ndimage.gaussian_filter(img_input, sigma=sigma)
    elif img_input.ndim == 3:
        img_filtered = np.zeros_like(img_input, dtype=np.float32)
        for i in range(img_input.shape[-1]):
            img_filtered[...,i] = ndimage.gaussian_filter(img_input[...,i], sigma=sigma)
    else:
        TypeError('Invalid input parameters')

    return img_filtered

def DerivativeIMG(img_input, order=1):
    """ derivative of image (2-order / 1-order)

    Args:
        img_input ( np.array ): 2D or 3D
        order (int, optional): the order of derivative. Defaults to 1. [1,2]

    Returns:
        gradient_map: share the same shape with img_input
    """

    assert order in [1,2], "stop passing invalid input to the parameter order"

    if img_input.ndim == 2 and order == 1:
        sobel_y = ndimage.sobel(img_input,axis=0)
        sobel_x = ndimage.sobel(img_input,axis=1)
        gradient_map = np.sqrt(sobel_x**2 + sobel_y**2)
    elif img_input.ndim == 3 and order == 1:
        gradient_map = np.zeros_like(img_input,dtype=np.float32)
        for i in range(img_input.shape[-1]):
            sobel_y = ndimage.sobel(img_input[...,i],axis=0)
            sobel_x = ndimage.sobel(img_input[...,i],axis=1)
            gradient_map[...,i] = np.sqrt(sobel_x**2 + sobel_y**2)
    elif img_input.ndim == 2 and order == 2:
        gradient_map = ndimage.laplace(img_input)
    elif img_input.ndim == 3 and order == 2:
        gradient_map = np.zeros_like(img_input,dtype=np.float32)
        for i in range(img_input.shape[-1]):
            gradient_map[...,i] = ndimage.laplace(img_input[...,i])
    else:
        TypeError('the valid value of input parameters')

    return gradient_map