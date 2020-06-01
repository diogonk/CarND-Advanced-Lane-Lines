import numpy as np
import cv2

offset_y = 50

def hls_select_threshold(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel

    b_channel = img[:,:,0]
    g_channel = img[:,:,1]
    s_channel = hls_img[:,:,2]
    binary_b = np.zeros_like(b_channel)
    binary_g = np.zeros_like(g_channel)

    binary_b[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    binary_g[(g_channel > thresh[0]) & (g_channel <= thresh[1])] = 1

    binary_output = np.zeros_like(s_channel)

    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    binary_output = np.bitwise_or(binary_output, binary_b, binary_g)

    out_img = np.dstack((binary_output, binary_output, binary_output)) * 255
    blur_img = blur(out_img, ksize=3)

    # 3) Return a binary image of threshold result
    return blur_img

def blur(img, ksize=3):
    return cv2.medianBlur(img, ksize=3)

def gray_convert(img, cmap=cv2.COLOR_BGR2GRAY):
    #conver image to gray
    return cv2.cvtColor(img, cmap)

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # get the edges in the horizontal direction
    abs_sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # get the edges in the vertical direction
    abs_sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the edge magnitudes
    #mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_dir = np.arctan2(abs_sobely,abs_sobelx)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(grad_dir)/255
    grad_dir = (grad_dir/scale_factor).astype(np.uint8)

    sbinary = np.zeros_like(grad_dir)
    sbinary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return sbinary