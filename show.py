"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 4 -- Advanced Lane Finding, demo script 

 Author: Ciaran Murphy
 Date: 7th Nov 2017

"""

import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lane_finding import *

CBOARD_IMGS_PATH = './camera_cal/'
CBOARD_IMG_PATTERN = 'cal*.jpg'

def show():
    """Main function that calls all others"""

    cboard_file = 'camera_cal/calibration9.jpg'
    img_file1 = './test_images/straight_lines1.jpg'
    img_file2 = './test_images/test1.jpg'

    show_chessboard(cboard_file)
    show_rgb_color_thresh(img_file1)
    show_hls_color_thresh(img_file1)
    show_abs_mag_dir_thresh(img_file1)
    #show_lane_lines(img_file1)

    show_rgb_color_thresh(img_file2)
    show_hls_color_thresh(img_file2)
    show_abs_mag_dir_thresh(img_file2)
    #show_lane_lines(img_file2)


def show_chessboard(img_file):
    """Show examples of camera calibration, distoration and prespective transform in action """
    
    # Create a figure to display the images inside
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    f.canvas.set_window_title('show.show_chessboard()')

    # Plot the original image
    img = cv2.imread(img_file)
    ax1.imshow(img)
    ax1.set_title("Original", fontsize=30)

    # Chessboard corner detection
    board_size = (9,6)
    img, nx, ny, corners = get_cboard_points(img, [], [], img_file, board_size)
    cv2.drawChessboardCorners(img, (nx, ny), corners, True)
    ax2.imshow(img)
    ax2.set_title('Corner Detection', fontsize=30)

    # Distortion Correction
    udist_img = undistort(img) 
    ax3.imshow(udist_img)
    ax3.set_title("Undistorted", fontsize=30)

    # Warp to front facing perspective
    # The offset value below is found by manually eye-balling the sample image `img_file` 
    # and is further tweaked below in `dst`. It's here just as an example.
    img_size = (udist_img.shape[1], udist_img.shape[0])
    src = np.float32([
        corners[0], 
        corners[nx-1], 
        corners[-1], 
        corners[-nx]
    ])

    dst = np.float32([(125, 125), (1200, 125), (1200, 575), (125, 575)])
    warped_img = warp(udist_img, src, dst, offset=125)
    ax4.imshow(warped_img)
    ax4.set_title("Undistorted & Warped", fontsize=30)
   
    # Show the generated image
    plt.tight_layout()
    plt.show()


def show_abs_mag_dir_thresh(img_file):
    """ Show sobel, magnitude, directional and combined thresholding """

    ksize = 3
    gradx_thresh = (20, 100)
    grady_thresh = (20, 100)
    mag_thresh = (70, 100)
    dir_thresh = (0.7, 1.3)

    img = cv2.imread(img_file)

    gradx = abs_sobel_threshold(img, orient='x', ksize=ksize, thresh=gradx_thresh)
    grady = abs_sobel_threshold(img, orient='y', ksize=ksize, thresh=grady_thresh)
    magnitude = mag_threshold(img, ksize=ksize, thresh=mag_thresh)
    directional = dir_threshold(img, ksize=ksize, thresh=dir_thresh)

    combined = np.zeros_like(directional)
    combined[((gradx == 1) & (grady == 1)) | ((magnitude == 1) & (directional == 1))] = 1
    
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30,10))
    f.canvas.set_window_title('show.show_abs_mag_dir_thresh()')

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original')

    ax2.imshow(gradx, cmap='gray')
    ax2.set_title('GradX')

    ax3.imshow(grady, cmap='gray')
    ax3.set_title('GradY')

    ax4.imshow(magnitude, cmap='gray')
    ax4.set_title('Magnitude')

    ax5.imshow(directional, cmap='gray')
    ax5.set_title('Directional')

    ax6.imshow(combined, cmap='gray')
    ax6.set_title('Combined')

    plt.tight_layout()

    plt.show()


def show_rgb_color_thresh(img_file):
    """ Show RGB color thresholding """

    r_thresh = (200, 255)
    g_thresh = (200, 255)
    b_thresh = (200, 255)

    img = cv2.imread(img_file)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 10))
    f.canvas.set_window_title('show.show_rgb_color_thresh()')

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original')

    ax2.imshow(color_threshold(img, tscheme='RGB', channel='R', thresh=r_thresh), cmap='gray')
    ax2.set_title('RGB, R channel')

    ax3.imshow(color_threshold(img, tscheme='RGB', channel='G', thresh=g_thresh), cmap='gray')
    ax3.set_title('RGB, G channel')

    ax4.imshow(color_threshold(img, tscheme='RGB', channel='B', thresh=b_thresh), cmap='gray')
    ax4.set_title('RGB, B channel')

    plt.tight_layout()
    
    plt.show()


def show_hls_color_thresh(img_file):
    """ Show HLS color thresholding """

    h_thresh = (15, 100)
    l_thresh = (180, 255)
    s_thresh = (90, 255)

    img = cv2.imread(img_file)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 10))
    f.canvas.set_window_title('show.show_hls_color_thresh()')

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original')

    ax2.imshow(color_threshold(img, tscheme='HLS', channel='H', thresh=h_thresh), cmap='gray')
    ax2.set_title('HLS, H channel')

    ax3.imshow(color_threshold(img, tscheme='HLS', channel='L', thresh=l_thresh), cmap='gray')
    ax3.set_title('HLS, L channel')

    ax4.imshow(color_threshold(img, tscheme='HLS', channel='S', thresh=s_thresh), cmap='gray')
    ax4.set_title('HLS, S channel')

    plt.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    show()

