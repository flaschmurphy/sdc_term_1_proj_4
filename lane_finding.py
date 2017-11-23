"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 4 -- Advanced Lane Finding, main script

 Author: Ciaran Murphy
 Date: 7th Nov 2017

 This script is by no means optimized! It's super slow... given that it's just an educational tool really.

"""

import os
import sys
import glob
import pickle
from moviepy.editor import VideoFileClip
from argparse import ArgumentParser

import cv2
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CBOARD_IMGS_PATH = './camera_cal'
CBOARD_IMG_PATTERN = 'cal*.jpg'
INPUT_VIDEO = './project_video.mp4'
OUTPUT_VIDEO = './project_video_out.mp4'
DEBUG_DIR = './test_images'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='pipeline_input', 
            help="""Video or image to pass through the processing pipeline. Defaults to `INPUT_VIDEO` 
            as configured inside this script. Supported media types are mp4, jpg and png.""")
    parser.add_argument('-o', '--output', dest='pipeline_output', 
            help="""Destination to store output. If an mp4 video filename is given, a new video of 
            that name will be created. If passing in a single image, specify a directory name 
            here and images will be created in that directory for each stage of the pipeline. 
            Defaults to `OUTPUT_VIDEO` as configured inside this script.""")
    parser.add_argument('-d', '--debug', dest='debug_mode',
            help="""Optional. If included and >-1, turn on debug output which will write all intermediary 
            images to ./output_images for every N frames.""")
    parser.add_argument('-s', '--stream', dest='stream', action='store_true',
            help="""Optional. If included, constantly update an image called ./output.jpg 
            with the current pipeline. You can then use eg `feh -R 0.2 output.jpg` to monitor progress.""")
    parser.add_argument('-0', '--t0', dest='start',
            help="""Optional. Clip an input video to start at t0 seconds.""")
    parser.add_argument('-1', '--t1', dest='end',
            help="""Optional. Clip an input video to end at t1 seconds.""")

    args = parser.parse_args()

    if args.pipeline_input is None or args.pipeline_output is None:
        parser.print_help()
        sys.exit()

    if args.debug_mode is None:
        args.debug_mode = -1
    else:
        args.debug_mode = int(args.debug_mode)

    return args


def main():
    """Entry point. Open the inpt video, run each image through the pipeline and save the new video 
    as output"""
    global __args

    __args = parse_args()
    if int(__args.debug_mode) > -1:
        cv2.imwrite('output.jpg', np.zeros((32, 32, 3)))

    input_data = __args.pipeline_input
    output_dest = __args.pipeline_output

    if input_data.split('.')[-1] == 'mp4':
        vin = VideoFileClip(input_data)

        if __args.start is not None:
            vin = vin.set_start(float(__args.start))
        if __args.end is not None:
            vin = vin.set_end(float(__args.end))

        vout = vin.fl_image(pipeline)
        vout.write_videofile(output_dest, audio=False)

    elif input_data.split('.')[-1] in ['jpg', 'png']:
        img = cv2.imread(input_data)

        if os.path.isfile(output_dest):
            raise Exception('{} exists and is a file, not a directory.'.format(output_dest))

        if not os.path.isdir(output_dest):
            os.mkdir(output_dest)

        pipeline(img, dest=output_dest, fname=os.path.basename(input_data))

    else:
        raise Exception('Invalid input media format. Supported types are mp4, jpg or png.')


def pipeline(img, dest='./output_images', fname=None, cmap='BGR'):
    """Pipeline for processing each image in the sequence. The goals of this script are to walk 
    through the following steps for each image in the video:

        (1) Compute the camera calibration matrix and distortion coefficients given a set of 
            chessboard images.
        (2) Apply a distortion correction to raw images.
        (3) Use color transforms, gradients, etc., to create a thresholded binary image.
        (4) Apply a perspective transform to rectify binary image ("birds-eye view").
        (5) Detect lane pixels and fit to find the lane boundary.
        (6) Determine the curvature of the lane and vehicle position with respect to center.
        (7) Warp the detected lane boundaries back onto the original image.
        (8) Output visual display of the lane boundaries and numerical estimation of lane 
            curvature and vehicle position.
    
    Args:
        img: the input image in BGR format
        dest: directory to save images to when in debug mode
        fname: file name to use as the base for saving snapshots
        cmap: color format, must be either RGB or BGR (cv2.imread() uses BGR)

    Returns:
        final: the processed image with overlay of lane lines, curvature, etc

    """
    assert cmap in ['BGR', 'RGB'], \
        'Invalid image color format specified. Valid options are "RGB" or "BGR", but got {}'.format(cmap)

    if __args.debug_mode > 0 and not os.path.exists(dest):
        os.mkdir(dest)

    # Add a counter to track the number of calls to `pipeline()` which is the same as the current frame number
    if 'counter' not in pipeline.__dict__:
        pipeline.counter = 0
    else:
        pipeline.counter += 1

    if __args.debug_mode > 0 and fname is None:
        fname = 'frame_{:05}.jpg'.format(pipeline.counter, '.jpg')

    # Create helper lambdas for easily adding text to images later on
    # Params: [img, txt, (pos_x, pos_y), font_scale, color_rgb, line_type]
    add_text = lambda params: cv2.putText(
            params[0], params[1], params[2], cv2.FONT_HERSHEY_DUPLEX, params[3], params[4], params[5])

    add_text_defaults = lambda img, txt: add_text([img, txt, (25, 150), 5, (255,255,255), 3])

    # Steps (1) and (2): calibration (happens automatically) and distortion correction
    undist = undistort(img)

    # Step (3): thresholding to produce a binary image
    color_r = color_threshold(undist, tscheme='RGB', channel='R', thresh=(220, 255))
    color_g = color_threshold(undist, tscheme='RGB', channel='G', thresh=(200, 255))
    color_h = color_threshold(undist, tscheme='HSV', channel='H', thresh=(20, 100))
    color_v = color_threshold(undist, tscheme='HSV', channel='V', thresh=(210, 256))
    sobel   = sobel_threshold(undist, orient='x', ksize=3, thresh=(20, 100))

    direct  = dir_threshold(undist, ksize=3, thresh=(0.7, 1.3))
    mag     = mag_threshold(undist, ksize=3, thresh=(20, 100))

    # The L channel in HLS picks up distant lane lines, in particular when the road
    # surface is different in the foreground than the background. However it also
    # picks up a lot of noise in the foreground. This noise can be removed using a 2D
    # convolution with the code below, but in the end I removed HLS color space altogether
    # in favor of HSV instead. I leave the code below for reference though.
    #kernel = np.array([[-1, -1, -1], [-1, 5, -1], [-1, -1, -1]])
    #color_l_conv = cv2.filter2D(color_l, -1, kernel)
    #color_l_conv[color_l_conv > 0] = 1

    # Combine all images
    right = (sobel | color_g | color_v) ^ color_h 
    left = ((color_h | sobel | color_r) & mag) ^ right
    combined = left | right
    #combined = (color_r | color_g | color_h | color_v) | sobel | (direct & mag)

    # Step (4): perspective transform
    binary_warped = warp(combined)

    # Steps (5) and (6): detect lane lines and determine curvature
    # Incase of exceptions being raised while generating the poly lines, store some 
    # debug help to disk for offline analysis. 
    color_warp, line_search_img, poly_left, poly_right, left_curve_rad, right_curve_rad, center = \
            find_lane_lines(binary_warped, img)

    # Step (7) and (8): warp the detected lane lines back onto the original image and visualize the results
    # Unwarp and combine the result with the original image
    unwarped = warp(color_warp, inverse=True)
    unwarped_withlines = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)

    # If snapshot is enabled, then dump all images to disk for manual inspection later on
    if __args.debug_mode > 0 and pipeline.counter % __args.debug_mode == 0:
        i = 1
        fname = fname.split('.')[0]
        _undist_rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'undistorted.jpg')), _undist_rgb); i+=1 
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'color_h.jpg')), color_h*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'color_v.jpg')), color_v*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'color_r.jpg')), color_r*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'color_g.jpg')), color_g*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'sobel.jpg')), sobel*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'directional.jpg')), direct*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'magnitude.jpg')), mag*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'combined.jpg')), combined*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'warp_binary.jpg')), binary_warped*255); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'rectangles.jpg')), line_search_img); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'lane.jpg')), color_warp); i+=1
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'withlines.jpg')), unwarped_withlines); i+=1

    # Add the descriptddive text to the main result image
    txt = 'Left Curvature: {:5}m | Right Curvature: {:5}m'.format(int(left_curve_rad), int(right_curve_rad))
    params = (unwarped_withlines, txt, (25, 50), 1, (255, 255, 255), 2)
    unwarped_withlines = add_text(params)

    txt = 'Distande from Center: {:0.2f}m'.format(center)
    params = (unwarped_withlines, txt, (25, 90), 1, (255, 255, 255), 2)
    unwarped_withlines = add_text(params)

    # Now create a composite image that shows all the main stages of the pipeline. Each
    # time this pipeline is called a new image will be returned and when run through moviepy
    # an output video will be created that shows all content.
    top = np.concatenate(
             (
                 add_text_defaults(cv2.cvtColor(color_r*255, cv2.COLOR_GRAY2BGR), 'R'), 
                 add_text_defaults(cv2.cvtColor(color_g*255, cv2.COLOR_GRAY2BGR), 'G'),
                 add_text_defaults(cv2.cvtColor(color_h*255, cv2.COLOR_GRAY2BGR), 'H'),
                 add_text_defaults(cv2.cvtColor(color_v*255, cv2.COLOR_GRAY2BGR), 'V'),
                 add_text_defaults(cv2.cvtColor(sobel*255, cv2.COLOR_GRAY2BGR), 'Sobel'),
                 add_text_defaults(cv2.cvtColor(direct*255, cv2.COLOR_GRAY2BGR), 'Dir'),
             ), axis=1)
    top = imresize(top, (undist.shape[0]//4, undist.shape[1]))

    bottom = np.concatenate(
            (
                 add_text_defaults(cv2.cvtColor(left*255, cv2.COLOR_GRAY2BGR), 'Left'),
                 add_text_defaults(cv2.cvtColor(right*255, cv2.COLOR_GRAY2BGR), 'Right'),
                 add_text_defaults(cv2.cvtColor(combined*255, cv2.COLOR_GRAY2BGR), 'Combined'),
                 add_text_defaults(cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR), 'Bird'),
                 add_text_defaults(line_search_img, 'Search'),
                 add_text_defaults(color_warp, 'Lane'),
             ), axis=1)
    bottom = imresize(bottom, (undist.shape[0]//4, undist.shape[1]))

    final = np.concatenate((top, unwarped_withlines, bottom), axis=0)

    if __args.debug_mode > 0 and pipeline.counter % __args.debug_mode == 0:
        _final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.sep.join((dest, fname + '-' + str(i) + '-' + 'final.jpg')), _final_rgb); i+=1

    # If we got this far, then all is more or less ok. So save the result and return.
    pipeline.previous = final

    if __args.stream:
        # From the shell, use `feh -R 0.2 ./output.jpg` to watch the render in realtime
        cv2.imwrite('output.jpg', cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

    return final


def get_cboard_points(img, objpoints, imgpoints, fname, board_size, cmap='BGR'):
    """Find the object and image points for a single chessboard image and append them to the supplied lists.

    Args:
        img: chessboard image
        objpoints: list of object points in 3d space. This list is updated in-place.
        imgpoints: list of image points in 2d space. This list is updated in-place.
        fname: filename of the image being supplied.
        cmap: color format, must be either RGB or BGR (cv2.imread() uses BGR)
        board_size: 2-tuple giving the number of internal chessboard in the x & y directions

    Returns:
        img: new image with the chessboard points marked on it
        nx: the number of squares along the x axis
        ny: the number of squares along the y axis
        corners: the corners of the detected squares

    """
    assert cmap in ['BGR', 'RGB'], \
            'Invalid image color format specified. Valid options are "RGB" or "BGR", but got {}'.format(cmap)

    nx = board_size[0] # number of inside corners in x direction
    ny = board_size[1] # number of inside corners in y direction

    # Prepare object points that look like below. Note that the 3rd
    # dimension is always zero since the image is a 2d plane
    #
    # [[ 0.  0.  0.]
    #  [ 1.  0.  0.]
    #  [ 2.  0.  0.]
    #  [ 3.  0.  0.]
    #  [ 4.  0.  0.]
    #  ....
    #  [ 3.  5.  0.]
    #  [ 4.  5.  0.]
    #  [ 5.  5.  0.]
    #  [ 6.  5.  0.]
    #  [ 7.  5.  0.]]
    #
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates

    # Convert to gray in preparation for cv2.findChessboardCorners()
    if cmap == 'BGR':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    print('Finding chess board corners for image: {}'.format(fname))
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    assert ret is True, 'Warning: cv2.findChessboardCorners() returned {} for image {}'.format(ret, fname)

    # Add object points and image points
    imgpoints.append(corners)
    objpoints.append(objp)

    return img, nx, ny, corners


def get_all_cboard_points(board_size):
    """Create two lists containing all image and object points detected in the chessboard calibration image set.

    Args:
        board_size: 2-tuple giving the number of internal chessboard in the x & y directions

    Returns:
        objpoints: list of detected object points (3d real world points)
        imgpoints: list of detected image points (2d image plane points)
        img_shape: the shape of the images (required for cv2.calibrateCamers())

    """
    cboard_images = glob.glob(os.sep.join((CBOARD_IMGS_PATH, CBOARD_IMG_PATTERN)))

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    for f in cboard_images:
        img = cv2.imread(f)
        try:
            get_cboard_points(img, objpoints, imgpoints, f, board_size)
        except AssertionError as e:
            print(e)

    return objpoints, imgpoints, img.shape


def undistort(img, cmap='BGR'):
    """Undistort an image. If the calibration has not been performed already, perform it now but
    only once for the lifetime of this python process.

    This function leverages two less well known Python features: embeded functions (a function within
    a function) and the fact that attributes can be assigned to Python functions (for example, see the
    attribute `calibrate.ready` below). An interesting characteristic of method attributes is that they
    are persistent across multiple calls of that process, thus can be used to ensure that a particular
    block of code runs once and only once for the lifetime of a Python program.

    Args:
        img: the image to undistort. If not in BGR, must specify RGB as cmap
        cmap: 'RGB' or 'BGR', defaults to 'BGR' inline with cv2.imread()

    Returns:
        udist_img: an undistorted version of img.

    """

    assert cmap in ['BGR', 'RGB'], \
            'Invalid image color format specified. Valid options are "RGB" or "BGR", but got {}'.format(cmap)

    def calibrate(board_size=(9,6)):
        """Compute image calibration data using a set of chessboard images

        Args:
            board_size: 2-tuple giving the number of internal chessboard in the x & y directions

        Returns:
            mtx: camera matrix for converting 3d object points to 2d image points
            dist: distortion cooeficients
            rvecs: rotation vector (position in real world)
            tvecs: translation vector (position in real world)

        """
        objpoints, imgpoints, img_shape = get_all_cboard_points(board_size)

        print("Generating calibration data...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[0:2][::-1], None, None)

        undistort.calibrated = True
        print("Done.")
        return mtx, dist, rvecs, tvecs


    # If calibration data is not available, compute it now
    if 'calibrated' not in undistort.__dict__:
        cal_data_fname = os.sep.join((CBOARD_IMGS_PATH, 'caldata.pckle'))
        if os.path.exists(cal_data_fname):
            caldata = pickle.load(open(cal_data_fname, 'rb'))
            mtx, dist, rvecs, tvecs = caldata['data']
            if 'calmsg' not in undistort.__dict__:
                print('Found and loaded cached calibration data from disk...')
                undistort.calmsg = None

        else:
            print("Calibration data is not available. Generating it now...")
            mtx, dist, rvecs, tvecs = calibrate()
            caldata = {'data': [mtx, dist, rvecs, tvecs]}
            pickle.dump(caldata, open(cal_data_fname, 'wb'))
            print('Cached calibration data to disk as {}'.format(cal_data_fname))

        undistort.cal_data = [mtx, dist, None, mtx]
        undistort.rvecs = rvecs
        undistort.tvecs = tvecs

    # Return the un-distorted the image
    return cv2.undistort(img, *undistort.cal_data)


def color_threshold(img, tscheme='HSV', cmap='BGR', channel='S', thresh=None):
    """Convert an image to `tscheme`, then to gray, and then filter out pixels that fall within `thresh` range.

    Args:
        img: the original image
        tscheme: the target color scheme to use for the thresholding, can be either 'HSV' or 'RGB'
        cmap: the input color map, can be either 'RGB' or 'BGR'
        channel: which channel to apply the threshold to - can be either 'H', 'S', 'V', 'R', 'G', or 'B'.
        thresh: 2-tuple specifying the min and max values. Pixels outside this range 
            will be black in the output. Pixels inside the range (inclusive) will be white.

    Returns:
        binary: a binary image with pixels within the thresholds white, all others black.

    """
    assert thresh is not None, "Must specify a threshold. See this function's help."
    assert cmap in ['BGR', 'RGB'], 'Invalid input color map, choose either BGR or RGB.'
    assert tscheme in ['HSV', 'RGB'], 'Invalid target color scheme, choose either HSV or RGB.'
    assert channel in ['R', 'G', 'B', 'H', 'S', 'V'], 'Invalid target channel for color map.'

    if cmap == 'BGR':
        if tscheme == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif tscheme == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cmap == 'RGB':
        if tscheme == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if tscheme == 'HSV':
        ch1 = img[:,:,0] # Hue channel
        ch2 = img[:,:,1] # Saturation channel
        ch3 = img[:,:,2] # Value channel

    else:
        ch1 = img[:,:,0] # Red channel
        ch2 = img[:,:,1] # Green channel
        ch3 = img[:,:,2] # Blue channel

    channel_select = {'H': ch1, 'S': ch2, 'V': ch3, 
                      'R': ch1, 'G': ch2, 'B': ch3}

    binary = np.zeros_like(ch3)
    thresh_min, thresh_max = thresh[0], thresh[1]                          
    binary[(channel_select[channel] >= thresh_min) & (channel_select[channel] <= thresh_max)] = 1

    # OpenCV's Morphological Transformations can help a lot with removing 
    # unwanted noise. See https://goo.gl/XFznnv for details of how this works.
    kernel = np.ones((2,2),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def sobel_threshold(img, cmap='BGR', orient='x', ksize=3, thresh=(20, 100)):
    """Compute the absolute value of the sobel of an image, then output a binary image where the 
    absolute value at each pixel is within a threshold range specified by `thresh` (inclusive).

    Args:
        img: the original image
        cmap: color map for the original image, should be either 'BGR' or 'RGB'
        orient: whether to perform sobel along the x or y axis
        ksize: the kernel size for the Sobel operation
        thresh: 2-tuple specifying the min and max pixel values. Pixels outside this range 
            will be black in the output. Pixels inside the range (inclusive) will be white.

    Returns:
        binary: new binary image

    """

    assert orient in ['x', 'y'], 'Invalid orient, chose from either x or y.'
    assert cmap in ['BGR', 'RGB'], 'Invalid color map, choose either BGR or RGB'

    # Convert to grayscale
    if cmap == 'BGR':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply cv2.Sobel()
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Take the absolute value of the output from cv2.Sobel()
    abs_sobel = np.absolute(sobel)

    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply lower and upper thresholds
    binary = np.zeros_like(scaled_sobel)
    thresh_min, thresh_max = thresh[0], thresh[1]
    binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # OpenCV's Morphological Transformations can help a lot with removing 
    # unwanted noise. See https://goo.gl/XFznnv for details of how this works.
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def mag_threshold(img, cmap='BGR', ksize=3, thresh=(20, 100)):
    """Compute the magnitude of the gradient of an image and output a binary image where the magnitude
    at each pixel is within the threshold range (inclusive).

    Args:
        img: the original image
        cmap: color map for the original image, should be either 'BGR' or 'RGB'
        ksize: the kernel size for the Sobel operation
        thresh: 2-tuple specifying the min and max pixel values. Pixels outside this range 
            will be black in the output. Pixels inside the range (inclusive) will be white.

    Returns:
        binary: new binary image

    """

    assert cmap in ['BGR', 'RGB'], 'Invalid color map, choose either BGR or RGB'

    # Convert to grayscale
    if cmap == 'BGR':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply cv2.Sobel() in both x and y according to the supplied kernel size
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Get the absolute value of the Sobels
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))

    # Apply lower and upper thresholds
    binary = np.zeros_like(scaled_sobel)
    thresh_min, thresh_max = thresh[0], thresh[1]
    binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # OpenCV's Morphological Transformations can help a lot with removing 
    # unwanted noise. See https://goo.gl/XFznnv for details of how this works.
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def dir_threshold(img, cmap='BGR', ksize=3, thresh=(0.7, 1.3)):
    """Compute the direction of the gradient of an image and output a binary image where the direction
    at each pixel is within the threshold range (inclusive).

    Args:
        img: the original image
        cmap: color map for the original image, should be either 'BGR' or 'RGB'
        ksize: kernel size for the Sobel operation (must be an odd number)
        thresh: 2-tuple specifying the min and max values (in radians). Pixels outside this range 
            will be black in the output. Pixels inside the range (inclusive) will be white.

    Returns:
        binary: np.uint8 binary image with pixels within the thresholds white, all others black

    """
    assert cmap in ['BGR', 'RGB'], 'Invalid color map, choose either BGR or RGB'
    
    # Convert to grayscale
    if cmap == 'BGR':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:    
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)                   
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Take the absolute value of the x and y gradients
    absx = np.absolute(sobelx)
    absy = np.absolute(sobely)
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arct2 = np.arctan2(absy, absx)
    
    # Create a binary mask where direction thresholds are met
    binary = np.zeros_like(arct2)
    thresh_min, thresh_max = thresh[0], thresh[1]                          
    binary[(arct2 >= thresh_min) & (arct2 <= thresh_max)] = 1 

    # OpenCV's Morphological Transformations can help a lot with removing 
    # unwanted noise. See https://goo.gl/XFznnv for details of how this works.
    #kernel = np.ones((3,3),np.uint8)
    #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Return this mask as your binary_output image
    return np.uint8(binary)


def warp(img, src=None, dst=None, inverse=False, offset=0):
    """Warp an image so that it looks like the camera took the image from a different angle. For 
    example, this could mean converting a front facing camera image to a top-down birds-eye-view camera.

    Args:
        img: the original image that will be warped
        src: 4-tuple of points in the source image. If left as `None`, a default 
            will be created inside the method.
        dst: 4-tuple of points to map src to in the destination image. If left as `None`, 
            a default will be created inside the method.
        inverse: if True, perform an inverse warp from dst to src instead of the other way around
        offset: offset for the destination image

    Returns:
        warped: a warped img where all points are mapped to the calculated perspective transform matrix

    """

    # The values below were obtained manually by eyeballing one particilar image.
    # It's not ideal to hard code this transformation, but for this project it's good enough.
    if src is None and dst is None:
        src = np.array(((521, 504), (771, 504), (1022, 670), (280, 670)), dtype="float32")
        dst = np.array(((290, 504), (1032, 504), (1022, 670), (280, 670)), dtype="float32")

    img_size = (img.shape[1], img.shape[0])
    if inverse is False:
        M = cv2.getPerspectiveTransform(src, dst) # calculate the perspective transform matrix
    else:
        M = cv2.getPerspectiveTransform(dst, src) # calculate the perspective transform matrix
    warped = cv2.warpPerspective(img, M, img_size)

    return warped


def find_lane_lines(binary_warped, orig):
    """Find the lane lines in the input using a sliding window approach and calculate their curvature.

    This code was taken from Udacity's Advanced Lane Finding project section 33, in term 1
    of the Self Driving Car nanodegree.

    The approach is to take a histogram over the bottom half of the image and find
    the peaks in the histogram on the left and right sides. Geven the preprocessing
    that has already happened on the image, these peaks can be reasonably assumed
    to correspond to the beginning of the lane lines at the very bottom of the
    image. Once these starting points have been established it's a matter of
    dividing the image up into small sections (9 of them in this case, see
    `nwindows`) and searching through each section to find the continuation of the
    lane lines, moving upwards through the image until we reach the top. Now that
    we have 9 rectangular boxes where we know the left lane line is contained, and
    the same for the right lane line, we can fit a polynomial for each lane line
    that passes through each rectangular box using numpy's `polyfit()` method. 

    Args:
        binary_warped: the image containing the lane lines. Assumes it has
            already been preprocessed for warp and distortion.
        orig: the original image as loaded from disk or video. This is saved back to disk in case
            of a failure getting polynomial lines, to be used for debugging.

    Returns:
        lane_lines: image showing warped version of lane lines
        poly_left: numpy polyfit for the left lane line
        poly_right: numpy polyfit for the right lane line
        left_curve_rad: the curvature of the left lane line in meters
        right_curve_rad: the curvature of the right lane line in meters
        dist_from_center: distance from the lane center in meters

    """
    def sanity_check():
        #
        # Sanity checks on the polynomials
        # 
        # I tried to do things like check for sudden changes in curvature relative to the last
        # image, or big differences in left and right curvature. But the best thing seems to be to
        # try as hard as possible to get clean images through the system, and then only check the
        # that the rate of change of the left and right x (horizontal) values are not very different. 
        # In otherwords, take the derivative of the left and right x values with respect to y
        # and if they are very different, rescan the entire image instead of just the margins. If 
        # the sanity check still fails after that, then reuse the last lane finding results. The 
        # color thresholding will work pretty well on it's own, even without this sanity check
        #

        # This function takes no args as all the data it needs are contained in the parent function
        # (so no globals needed either)

        get_dx_left = lambda: np.mean( 2*poly_left[0]*ploty + poly_left[1] )
        get_dx_right = lambda: np.mean( 2*poly_right[0]*ploty + poly_right[1] )

        # On the 1st image in the video there will be no history
        if 'previous_dx_left' not in find_lane_lines.__dict__:
            find_lane_lines.previous_dx_left = get_dx_left()
            find_lane_lines.previous_dx_right = get_dx_right()
            return 'LEFTRIGHT'
            
        # Configure some thresholds to test against. If any of these numbers are breached
        # it will trigger a full image scan with fallback to using the previous image's data
        left_tollerance_instantaneous = 0.8
        left_tollerance_past = 1
        right_tollerance_instantaneous = 0.8
        right_tollerance_past = 1

        # Calculate the rate of change of left and right lanes  (x wrt y)
        dx_left = np.mean(get_dx_left())
        dx_right = np.mean(get_dx_right())

        # Now check if we breached any of the thresholds
        ret = ''
        # Instantaneous refers to the current rate of change
        if dx_left > left_tollerance_instantaneous:
            if __args.debug_mode > -1:
                print('Instantaneous check failed for left lane on frame {};'.format(pipeline.counter), dx_left)
            ret += 'LEFT'

        elif dx_right > right_tollerance_instantaneous:
            if __args.debug_mode > -1:
                print('Instantaneous check failed for right lane on frame {};'.format(pipeline.counter), dx_right)
            ret += 'RIGHT'
        if ret != '': 
            return ret

        dx_left_last = find_lane_lines.previous_dx_left
        dx_right_last = find_lane_lines.previous_dx_right

        # Past refers to the comparison of the rate of change on this image vs the last one
        if abs(dx_left - dx_left_last) > left_tollerance_past:
            if __args.debug_mode > -1:
                print('Past check failed for left lane on frame {};'.format(pipeline.counter), 
                        abs(dx_left - dx_left_last), abs(dx_right - dx_right_last))
            ret += 'LEFT'

        elif abs(dx_right - dx_right_last) > right_tollerance_past:
            if __args.debug_mode > -1:
                print('Past check failed for right lane on frame {};'.format(pipeline.counter), 
                        abs(dx_left - dx_left_last), abs(dx_right - dx_right_last))
            ret += 'RIGHT'
        if ret != '': 
            return ret

        # If we got this far, store the results for the next iteration
        find_lane_lines.previous_dx_left = dx_left_last 
        find_lane_lines.previous_dx_right = dx_right_last 

        return 'PASS'


    def store_results():
        # Helper function to cache the most recent calculations
        find_lane_lines.previous = {}
        find_lane_lines.previous['line_search_img'] = line_search_img
        find_lane_lines.previous['leftx'] = leftx
        find_lane_lines.previous['lefty'] = lefty
        find_lane_lines.previous['rightx'] = rightx
        find_lane_lines.previous['righty'] = righty
        find_lane_lines.previous['poly_left'] = poly_left
        find_lane_lines.previous['poly_right'] = poly_right


    def get_avg_polys():
        # Get two polynomials taking into account the history of pevious images

        history_length = 10

        # Remove the earliest stored sample if needed and append this new one instead
        if 'poly_left' in pipeline.__dict__:
            if len(pipeline.poly_left) == history_length:
                pipeline.poly_left.pop(0)
                pipeline.poly_right.pop(0)
        else:
            pipeline.poly_left = []
            pipeline.poly_right = []

        poly_left = np.polyfit(lefty, leftx, 2)
        poly_right = np.polyfit(righty, rightx, 2)

        pipeline.poly_left.append(poly_left)
        pipeline.poly_right.append(poly_right)

        poly_left = np.mean(pipeline.poly_left, axis=0)
        poly_right = np.mean(pipeline.poly_right, axis=0)

        return poly_left, poly_right


    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

    # If is the first image in the video, or the sanity check on the basic 
    # parse fails, process the entire image
    if 'previous' not in find_lane_lines.__dict__:
        line_search_img, leftx, lefty, rightx, righty = get_polys_full(binary_warped)
        poly_left, poly_right = get_avg_polys()
        store_results()


    # If we have past data, just scan around the margins of the new image 
    # for lane lines instead of processing the entire image. There doesn't seem
    # to be much of a speedup in computation with this, but the stability of 
    # the lane detection improves.
    else:
        leftx, lefty, rightx, righty = get_polys_margin(
             binary_warped, 
             find_lane_lines.previous['poly_left'],
             find_lane_lines.previous['poly_right']
        )
        poly_left, poly_right = get_avg_polys()
        line_search_img = find_lane_lines.previous['line_search_img']

        # Now run the sanity check and figure out what to do
        if sanity_check() == 'PASS':
            store_results()
        else:
            line_search_img, leftx, lefty, rightx, righty = get_polys_full(binary_warped)
            poly_left, poly_right = get_avg_polys()

            check = sanity_check()
            if check != 'PASS':
                # Just take the lane lines from the last image
                line_search_img = find_lane_lines.previous['line_search_img']
                if check in ['LEFT', 'LEFTRIGHT']:
                    if __args.debug_mode > -1:
                        print('Using pervious left lines instead.')
                    leftx = find_lane_lines.previous['leftx']
                    lefty = find_lane_lines.previous['lefty']
                    poly_left = find_lane_lines.previous['poly_left']
                if check in ['RIGHT', 'LEFTRIGHT']:
                    if __args.debug_mode > -1:
                        print('Using pervious right lines instead.')
                    rightx = find_lane_lines.previous['rightx']
                    righty = find_lane_lines.previous['righty']
                    poly_right = find_lane_lines.previous['poly_right']
            else:
                if __args.debug_mode > -1:
                    print('Full scan worked!')
                store_results()

    # Create lines for the output image showing the lanes
    left_fitx = poly_left[0]*ploty**2 + poly_left[1]*ploty + poly_left[2]
    right_fitx = poly_right[0]*ploty**2 + poly_right[1]*ploty + poly_right[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    ###########################################################################################
    #
    # Calculate real world measurements
    #
    #
    # Find the radius of curvature in real world dimensions. We are using a hard
    # coded dimension conversion here for simplicity (`ym_per_pix` and `xm_per_pix`)
    # but in a real system it would be necessary to use some kind of markers in the
    # image that are of known dimensions. This is needed for example to handle cases
    # where the road is ascending or descending a hill.
    # 
    # Define conversions in x and y from pixels space to meters. These are assumed
    # values based on US regulations for lane sizes & road markings, and assume that
    # all images are of flat terrain (not necessarily the case, but good enough for
    # this project).
    #
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/750 # meters per pixel in x dimension

    # Now we can get the distance from center in meters
    true_center = binary_warped.shape[1] / 2
    cur_center = rightx[0] - leftx[0] 
    dist_from_center_px = true_center - cur_center
    dist_from_center = dist_from_center_px * xm_per_pix

    poly_left_cr = np.polyfit( lefty * ym_per_pix, leftx * xm_per_pix, 2 )
    poly_right_cr = np.polyfit( righty * ym_per_pix, rightx * xm_per_pix, 2 )

    # Calculate the new radii of curvature
    y_eval = np.max(lefty)

    # Calculate radius of curvature in meters
    left_curve_rad = (((1 + (2*poly_left_cr[0]*y_eval*ym_per_pix + \
            poly_left_cr[1])**2)**1.5) / np.absolute(2*poly_left_cr[0]))
    right_curve_rad = (((1 + (2*poly_right_cr[0]*y_eval*ym_per_pix + \
            poly_right_cr[1])**2)**1.5) / np.absolute(2*poly_right_cr[0]))
    #
    #
    ###########################################################################################

    return color_warp, line_search_img, left_fitx, right_fitx, left_curve_rad, right_curve_rad, dist_from_center


def get_polys_full(binary_warped, margin_l=100, margin_r=110, minpix=20, nwindows=9):
    """Scan the input image using a sliding window approach to find left and right
    lane pixel positions.

    Args:
        binary_warped: birdseye view of the road
        margin_l: the margin in pixels to seach around for the left lane
        margin_r: the margin in pixels to seach around for the right lane
        minpix: minimum number of pixels to detect for each window
        nwindows: number of sliding windows to use

    Returns:
        tuple containing an image of the search process,
            left and right x & y pixels,
            and left polynomial, right polynomial

    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Capture the lane search process in a dedicatd image
    line_search_img = np.stack((binary_warped, binary_warped, binary_warped), axis=2)*255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin_l
        win_xleft_high = leftx_current + margin_l

        win_xright_low = rightx_current - margin_r
        win_xright_high = rightx_current + margin_r

        cv2.rectangle(line_search_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
        cv2.rectangle(line_search_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return line_search_img, leftx, lefty, rightx, righty


def get_polys_margin(binary_warped, left_fit, right_fit, margin=100):
    """Get polynomials within a margin of an image given a pre-exiting polynomial result.

    Args:
        binary_warped: the binary warped image
        left_fit: the pre-existing left fit
        right_fit: the pre-existing right fit
        margin: margin to search within
    Returns:
        tuple containing an image of the search, left and right x & y pixels, left polynomial, right polynomial

    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


if __name__ == '__main__':
    main()



