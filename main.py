import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from lane_data import Line
from camera_data import CameraSensor
import image_processing


camera = CameraSensor()
camera.load_calibration()

def lane_pipeline(img, ad_line, show_lane=False, show_curve_radius=False, show_mask_roi=False):
    # find the edges in the image for line detection
    thr_lanes = ad_line.find_driving_lanes(img)
    warped = ad_line.warp_image(thr_lanes)
    # Warp the image using OpenCV warpPerspective()
    # get the polynomials of the lane lines
    ad_line.left_fit, ad_line.right_fit = ad_line.lane_detection(warped, draw_rectangle=False, draw_poly=False)

    if show_lane:
        # draw the overlay onto the image
        img = ad_line.draw_lane(img)
    
    if show_curve_radius:
        c_curv = ad_line.measure_curvature_real(669)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'Curve Radius : {:.0f}m'.format(c_curv)
        cv2.putText(img, text, (850,90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if show_mask_roi:
        img = ad_line.draw_lane_roi(img)
    return img


# Read in an image


def image_test():
    img = cv2.imread('test_images/fail_image.jpg')
    img = img[:-image_processing.offset_y,:]
    h, w = img.shape[:2]
    #The focal point is
    focal_point = [w//2, 400]

    ad_line = Line([h,w], 40, focal_point,[[0,h],[w,h]])
    ad_line.get_roi()

    undist = camera.undistort_image(img)

    image_processed = lane_pipeline(undist, ad_line)

    cv2.imshow("image", image_processed)
    cv2.waitKey(0)

def video_test():
    cap = cv2.VideoCapture("project_video.mp4")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if ret != True:
            return

        best_frame=frame[:-image_processing.offset_y,:]
        try:
            ad_line
        except:
            #instatiates lane
            h, w = best_frame.shape[:2]
            #focal_point = [w//2, 410]
            focal_point = [w//2, 410]
    
            ad_line = Line([h,w], 45, focal_point,[[0,h],[w,h]])
            ad_line.get_roi()
    
        undist = camera.undistort_image(best_frame)
        #cv2.imwrite('output_images/undist.jpg', undist)
        image_processed = lane_pipeline(undist, ad_line, True, True,True)
        frame[:-image_processing.offset_y,:] = image_processed[:,:]
        cv2.imshow("image", frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


#image_test()
video_test()