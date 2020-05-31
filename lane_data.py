import numpy as np
import cv2
import image_processing
import inspect

from collections import deque

def printDebug(msg):
  callerframerecord = inspect.stack()[1]    # 0 represents this line
                                            # 1 represents line at caller
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  print(info.function, info.lineno,": ",msg)

class Line():
    def __init__(self, base_points, roi_y_offset=0, focal_point=None, 
        source_pts=None, lane_width=3.7, lane_length=24, filter_size=24):

        # was the line detected in the last iteration?
        self.left_detected = False
        self.right_detected = False
        #average x values of the fitted line over the last n iterations
        self.bestx_left = None     
        self.bestx_right = None     
        #polynomial coefficients averaged over the last n iterations
        self.left_fit = None
        self.right_fit = None

        #distance in pixel between the line
        self.ym_per_pix = lane_length /670
        self.xm_per_pix = lane_width/700


        self.filter_size = filter_size
        self.left_fit_filter = deque(maxlen=filter_size)
        self.right_fit_filter = deque(maxlen=filter_size)
        #radius of curvature of the line in meters
        self.radius_of_curvature = deque(maxlen=filter_size)

        if focal_point is None:
            self.focal_point = [base_points[1]//2,base_points[0]//2]
        else:
            self.focal_point = focal_point

        self.roi_y_offset = roi_y_offset

        self.roi_pts = None

        self.h = base_points[0] #image perpective transformation
        self.w = base_points[1] #image perpective transformation

        if source_pts is None:
            self.source_pts = [[0, 0], [0, 0], [0, 0], [0, 0]]
        else:
            self.source_pts = source_pts

    def draw_lane_roi(self, img, roi_pts=None, color=(0, 0, 255)):

        if roi_pts is None:
            roi_pts = self.roi_pts

        image = img.copy()
        pts = np.int32(roi_pts)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, 2)

        return image

    def get_roi(self, roi_y_offset=None, focal_point=None, source_pts=None):
        # defines a lanes region of interest

        if focal_point is None:
            print('focal point')
            focal_point = self.focal_point

        if roi_y_offset is None:
            roi_y_offset = self.roi_y_offset

        if source_pts is None:
            source_pts = self.source_pts

        # height of focal point
        h_top = roi_y_offset + focal_point[1]

        #Get the line equation, then calculate a x to a given y

        m_left = (focal_point[1] - source_pts[0][1]) / (focal_point[0] - source_pts[0][0])
        b_left = focal_point[1] - (m_left * focal_point[0])
        x_left = (h_top - b_left) // m_left

        m_right = (focal_point[1] - source_pts[1][1]) / (focal_point[0] - source_pts[1][0])
        b_right = focal_point[1] - (m_right * focal_point[0])
        x_right = (h_top - b_right) // m_right

        self.roi_pts = np.float32([source_pts[0], [x_left, h_top], [x_right, h_top], source_pts[1]])
        return self.roi_pts

    # returns : the warped perspective image with the supplied points
    def warp_image(self, img, roi_pts=None, location_pts=None, padding=(0,0)):
        
        if roi_pts is None:
            roi_pts = self.roi_pts

        if location_pts is None:
            location_pts = np.float32([[padding[0], self.h-padding[1]], # bot-left
                                       [padding[0], padding[1]], # top-left
                                       [self.w-padding[0], padding[1]], # top-right
                                       [self.w-padding[0], self.h-padding[1]]]) # bot-right

        # calculate the perspective transform matrix between the old and new points
        self.M = cv2.getPerspectiveTransform(roi_pts, location_pts)
        # Warp the image to the new perspective
        return cv2.warpPerspective(img, self.M, (self.w, self.h))

    def find_driving_lanes(self, image):
        # Convert to HSV color space and separate the V channel
        # hls for Sobel edge detection
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        b_channel = image[:,:,0]
        g_channel = image[:,:,1]
        r_channel = image[:,:,2]
        s_channel = hls[:,:,2]

        _, binary_b = cv2.threshold(b_channel, 100, 255, cv2.THRESH_BINARY)
        _, binary_g = cv2.threshold(g_channel, 170, 255, cv2.THRESH_BINARY)
        _, binary_r = cv2.threshold(r_channel, 110, 255, cv2.THRESH_BINARY)

        binary_y = np.bitwise_or(binary_b, binary_g)
        binary_y = image_processing.blur(binary_y)

        _, binary_s = cv2.threshold(s_channel, 120, 255, cv2.THRESH_BINARY)

        binary_rg = np.bitwise_and(binary_r, binary_g)

        blur_img = image_processing.blur(binary_rg, ksize=3)
        magbinary = image_processing.mag_thresh(blur_img, sobel_kernel=3, thresh=(110, 255))

        rs_binary = cv2.bitwise_and(binary_s, binary_r)

        out = cv2.bitwise_or(rs_binary, magbinary.astype(np.uint8))
        out_bir = np.zeros_like(b_channel)
        out_bir[out > 0] = 255

        return out_bir

    def lane_detection(self, image, draw_rectangle=False, draw_poly=False):
        # finds the location of the lanes lines
        #img = image.copy()
        img = np.zeros_like(image)
        img[image>1] = 1
        if self.left_fit is None or self.right_fit is None or self.left_detected == False or self.right_detected == False:
            self.find_lane_pixels(img, draw_rectangle, draw_poly=draw_poly)
        else:
            self.search_around_poly(img, draw_poly=draw_poly)

        return self.left_fit, self.right_fit

    def update_fit_coef(self, image):
            ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        self.ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        try:
            self.bestx_left = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            self.bestx_right = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.bestx_left = 1*self.ploty**2 + 1*self.ploty
            self.bestx_right = 1*self.ploty**2 + 1*self.ploty

    def find_lane_pixels(self, binary_warped, draw_rectangle=False, draw_poly=False):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = binary_warped.copy()

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            if draw_rectangle:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.left_detected = True
        self.right_detected = True

        # Fit a second order polynomial to each
        try:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        except:
            self.left_fit = [0,0,0]
            self.left_detected = False
        try:
            self.right_fit = np.polyfit(righty, rightx, 2)
        except:
            self.right_fit = [0,0,0]
            self.right_detected = False

        self.left_fit = self.moving_average(self.left_fit_filter, self.left_fit)
        self.right_fit = self.moving_average(self.right_fit_filter, self.right_fit)

        self.update_fit_coef(binary_warped)
        if draw_poly==True:
            self.plot_fit(binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)


        return self.left_fit, self.right_fit



    def search_around_poly(self, binary_warped, margin=100, draw_poly=False):
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ###Set the area of search based on activated x-values ###
        left_lane_inds = (
            (nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & 
            (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)) )

        right_lane_inds = (
            (nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & 
            (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)) )
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        
        self.left_detected = True
        self.right_detected = True

        # Fit a second order polynomial to each
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
        except:
            printDebug("Error to fit left line")
            left_fit = self.left_fit
            self.left_detected = False
        try:
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            printDebug("Error to fit right line")
            right_fit = self.right_fit
            self.right_detected = False

        if(abs(left_fit[0] - self.left_fit[0]) < 0.5):
            self.left_fit = left_fit

        if(abs(right_fit[0] - self.right_fit[0]) < 0.5):
            self.right_fit = right_fit

        self.left_fit = self.moving_average(self.left_fit_filter, self.left_fit)
        self.right_fit = self.moving_average(self.right_fit_filter, self.right_fit)
        
        self.update_fit_coef(binary_warped)
        
        # Draw the lane onto the warped blank image
        if draw_poly==True:
            self.plot_fit(binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

        return self.left_fit, self.right_fit

    def plot_fit(self, binary_warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, margin=100):
        ## Visualization ##

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([self.bestx_left-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx_left+margin, 
                                self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
        right_line_window1 = np.array([np.transpose(np.vstack([self.bestx_right-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx_right+margin, 
                                self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        self.bestx_right[self.bestx_right >= binary_warped.shape[1]] = binary_warped.shape[1]
        self.bestx_left[self.bestx_left >= binary_warped.shape[1]] = binary_warped.shape[1]

        result[np.int32(self.ploty),np.int32(self.bestx_left)] = [0,255,255]
        result[np.int32(self.ploty),np.int32(self.bestx_right)] = [0,255,255]

        cv2.imshow("find_lanes", result)
        return

    def draw_lane(self, img, color=(50,255,50), overlay_weight=0.3):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.bestx_left, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.bestx_right, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP)

        # Combine the result with the original image        
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    def moving_average(self, filter_q, data):
        # Calculates the moving average of an array
        filter_q.appendleft(data)
        queue_length = len(filter_q)
        try:
            # find the moving average
            average = sum(filter_q) / queue_length
        except:
            average = -1

        if queue_length >= self.filter_size:
            filter_q.pop()
        return average


    def measure_curvature_real(self, y_eval):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        #self.ym_per_pix = 30/720 # meters per pixel in y dimension
        #self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!

        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        #y_eval = np.max(self.ploty)
        # Calculation of R_curve (radius of curvature)
        cent_fit = (self.left_fit + self.right_fit) / 2
        rad = ((1 + (2 * cent_fit[0] * y_eval * self.ym_per_pix + cent_fit[1]) ** 2) ** 1.5) / np.absolute(2 * cent_fit[0])

        rad = self.moving_average(self.radius_of_curvature, rad)

        return rad/2