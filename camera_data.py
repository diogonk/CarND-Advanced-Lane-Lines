import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class CameraSensor():

    def __init__(self, cal_path='camera_cal/calibration*.jpg', pickle_path='camera_cal/wide_dist_pickle.p'):
        self.cal_path = cal_path
        self.pickle_path = pickle_path
        self.mtx = None
        self.newmtx = None
        self.dist = None
        self.corners = None
        self.roi = None

    def perform_calibration(self):

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.cal_path)

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(100)
        cv2.destroyAllWindows()

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)

        # Test undistortion on an image
        img = cv2.imread('camera_cal/calibration1.jpg')

        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        cv2.imwrite('camera_cal/test_undist.jpg',dst)


        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump( dist_pickle, open("self.pickle_path", "wb" ) )

        #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.show()
        return [self.mtx, self.dist]

    def load_calibration(self, pickle_path=None):
        if pickle_path == None:
            pickle_path = self.pickle_path
        # Read in the saved camera matrix and distortion coefficients
        # These are the arrays you calculated using cv2.calibrateCamera()
        dist_pickle = pickle.load(open( pickle_path, "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

        return [self.mtx, self.dist]

    def undistort_image(self, img, mtx=None, dist=None):
        #returns the undistor image
        if mtx == None: mtx = self.mtx
        if dist == None: dist = self.dist

        return cv2.undistort(img, mtx, dist, None, mtx)