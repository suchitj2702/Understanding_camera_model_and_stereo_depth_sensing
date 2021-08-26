import sys
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt

TestImagesDirectory = "../../images/task_3_and_4"

def depthTriangulation():
    CamMatrix_left = np.load('../../parameters/leftCameraIntrinsics/IntrinsicMatrix.npy')
    DistortionCoeff_left = np.load('../../parameters/leftCameraIntrinsics/DistortCoeff.npy')

    CamMatrix_right = np.load('../../parameters/rightCameraIntrinsics/IntrinsicMatrix.npy')
    DistortionCoeff_right = np.load('../../parameters/rightCameraIntrinsics/DistortCoeff.npy')

    RotationMatrix = np.load('../../parameters/SterioCalibration/RotationMatrix.npy')
    TranslationVector = np.load('../../parameters/SterioCalibration/TranslationVector.npy')
    EssentialMatrix = np.load('../../parameters/SterioCalibration/EssentialMatrix.npy')
    FundamentalMatrix = np.load('../../parameters/SterioCalibration/FundamentalMatrix.npy')

    RectifiedRotationMatrix_left = np.load('../../parameters/SterioRectification/RectifiedRotationMatrix_left.npy')
    RectifiedRotationMatrix_right = np.load('../../parameters/SterioRectification/RectifiedRotationMatrix_right.npy')
    RectifiedProjection_left = np.load('../../parameters/SterioRectification/RectifiedProjection_left.npy')
    RectifiedProjection_right = np.load('../../parameters/SterioRectification/RectifiedProjection_right.npy')
    depthMappingMatrix = np.load('../../parameters/SterioRectification/depthMappingMatrix.npy')

    # using the parameters used by Duo
    blockSize = 7
    numDisparities = 64
    speckleWindowSize = 10
    
    print("Generating disparity map...")
    matcher = cv2.StereoSGBM_create(minDisparity = 0, numDisparities = numDisparities, blockSize = blockSize, preFilterCap = 63, uniquenessRatio = 15, speckleWindowSize = speckleWindowSize, speckleRange = 1, disp12MaxDiff = 20, P1 = 8*3*blockSize**2, P2 = 32*3*blockSize**2, mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    matcher_left = matcher
    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)

    disparity_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
    disparity_filter.setLambda(70000)
    disparity_filter.setSigmaColor(1.2)

    object_indx = [2, 5, 8, 9]
    for object_number in object_indx:
        gray_left_undistorted_rectified, gray_right_undistorted_rectified = loadUndistortedRectifiedLeftRightImages(str(object_number), CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right, RectifiedRotationMatrix_left, RectifiedRotationMatrix_right, RectifiedProjection_left, RectifiedProjection_right)

        disparity_left = matcher_left.compute(gray_left_undistorted_rectified, gray_right_undistorted_rectified)
        disparity_right = matcher_right.compute(gray_right_undistorted_rectified, gray_left_undistorted_rectified)

        disparity_left = np.int16(disparity_left)
        disparity_right = np.int16(disparity_right)

        disparity_normalized = cv2.normalize(disparity_left, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

        disparity_filtered = disparity_filter.filter(disparity_left, gray_left_undistorted_rectified, None, disparity_right)

        disparity_normalized_filtered = cv2.normalize(disparity_filtered, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

        cv2.imwrite('../../output/task_4/object_'+ str(object_number) +'_disparity.png', disparity_normalized)
        # cv2.imshow('disparity_left', disparity_normalized)
        # cv2.waitKey(0)

        cv2.imwrite('../../output/task_4/object_'+ str(object_number) +'_filtered_disparity.png', disparity_normalized_filtered)
        # cv2.imshow('disparity_filtered', disparity_normalized_filtered)
        # cv2.waitKey(0)

    print("Done")


def loadUndistortedRectifiedLeftRightImages(indx, CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right, RectifiedRotationMatrix_left, RectifiedRotationMatrix_right, Projection_left, Projection_right):
    start_left = 'left_'
    start_right = 'right_'
    end = '.png'
    for filename in os.listdir(TestImagesDirectory):
        if filename[filename.find(start_left)+len(start_left):filename.rfind(end)] == indx:
            image_filename = os.path.join(TestImagesDirectory, filename)
            img_left = cv2.imread(image_filename)
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            
            h, w = gray_left.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix_left, DistortionCoeff_left, RectifiedRotationMatrix_left, Projection_left, (w, h), cv2.CV_16SC2)
            # mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix_left, DistortionCoeff_left, None, CamMatrix_left, (w, h), cv2.CV_16SC2)
            gray_left = cv2.remap(gray_left, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)

            # x, y, w, h = roi_left
            # gray_left = gray_left[y: y + h, x: x + w]
        
        if filename[filename.find(start_right)+len(start_right):filename.rfind(end)] == indx:
            image_filename = os.path.join(TestImagesDirectory, filename)
            img_right = cv2.imread(image_filename)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            h, w = gray_right.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix_right, DistortionCoeff_right, RectifiedRotationMatrix_right, Projection_right, (w, h), cv2.CV_16SC2)
            # mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix_right, DistortionCoeff_right, None, CamMatrix_right, (w, h), cv2.CV_16SC2)
            gray_right = cv2.remap(gray_right, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)

            # x, y, w, h = roi_right
            # gray_left = gray_right[y: y + h, x: x + w]

    return gray_left, gray_right

def plotUndistortedAndRectified(camera, CamMatrix, DistortionCoeff, RectifiedRotationMatrix, RectifiedCoordinate):
    for filename in os.listdir(UndistortedAndRectifiedDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(UndistortedAndRectifiedDirectory, filename)
            
            TestImg = cv2.imread(image_filename)
            h, w = TestImg.shape[:2]

            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix, DistortionCoeff, RectifiedRotationMatrix, RectifiedCoordinate, (w, h), cv2.CV_32FC1)
            UndistortedAndRectifiedImg = cv2.remap(TestImg, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)

            RectifiedRotationMatrix_left, RectifiedRotationMatrix_right, RectifiedCoordinate_left, RectifiedCoordinate_right, Q, roi_left, roi_right = cv2.stereoRectify(CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right, (480, 640), RotationMatrix, TranslationVector, alpha = 0.25, flags = cv2.CALIB_FIX_INTRINSIC)
            # x, y, w, h = roi
            # UndistortedAndRectifiedImg = UndistortedAndRectifiedImg[y: y + h, x: x + w]

            cv2.imwrite(os.path.join(UndistortedAndRectifiedDirectory, filename), UndistortedAndRectifiedImg)

if __name__ == '__main__':
    depthTriangulation()