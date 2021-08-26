import sys
import numpy as np
from numpy import transpose
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

    object_indx = [8]
    for object_number in object_indx:

        print("Undistorting Image...")
        gray_left_undistorted, gray_right_undistorted = loadUndistortedLeftRightImages(str(object_number), CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right)

        print("Finding key points...")
        orb = cv2.ORB_create(nfeatures = 1000)
        key_points_left, descriptors_left = orb.detectAndCompute(gray_left_undistorted, None)
        key_points_right, descriptors_right = orb.detectAndCompute(gray_right_undistorted, None)

        optimal_key_points_left, optimal_descriptors_left = nms(key_points_left, descriptors_left, 5)
        optimal_key_points_right, optimal_descriptors_right = nms(key_points_right, descriptors_right, 5)

        gray_left_undistorted_raw_keypoints = cv2.drawKeypoints(gray_left_undistorted, key_points_left, None, color = (255, 0, 0), flags = 0)
        cv2.imwrite('../../output/task_3/object_'+ str(object_number) +'_left_key_points.png', gray_left_undistorted_raw_keypoints)

        gray_left_undistorted_optimal_keypoints = cv2.drawKeypoints(gray_left_undistorted, optimal_key_points_left, None, color = (255, 0, 0), flags = 0)
        cv2.imwrite('../../output/task_3/object_'+ str(object_number) +'_left_key_points_with_nms.png', gray_left_undistorted_optimal_keypoints)
        
        gray_right_undistorted_raw_keypoints = cv2.drawKeypoints(gray_right_undistorted, key_points_right, None, color = (255, 0, 0), flags = 0)
        cv2.imwrite('../../output/task_3/object_'+ str(object_number) +'_right_key_points.png', gray_right_undistorted_raw_keypoints)

        gray_right_undistorted_optimal_keypoints = cv2.drawKeypoints(gray_right_undistorted, optimal_key_points_right, None, color = (255, 0, 0), flags = 0)
        cv2.imwrite('../../output/task_3/object_'+ str(object_number) +'_right_key_points_with_nms.png', gray_right_undistorted_optimal_keypoints)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        print("Matching keypoints...")
        matches = bf.match(np.asarray(optimal_descriptors_left), np.asarray(optimal_descriptors_right))
        matches = sorted(matches, key = lambda x: x.distance)

        # inspired by https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html and Duo's implementation for matching filtering

        threshold = matches[0].distance*20

        filtered_matches = []
        optimal_key_points_filtered_left = []
        optimal_key_points_filtered_right = []

        for i, match in enumerate(matches):
            if match.distance > threshold:
                break

            optimal_key_points_left_homogeneous = np.ones((3, 1))
            optimal_key_points_left_homogeneous[0:2, :] = np.asarray(optimal_key_points_left[match.queryIdx].pt).reshape((2, 1))
            optimal_key_points_right_homogeneous = np.ones((3, 1))
            optimal_key_points_right_homogeneous[0:2, :] = np.asarray(optimal_key_points_right[match.trainIdx].pt).reshape((2, 1))

            epipolar_constraint = np.matmul(np.matmul(optimal_key_points_left_homogeneous.T, FundamentalMatrix), optimal_key_points_right_homogeneous)

            if math.fabs(epipolar_constraint) < 0.5:
                filtered_matches.append(match)
                optimal_key_points_filtered_left.append(optimal_key_points_left[match.queryIdx].pt)
                optimal_key_points_filtered_right.append(optimal_key_points_right[match.trainIdx].pt)

        image_matches = cv2.drawMatches(gray_left_undistorted, optimal_key_points_left, gray_right_undistorted, optimal_key_points_right, filtered_matches, None, flags = 2)

        checkSparseDepth(transpose(optimal_key_points_filtered_left), transpose(optimal_key_points_filtered_right), RotationMatrix, TranslationVector)

        cv2.imwrite('../../output/task_3/object_'+ str(object_number) +'_filtered_matches.png', image_matches)

        print("Done")

def loadUndistortedLeftRightImages(indx, CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right):
    start_left = 'left_'
    start_right = 'right_'
    end = '.png'
    for filename in os.listdir(TestImagesDirectory):
        if filename[filename.find(start_left)+len(start_left):filename.rfind(end)] == indx:
            image_filename = os.path.join(TestImagesDirectory, filename)
            img_left = cv2.imread(image_filename)
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            
            h, w = gray_left.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix_left, DistortionCoeff_left, None, CamMatrix_left, (w, h), cv2.CV_16SC2)
            gray_left = cv2.remap(gray_left, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)
        
        if filename[filename.find(start_right)+len(start_right):filename.rfind(end)] == indx:
            image_filename = os.path.join(TestImagesDirectory, filename)
            img_right = cv2.imread(image_filename)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            h, w = gray_right.shape[:2]
            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix_right, DistortionCoeff_right, None, CamMatrix_right, (w, h), cv2.CV_16SC2)
            gray_right = cv2.remap(gray_right, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)

    return gray_left, gray_right
    
# inspired by Duo's implementation of nms
def nms(key_points, descriptors, radius):

    optimal_key_points = []
    optimal_descriptors = []

    for i in range(len(key_points)):
        maxima = True

        for j in range(len(key_points)):
            if i == j:
                continue

            if math.dist(key_points[i].pt, key_points[j].pt) <= radius and key_points[j].response > key_points[i].response:
                maxima = False
                break
    
        if maxima:
            optimal_key_points.append(key_points[i])
            optimal_descriptors.append(descriptors[i])

    return optimal_key_points, optimal_descriptors

def checkSparseDepth(key_points_left, key_points_right, RotationMatrix, TranslationVector):
    R1 = np.eye(3, 3)
    t1 = np.zeros((3, 1))
    P1 = np.concatenate((R1, t1), axis = 1)

    P2 = np.concatenate((RotationMatrix, TranslationVector), axis = 1)
    
    points3DH = cv2.triangulatePoints(projMatr1 = P1, projMatr2 = P2, projPoints1 = key_points_left, projPoints2 = key_points_right)

    x = points3DH[0]/points3DH[3]
    y = points3DH[1]/points3DH[3]
    z = points3DH[2]/points3DH[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('../../output/task_3/keyPoints3D.png')
    plt.show()

if __name__ == '__main__':
    depthTriangulation()
