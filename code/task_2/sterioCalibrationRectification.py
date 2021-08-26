import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import transformations as t

CalibrateImagesDirectory = "../../images/task_1"
TestImagesDirectory = "../../images/task_2"
UndistortedNotRectifiedDirectory = "../../output/task_2/UndistortedNotRectified"
UndistortedAndRectifiedDirectory = "../../output/task_2/UndistortedAndRectified"
CornerPointsDirectory = "../../output/task_2/CornerPointsMarked"

def extract3Dto2Dcorrespondence(camera):    

    n = 9
    m = 6

    Points3D = np.zeros((n*m, 3), np.float32)
    Points3D[:,:2] = np.mgrid[0: n, 0: m].T.reshape(-1, 2)

    PointsArray3D = []
    ImageArray2D = []

    for filename in os.listdir(CalibrateImagesDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(CalibrateImagesDirectory, filename)

            img = cv2.imread(image_filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (n, m), None)
            
            if ret == True:
                PointsArray3D.append(Points3D)
                ImageArray2D.append(corners)

    CamMatrix = np.load('../../parameters/' + camera + 'CameraIntrinsics' + '/IntrinsicMatrix.npy')
    DistortionCoeff = np.load('../../parameters/' + camera + 'CameraIntrinsics' + '/DistortCoeff.npy')

    for filename in os.listdir(TestImagesDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(TestImagesDirectory, filename)

            img = cv2.imread(image_filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (n, m), None)
            
            if ret == True:
                cv2.drawChessboardCorners(img, (n, m), corners, ret)
                cv2.imwrite(os.path.join(CornerPointsDirectory, filename), img)
                cv2.imwrite(os.path.join(UndistortedNotRectifiedDirectory, filename), img)
                cv2.imwrite(os.path.join(UndistortedAndRectifiedDirectory, filename), img)

    return PointsArray3D, ImageArray2D, CamMatrix, DistortionCoeff

def sterioCalibration():
    print("Calibrating cameras...")
    PointsArray3D_left, ImageArray2D_left, CamMatrix_left, DistortionCoeff_left= extract3Dto2Dcorrespondence("left")
    PointsArray3D_right, ImageArray2D_right, CamMatrix_right, DistortionCoeff_right = extract3Dto2Dcorrespondence("right")
    
    ret, CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right, RotationMatrix, TranslationVector, EssentialMatrix, FundamentalMatrix = cv2.stereoCalibrate(PointsArray3D_left, ImageArray2D_left, ImageArray2D_right, CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right, (640, 480), flags = cv2.CALIB_FIX_INTRINSIC)

    undistortedArray_left = undistort(ImageArray2D_left, CamMatrix_left, DistortionCoeff_left)
    undistortedArray_right = undistort(ImageArray2D_right, CamMatrix_right, DistortionCoeff_right)

    checkCalibration(undistortedArray_left, undistortedArray_right, RotationMatrix, TranslationVector)

    np.save('../../parameters/SterioCalibration/RotationMatrix', RotationMatrix)
    np.save('../../parameters/SterioCalibration/TranslationVector', TranslationVector)
    np.save('../../parameters/SterioCalibration/EssentialMatrix', EssentialMatrix)
    np.save('../../parameters/SterioCalibration/FundamentalMatrix', FundamentalMatrix)

    print("Rectifying...")
    RectifiedRotationMatrix_left, RectifiedRotationMatrix_right, RectifiedProjection_left, RectifiedProjection_right, depthMappingMatrix, roi_left, roi_right = cv2.stereoRectify(CamMatrix_left, DistortionCoeff_left, CamMatrix_right, DistortionCoeff_right, (640, 480), RotationMatrix, TranslationVector, alpha = 0.25, flags = cv2.CALIB_FIX_INTRINSIC)

    np.save('../../parameters/SterioRectification/RectifiedRotationMatrix_left', RectifiedRotationMatrix_left)
    np.save('../../parameters/SterioRectification/RectifiedRotationMatrix_right', RectifiedRotationMatrix_right)
    np.save('../../parameters/SterioRectification/RectifiedProjection_left', RectifiedProjection_left)
    np.save('../../parameters/SterioRectification/RectifiedProjection_right', RectifiedProjection_right)
    np.save('../../parameters/SterioRectification/depthMappingMatrix', depthMappingMatrix)
    
    plotUndistortedNotRectified("left", CamMatrix_left, DistortionCoeff_left)
    plotUndistortedNotRectified("right", CamMatrix_right, DistortionCoeff_right)

    plotUndistortedAndRectified("left", CamMatrix_left, DistortionCoeff_left, RectifiedRotationMatrix_left, RectifiedProjection_left, roi_left)
    plotUndistortedAndRectified("right", CamMatrix_right, DistortionCoeff_right, RectifiedRotationMatrix_right, RectifiedProjection_right, roi_right)

print("Done")

def undistort(ImageArray2D, CamMatrix, DistortionCoeff):
    undistortedArray = []
    for ImageArray in ImageArray2D:
        undistortedArray.append(cv2.undistortPoints(ImageArray, CamMatrix, DistortionCoeff))
    
    for i in range(len(undistortedArray)):
        undistortedArray[i] = undistortedArray[i].transpose().reshape(2, 54)

    return undistortedArray

def checkCalibration(undistortedArray_left, undistortedArray_right, RotationMatrix, TranslationVector):
    R1 = np.eye(3, 3)
    t1 = np.zeros((3, 1))
    P1 = np.concatenate((R1, t1), axis = 1)

    P2 = np.concatenate((RotationMatrix, TranslationVector), axis = 1)
    
    points3DH = cv2.triangulatePoints(projMatr1 = P1, projMatr2 = P2, projPoints1 = undistortedArray_left[0], projPoints2 = undistortedArray_right[0])

    x = points3DH[0]/points3DH[3]
    y = points3DH[1]/points3DH[3]
    z = points3DH[2]/points3DH[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    plot_camera(0, 0, 0, ax)
    
    plt.savefig('../../output/task_2/boardPoints3D.png')

def plotUndistortedNotRectified(camera, CamMatrix, DistortionCoeff):
    for filename in os.listdir(UndistortedNotRectifiedDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(UndistortedNotRectifiedDirectory, filename)
            
            TestImg = cv2.imread(image_filename)
            h, w = TestImg.shape[:2]
            
            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix, DistortionCoeff, np.eye(3, 3), CamMatrix, (w, h), cv2.CV_16SC2)
            UndistortedImg = cv2.remap(TestImg, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)

            cv2.imwrite(os.path.join(UndistortedNotRectifiedDirectory, filename), UndistortedImg)

def plotUndistortedAndRectified(camera, CamMatrix, DistortionCoeff, RectifiedRotationMatrix, RectifiedProjection, roi):
    for filename in os.listdir(UndistortedAndRectifiedDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(UndistortedAndRectifiedDirectory, filename)
            
            TestImg = cv2.imread(image_filename)
            h, w = TestImg.shape[:2]

            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix, DistortionCoeff, RectifiedRotationMatrix, RectifiedProjection, (w, h), cv2.CV_32FC1)
            UndistortedAndRectifiedImg = cv2.remap(TestImg, mapx, mapy, cv2.INTER_LINEAR, borderMode = cv2.BORDER_TRANSPARENT)

            # x, y, w, h = roi
            # UndistortedAndRectifiedImg = UndistortedAndRectifiedImg[y: y + h, x: x + w]

            cv2.imwrite(os.path.join(UndistortedAndRectifiedDirectory, filename), UndistortedAndRectifiedImg)


def plot_camera(f, tan_x, tan_y, ax):
    R_prime = np.identity(3)
    t_prime = np.zeros((3, 1))
    cam_center_local = np.asarray([[0, 0, 0], [tan_x, tan_y, 1], [tan_x, -tan_y, 1], [0, 0, 0], [tan_x, -tan_y, 1], [-tan_x, -tan_y, 1], [0, 0, 0], [-tan_x, -tan_y, 1], [-tan_x, tan_y, 1], [0, 0, 0], [-tan_x, tan_y, 1], [tan_x, tan_y, 1], [0, 0, 0]]).T
    cam_center_local *= f
    cam_center = np.matmul(R_prime, cam_center_local) + t_prime
    ax.plot(cam_center[0, :], cam_center[1, :], cam_center[2, :], color='k', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

if __name__ == '__main__':
    sterioCalibration()
