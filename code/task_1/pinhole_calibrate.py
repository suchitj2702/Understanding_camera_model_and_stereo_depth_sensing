import os
import cv2
import numpy as np


def CalibrateCameraForImages(camera):
    ImagesDirectory = "../../images/task_1"
    CornerPointsDirectory = "../../output/task_1/CornerPointsMarked"
    UndistortedDirectory = "../../output/task_1/Undistorted"
    n = 9
    m = 6

    Points3D = np.zeros((n*m, 3), np.float32)
    Points3D[:,:2] = np.mgrid[0: n, 0: m].T.reshape(-1, 2)

    PointsArray3D = []
    ImageArray2D = []

    for filename in os.listdir(ImagesDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(ImagesDirectory, filename)

            img = cv2.imread(image_filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (n, m), None)
            
            if ret == True:
                PointsArray3D.append(Points3D)
                ImageArray2D.append(corners)
                cv2.drawChessboardCorners(img, (n, m), corners, ret)
                cv2.imwrite(os.path.join(CornerPointsDirectory, filename), img)
    
    ret, CamMatrix, DistortionCoeff, RotationVecs, TranslationVecs = cv2.calibrateCamera(PointsArray3D, ImageArray2D, gray.shape[::-1], None, None)

    np.save('../../parameters/' + camera + 'CameraIntrinsics' + '/IntrinsicMatrix', CamMatrix)
    np.save('../../parameters/' + camera + 'CameraIntrinsics' + '/DistortCoeff', DistortionCoeff)

    for filename in os.listdir(ImagesDirectory):
        if filename.startswith(camera):
            image_filename = os.path.join(ImagesDirectory, filename)
            
            TestImg = cv2.imread(image_filename)
            h, w = TestImg.shape[:2]

            mapx, mapy = cv2.initUndistortRectifyMap(CamMatrix, DistortionCoeff, np.eye(3, 3), CamMatrix, (w,h), cv2.CV_16SC2)
            UndistortedImg = cv2.remap(TestImg, mapx, mapy, cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(UndistortedDirectory, filename), UndistortedImg)
    

if __name__ == '__main__':
    print("Calibrating using left camera images...")
    CalibrateCameraForImages("left")

    print("Calibrating using right camera images...")
    CalibrateCameraForImages("right")

    print("Done")
