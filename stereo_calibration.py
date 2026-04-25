#(1) セットアップとインポート
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

CHECKERBOARD = (6,8) #内側の角 (cols, rows)
SQUARE_SIZE = 10 #mm

CALIB_DIR = "C:\\Users\\Senthil\\Desktop\\calibration\\test_02"
LEFT_KEYWORD = "D1"  #MCX12
RIGHT_KEYWORD = "D0" #MC2026

print("Checkerboard:", CHECKERBOARD,"inner corners")
print("Square size:", SQUARE_SIZE)
print("Calibration directory:", CALIB_DIR)
print("Ready to load images...")

#（２）画像の読み込みと分離
def read_imgs(CALIB_DIR):

    all_images = glob.glob(os.path.join(CALIB_DIR,"*.png"))

    left_images = sorted([f for f in all_images if LEFT_KEYWORD in f])
    right_images = sorted([f for f in all_images if RIGHT_KEYWORD in f])

    print(f"Found {len(left_images)} left images (D1)")
    print(f"Found {len(right_images)} right images (D0)")

    if len(left_images) != len(right_images):
        print("WARNING: Count mismatch!")

    return left_images, right_images

left_imgs, right_imgs = read_imgs(CALIB_DIR)


#（３）チェッカーボードの角を検出する
def detect_corners(left_imgs, right_imgs):
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_size = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    for i, (left_path, right_path) in enumerate(zip(left_imgs, right_imgs)):
        #画像を読む 
        img_left = cv.imread(left_path)
        img_right = cv.imread(right_path)
        #グレースケールに変換する
        gray_left  = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(img_right,cv.COLOR_BGR2GRAY)

        #角を検出する
        ret_left, corners_left  = cv.findChessboardCorners(gray_left, CHECKERBOARD, None)
        ret_right, corners_right= cv.findChessboardCorners(gray_right, CHECKERBOARD, None)
    
        if ret_left and ret_right:
            #角をきれいに仕上げる
            corners_left  = cv.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1), criteria)
            corners_right = cv.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1), criteria)
    
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            image_size = gray_left.shape[::-1]


            print(f"Pair {i+1}/{len(left_imgs)}: Detected")
        else:
            print(f"Pair {i+1}/{len(left_imgs)}: Failed")

    print(f"\nSuccessfully detected: {len(objpoints)} pairs")
    return objpoints, imgpoints_left, imgpoints_right, image_size

#コーナー検出を視覚化する

def visualize_corners(left_imgs, right_imgs, imgpoints_left, imgpoints_right):

    for i in range(min(5, len(imgpoints_left))):
        img_left = cv.imread(left_imgs[i])
        img_right = cv.imread(right_imgs[i])
        
        cv.drawChessboardCorners(img_left, CHECKERBOARD, imgpoints_left[i], True)
        cv.drawChessboardCorners(img_right, CHECKERBOARD, imgpoints_right[i], True)
        #並べて表示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(cv.cvtColor(img_left,  cv.COLOR_BGR2RGB))
        ax2.imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
        ax1.set_title(f"Left Camera - Pair {i+1}")
        ax2.set_title(f"Right Camera - Pair {i+1}")
        plt.tight_layout()
        plt.show()

#すべての関数定義が完了したら、メイン処理を実行します
objpoints, imgpoints_left, imgpoints_right, image_size = detect_corners(left_imgs, right_imgs)
visualize_corners(left_imgs, right_imgs, imgpoints_left, imgpoints_right)

#固有キャリブレーション
def calibrate_intrinsics(objpoints, imgpoints, image_size, camera_name):
    flags = cv.CALIB_RATIONAL_MODEL
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    #キャリブレーションを実行し、5つの値をすべて保存します。
    rms, K, D, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size,
        None, None,
        flags=flags,
        criteria=criteria
    )

    print(f"\n=== {camera_name} Intrinsic Calibration ===")
    print(f"RMS Reprojection Error: {rms:.4f} pixels")
    print(f"Camera Matrix K:\n{K}")
    print(f"Distortion Coefficients:\n{D}")

    return rms, K, D, rvecs, tvecs

#関数の外で呼び出します（カメラごとに1回ずつ）。
rms1, K1, D1, rvecs1, tvecs1 = calibrate_intrinsics(
    objpoints, imgpoints_left, image_size, "Left Camera (D1)")

rms2, K2, D2, rvecs2, tvecs2 = calibrate_intrinsics(
    objpoints, imgpoints_right, image_size, "Right Camera (D0)")

#外れ値の除去

def remove_outliers(objpoints, imgpoints_left, imgpoints_right,
                    K1, D1, rvecs1, tvecs1,
                    K2, D2, rvecs2, tvecs2,
                    threshold=1.0):

    good_objpoints = []
    good_imgpoints_left = []
    good_imgpoints_right = []

    for i in range(len(objpoints)):

        #左カメラの画像ごとの誤差を計算します
        projected_left, _ = cv.projectPoints(
            objpoints[i], rvecs1[i], tvecs1[i], K1, D1)
        error_left = cv.norm(
            imgpoints_left[i], projected_left, cv.NORM_L2) / len(projected_left)

        #右カメラの画像ごとの誤差を計算します
        projected_right, _ = cv.projectPoints(
            objpoints[i], rvecs2[i], tvecs2[i], K2, D2)
        error_right = cv.norm(
            imgpoints_right[i], projected_right, cv.NORM_L2) / len(projected_right)

        #両方のカメラのエラーが低い場合にのみ保持する
        if error_left < threshold and error_right < threshold:
            good_objpoints.append(objpoints[i])
            good_imgpoints_left.append(imgpoints_left[i])
            good_imgpoints_right.append(imgpoints_right[i])
            print(f"Pair {i+1}: KEPT   (L:{error_left:.3f}px R:{error_right:.3f}px)")
        else:
            print(f"Pair {i+1}: REMOVED (L:{error_left:.3f}px R:{error_right:.3f}px)")

    print(f"\nKept {len(good_objpoints)}/{len(objpoints)} pairs after outlier removal")
    return good_objpoints, good_imgpoints_left, good_imgpoints_right

#外れ値を除去する
objpoints_clean, imgpoints_left_clean, imgpoints_right_clean = remove_outliers(
    objpoints, imgpoints_left, imgpoints_right,
    K1, D1, rvecs1, tvecs1,
    K2, D2, rvecs2, tvecs2,
    threshold=1.0
)

#クリーンなデータで再調整する
rms1, K1, D1, rvecs1, tvecs1 = calibrate_intrinsics(
    objpoints_clean, imgpoints_left_clean, image_size, "Left Camera (D1) - Cleaned")

rms2, K2, D2, rvecs2, tvecs2 = calibrate_intrinsics(
    objpoints_clean, imgpoints_right_clean, image_size, "Right Camera (D0) - Cleaned")


#ステレオキャリブレーション

def stereo_calibrate(objpoints, imgpoints_left, imgpoints_right, 
                     K1, D1, K2, D2, image_size):
    flags = cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_RATIONAL_MODEL
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K1, D1,
        K2, D2,
        image_size,
        flags=flags,
        criteria=criteria
    )
    print(f"\n=== Stereo Calibration ===")
    print(f"Stereo RMS: {ret:.4f} pixels")
    print(f"Baseline T:\n{T}")
    print(f"Rotation R:\n{R}")

    return ret, K1, D1, K2, D2, R, T, E, F

#ステレオキャリブレーションを実行します
ret, K1, D1, K2, D2, R, T, E, F = stereo_calibrate(
    objpoints_clean, imgpoints_left_clean, imgpoints_right_clean,
    K1, D1, K2, D2, image_size)

#ステレオ整流

def stereo_rectify(K1, D1, K2, D2, R, T, image_size):

    #整流変換を計算する
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        K1, D1,
        K2, D2,
        image_size,
        R, T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=0.9
    )

    #再マッピングのための補正マップを計算する
    map1_left,  map2_left  = cv.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv.CV_32FC1)

    map1_right, map2_right = cv.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv.CV_32FC1)

    print(f"\n=== Stereo Rectification ===")
    print(f"R1:\n{R1}")
    print(f"R2:\n{R2}")
    print(f"P1:\n{P1}")
    print(f"P2:\n{P2}")

    return R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right

#整流を視覚化する

def visualize_rectification(left_imgs, right_imgs,
                             map1_left, map2_left,
                             map1_right, map2_right):

    #最初の画像ペアを使用する
    img_left  = cv.imread(left_imgs[0])
    img_right = cv.imread(right_imgs[0])

    #補正マップを適用する
    rect_left  = cv.remap(img_left,  map1_left,  map2_left,  cv.INTER_LINEAR)
    rect_right = cv.remap(img_right, map1_right, map2_right, cv.INTER_LINEAR)

    #50ピクセルごとに水平方向のエピポーラ線を描画します。
    h, w = rect_left.shape[:2]
    for y in range(0, h, 50):
        cv.line(rect_left,  (0, y), (w, y), (0, 255, 0), 1)
        cv.line(rect_right, (0, y), (w, y), (0, 255, 0), 1)

    #並べて表示する
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(cv.cvtColor(rect_left,  cv.COLOR_BGR2RGB))
    ax2.imshow(cv.cvtColor(rect_right, cv.COLOR_BGR2RGB))
    ax1.set_title("Rectified Left Camera")
    ax2.set_title("Rectified Right Camera")
    plt.suptitle("Epipolar lines should pass through same features!")
    plt.tight_layout()
    plt.show()

# ステレオ整流を実行する
R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right = stereo_rectify(
    K1, D1, K2, D2, R, T, image_size)

# 整流を視覚化する
visualize_rectification(
    left_imgs, right_imgs,
    map1_left, map2_left,
    map1_right, map2_right)