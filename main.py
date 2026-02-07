import cv2
import time 
import os
import numpy as np 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = "hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands = 1
)

detector = HandLandmarker.create_from_options(options)

folderpath = "presentation"
imgpath = os.listdir(folderpath)
print(imgpath)


# --------------Variables--------------
WIDTH, HEIGTH = 1280, 720
imgNumber = 0
wcam, hcam = 120, 213



cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGTH)


while True:
    success, img = cap.read()
    
    pathimage = os.path.join(folderpath, imgpath[imgNumber])
    currentimg = cv2.imread(pathimage)
    
    if not success:
        break
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
    
    result = detector.detect(mp_image)
    
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            
            h, w, _ = img.shape
            lm_list = []
            
            for lm in hand:
                lm_list.append((int(lm.x * w), int(lm.y * h)))
            
            for x, y in lm_list:
                cv2.circle(img, (x, y), 8, (0, 255, 0), cv2.FILLED)
    
    h, w, _ = currentimg.shape
    
    resized_webcam = cv2.resize(img, (wcam, hcam))
    currentimg = cv2.resize(currentimg, (WIDTH, HEIGTH))
    currentimg[0:hcam, WIDTH - wcam:WIDTH] = resized_webcam
    cv2.imshow("Presentation", currentimg)
    # cv2.imshow("WebCam", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows