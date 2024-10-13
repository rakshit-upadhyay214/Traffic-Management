import math

import numpy as np

from sort import *

from ultralytics import YOLO
import cv2
import cvzone
# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/cars.mp4")
mask= cv2.imread("../Videos/mask.png")
model = YOLO('../Yolo-Weights/yolov8l.pt')

classNames= ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker= Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits= [400,297,673,297]
totalCount=[]
while True:
    success, img= cap.read()
    imgRegion= cv2.bitwise_and(img,mask)
    imgGraphics = cv2.imread("../Videos/graphics.png",cv2.IMREAD_UNCHANGED)
    img= cvzone.overlayPNG(img,imgGraphics,(0,0))
    if not success:
        break  # Stop if the video ends or there's an issue with the video stream

    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1, y1, x2, y2=box.xyxy[0]
            x1, y1, x2, y2= int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1
            conf=math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            if (classNames[cls]== "car" or classNames[cls]=="truck" or classNames[cls]=="bus" or classNames[cls]=="motorbike") and conf>0.3:
                # Display the rectangle
                # cvzone.cornerRect(img, (x1, y1, w, h), l=8)
                # Display the class name and confidence
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections= np.vstack((detections, currentArray))
            else:
                print(f"Class index {cls} out of range for classNames")


    resultsTracker= tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]),(0,0,255), 5)
    for result in resultsTracker:
        x1,y1,x2,y2,id =result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255,0,0))
        # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
        #                    offset=8)

        cx,cy= x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy),5,(0,255,0),cv2.FILLED)
        if limits[0]< cx <limits[2] and limits[1]-15 < cy <limits[1]+15:
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'Count:{len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
