import cv2
import numpy as np
import dlib

webcam = False
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_20_lip_landmarks.dat")

def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Blue","BGR",153,255,empty)
cv2.createTrackbar("Green","BGR",0,255,empty)
cv2.createTrackbar("Red","BGR",137,255,empty)

def createBox(img,points,scale=5,masked= False,cropped= True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        img = cv2.bitwise_and(img,mask)


    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        cv2.imwrite("Mask.jpg",imgCrop)
        return imgCrop
    else:
        return mask

while True:

    if webcam: success,img = cap.read()
    else: img = cv2.imread('GettyImages-1092658864_hero-1024x575.jpg')
    img = cv2.resize(img,(0,0),None,0.6,0.6)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(imgOriginal)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        landmarks = predictor(imgGray, face)
        myPoints =[]
        for n in range(20):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])

        if len(myPoints) != 0:
            try:
                myPoints = np.array(myPoints)
                imgLips = createBox(img, myPoints)
                maskLips = createBox(img, myPoints,masked = True,cropped=False)
                imgColorLips = np.zeros_like(maskLips)

                b = cv2.getTrackbarPos("Blue", "BGR")
                g = cv2.getTrackbarPos("Green", "BGR")
                r = cv2.getTrackbarPos("Red", "BGR")

                imgColorLips[:] = b,g,r
                imgColorLips = cv2.bitwise_and(maskLips,imgColorLips)
                imgColorLips = cv2.GaussianBlur(imgColorLips,(7,7),10)
                imgColorLips = cv2.addWeighted(imgOriginal ,1,imgColorLips,0.4,0)

                cv2.imshow('BGR', imgColorLips)

            except:
                pass

    cv2.imshow("Originial", imgOriginal)
    cv2.waitKey(1)