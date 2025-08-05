import cv2

alg = "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg) #Loading algorithm

cam = cv2.VideoCapture(0) #Cam id

while True:
    _,img = cam.read() #Read frame
    
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray
    
    face = haar_cascade.detectMultiScale(grayImg,1.3,4) #coordinate
    
    for (x,y,w,h) in face:
        
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,255,0),5)
    cv2.imshow("FaceDetection",img)
    
    key = cv2.waitKey(10)
    print(key)
    if key == 27: #Esc
        break
cam.release()
cv2.destroyAllWindows()
