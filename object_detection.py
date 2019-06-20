
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

watch_cascade = cv2.CascadeClassifier('cascade.xml')

hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(5,5),0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    hand = hand_cascade.detectMultiScale(gray, 1.3, 5)
    mask = np.zeros(thresh1.shape, dtype = "uint8")
    
    
    watches = watch_cascade.detectMultiScale(gray, 100, 100)
    
   
    for (x,y,w,h) in hand:
            print ("Found hand"), len(hand), ("Hand!")
            cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2)
            cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Hand',(x,y), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
            #img2 = cv2.bitwise_and(thresh1, mask)
	
    for (x,y,w,h) in watches:
        print ("Found watch"), len(watches), ("Watch!")
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Watch',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    for (x,y,w,h) in faces:
        print ("Found face"), len(faces), ("face!")
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Face',(x,y), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.7,
        minNeighbors=22,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            print ("Found eyes"), len(eyes), ("eyes!")
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        for (x, y, w, h) in smile:
            print ("Found smiles"), len(smile), ("smiles!")
            cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 0, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Smile',(w,h), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('img',img)
   #cv2.imshow('img2',img2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
