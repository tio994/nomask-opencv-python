import numpy as np
import cv2
import winsound

#Library Untuk Mendeteksi Muka
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
#Library Untuk Mendeteksi Mata
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_nose.xml')

#Untuk menjadikan video sebagai pendeteksinya
cap = cv2.VideoCapture(0)
img_counter = 0

while True:
#Untuk Mengatur scale muka 
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
#Untuk Mengatur scale Hidung  
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            winsound.PlaySound('Pakai masker.wav', winsound.SND_FILENAME)
            
#menampilkan video live
    cv2.imshow('img',img)
#Tombol Space untuk menggambil gambar dan menjadikannya format jpg

    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "Muhammad Setyo H_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
      # Press Q on keyboard to  exit
    elif cv2.waitKey(25) & 0xFF == ord('q'):
        print("Escape hit, closing...")
        break

#menampilkan video
cap.release()
cv2.destroyAllWindows()
