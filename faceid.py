import cv2

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

# Como un video es una secuencia de imágenes, vamos a procesarlas una a una:
while True:
    sucess, frame = video.read() # sucess nos dira si hemos alamcenado bien las imagenes o no
    
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(frameGray, scaleFactor=1.5, minNeighbors=5)
    if len(face)!=0:
        for (x, y, w, h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
   
    cv2.imshow("video", frame) # mostramos

    if cv2.waitKey(1) & 0xFF==ord('q'): # hay que hacer la espera para que vaya chekeando si pasa algo
        break