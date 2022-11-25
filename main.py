import cv2
from face_detection import facedet, matches

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoCapture('clips/clip22.mp4')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(video.get(3)),int(video.get(4))))
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

while True:
    sucess, frame = video.read() # sucess dice si hemos almacenado bien las imagenes
   
    if sucess:
        
        frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(frameGray, scaleFactor=1.5, minNeighbors=5)
        
        if len(faces)!=0: # si detecta alguna cara
            for (x, y, w, h) in faces:
                face = frame[y:y+w,x:x+h]
                nombre = matches(face)
                cv2.putText(frame, nombre, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
           
        out.write(frame)
        cv2.imshow("video", frame) # mostramos

        
        if cv2.waitKey(1) & 0xFF==ord('q'): # hay que hacer la espera para que vaya chekeando si pasa algo
            break

