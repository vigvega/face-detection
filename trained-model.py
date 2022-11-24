import os
import cv2

images = os.path.dirname(os.path.abspath(__file__)+"images")
# ver forma de iterar aqui
format = ["jpg", "jpeg", "JPG", "png"]
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
xtrain = []

# Primero voy a buscar los archivos que sean imagenes y voy a quedarme con su ruta
for root, dir, files in os.walk(images):
    for f in files:
        if f.endswith(format[0]) | f.endswith(format[1]) | f.endswith(format[2]) |  f.endswith(format[3]):
            path = os.path.join(root,f)
            label = os.path.basename(root)
            print(label, path)

            # Leo mi imagen y la paso a gris
            img = cv2.imread(path)
            imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # A la imagen gris le voy a pasar el detector facial
            face = face_cascade.detectMultiScale(imgGray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in face:
                imgGray = imgGray[x:x+w, y:y+h] #recorto la cara, que es la parte que me interesa
                xtrain.append(imgGray)

# ahora a cada uno de esas matrices tendré que asignarle un label
# DUDA: cómo obtengo una sola matriz para dos imagenes

print(xtrain)