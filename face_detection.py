import cv2
from facenet_pytorch import InceptionResnetV1
import torch
import torchvision
import os
import numpy as np

# rutas
images = "/home/vi/fr/images"
images_processed = "/home/vi/fr/images_processed"

embeddings = []
ids = []



# detector
def facedet(image):
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    img = cv2.imread(image)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # invierto canales
    imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY) # imagen en blanco y negro

    faces = face_cascade.detectMultiScale(imgGray) # detecta todas las caras que haya en la imagen
    # aunque en este caso esta funcion será para procesar imágenes con una cara
    
    for (x, y, w, h) in faces:
        f = img[y:y+w,x:x+h] # recorto las caras

# (si reconociese más de una cara, me devolvería la última)
    return f



def saveImg(img, name):
    fname =  f'images_processed/{name}.jpg'
    cv2.imwrite(fname, img)



def embedding(image):
    resnet = InceptionResnetV1(pretrained='vggface2').eval() # inicializo resnet

    if isinstance(image, np.ndarray)==False:
        img = cv2.imread(image) # leo mi imagen (y la utilizaré como nparray)
    else:
        img = image

    img_tensor = torchvision.transforms.functional.to_tensor(img) # transformo mi imagen a tensor
    img_embedding = resnet(img_tensor.unsqueeze(0)).detach() #embedding matrix
    
    return img_embedding




def recolect():
    # recorro la carpeta donde están todas mis imagenes sin procesar y las proceso
    # primero voy a buscar los archivos que sean imagenes y voy a quedarme con su ruta
    print("Detecting...")
    for root, dir, files in os.walk(images):
        for f in files:
            path = os.path.join(root,f) # ruta de todas mis imagenes
            face = facedet(path) # proceso las caras
            saveImg(face, f.split(".")[0])
   
    print("Embedding...")
    # a las imagenes procesadas les meto lo de embedding y guardo el resultado en un array
    for root, dir, files in os.walk(images_processed):
        for f in files:
            path = os.path.join(root,f) # ruta de todas mis imagenes
            id = f.split(".")[0]
            matrix = embedding(path)

            embeddings.append(matrix.detach())
            ids.append(id)

    print("Saving results...")
    # creo otro array con el array de embeddings y los nombres
    known_faces = [embeddings, ids]
    torch.save(known_faces, 'known_faces.pt') # guardo mi resultado



def matches(img):
    emb = embedding(img)

    faces = torch.load("known_faces.pt")
    data = faces[0]
    names = faces[1]

    parecidos = []

    for i, e in enumerate(data):
        parecido = torch.dist(e, emb).item()
        parecidos.append(parecido)
    
    
    return names[parecidos.index(min(parecidos))]
    '''
    if min(parecidos)>0.8:
        return "?????"
    else:
        return names[parecidos.index(min(parecidos))]
    '''
