## Face-recognition

### Introducción
El objetivo de este proyecto es diseñar un programa sencillo de reconocimiento facial en tiempo real. Para ello, he dividido el proceso en dos fases claramente diferenciadas: detección y reconocimiento.

Todo el código se ha desarrollado en python, con el apoyo de librerías como OpenCV, Pytorch, Numpy y Tensor.

---

## Detección
El primer paso era detectar los rostros. Para ello, he probado con dos acercamientos distintos:
 #### Algoritmo MTCNN
Este algoritmo es bastante fiable, pues es capaz de detectar rostros usando tres redes convolucionales en cascada. No me voy a extender en su explicación pues, debido a su alto tiempo de cómputo, he optado por utilizar otra alternativa. Pero es un método que se puede aplicar fácilmente si disponemos de la librería Pytorch:
    
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=True, device='cpu')
        # sería mejor utilizar 'cuda' si es posible


 #### Haar cascade
 Esta técnica se basa en el entrenamiento de un [clasificador en cascada](https://en.wikipedia.org/wiki/Cascading_classifiers) con muestras positivas y negativas. Así, lo que conseguirá es diferenciar entre la presencia y la no presencia de cierto objeto en una imagen (en nuestro caso rostros).
 
 Con el clasificador ya entrenado, se le dará una imagen e intentará buscar ciertas características aplicando a porciones de ella la función Haar. Estas características las vamos a llamar "clasificadores débiles", pues suelen dar lugar a falsos positivos. Posteriormente, vamos a unir estos clasificadores, obteniendo así el resultado final, que será mucho más fiable.

 Por suerte, OpenCV tiene ya una serie de [clasificadores pre entrenados](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html). Lo único que debemos de tener en cuenta es que están entrenados con imágenes en blanco y negro, por lo que así deberá de ser la imagen que le demos.
 
 Lo que nos devolverá para cada rostro será una cuádrupla del modo (x, y, w, h), donde (x, y) son las coordenadas de la esquina superior izquierda y (w, h) el ancho y alto del recuadro en el que está el rostro.
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
        # Inicializo el clasificador que desee
        
        faces = face_cascade.detectMultiScale(imgGray) 
        # Me devolverá los bounding boxes de cada rostro

---

### Reconocimiento
Para esta fase he utilizado la arquitectura de red neuronal convolucional [Inception-ResNet-V1](https://pypi.org/project/facenet-pytorch/) que, como su propio nombre indica, está basada en las redes Inception y ResNet.

El modelo Inception-V1 se basa en el problema de que es muy complicado obtener características similares entre varias imagenes, pues los resultados pueden cambiar significativamente en función del tamaño de la imagen y otros criterios.

Este problema lo soluciona generando redes que son más amplias que profundas. Es decir, la red va a aplicar filtros de distinto tamaño en el mismo nivel. Al aplicar estos filtros, se concatenarán todas las salidas y eso será lo que le llegue a la siguiente capa. 

Por su parte, el modelo [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) es un método sorprendentemente eficiente, pues se basa en la idea de que un mayor número de capas, no nos asegura resultados mejores. Es por esto por lo que busca "atajos" que aceleran el proceso y, además, nos da resultados incluso mejores que utilizando un mayor número de capas.

La combinación de estos dos modelos nos otorga una herramienta verdaderamente potente para, en este caso, el reconocimiento facial. Es por eso que Pytorch nos proporciona un módulo con el que podemos utilizarlos:

        from facenet_pytorch import InceptionResnetV1
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        ...
        img_embedding = resnet(img_tensor.unsqueeze(0))

---

To-do:
 * procesar varias imagenes de cada persona
 * cambiar outputs con distancias demasiado altas
 * excepciones
 * mejorar rendimiento