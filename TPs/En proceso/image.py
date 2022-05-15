import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

class Imagenes():
    """Documentación de Image"""

    def __init__(self, image = None, filename:str = ""):
        """Crea una matriz 2D a partir de image o de filename"""

        try:
            self.image = image
            self.fil = self.size()[0] #número de filas
            self.col = self.size()[1] #número de columnas
        
        except:
            self.image = self.loadImage(filename)
            self.fil = self.size()[0] #número de filas
            self.col = self.size()[1] #número de columnas
        
        # else:
        #     raise ValueError("No se ha dado una imágen ni un archivo")

    def loadImage(self, filename:str):
        return cv2.imread(filename)

    def showImage(self):
        dpi = 80

        height, width  = self.image.shape[:2]

        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')

        ax.imshow(self.image)

        plt.show()

    def size(self):
        return self.image.shape

    def saveImage(self, nombre_archivo, filename:str):
        imagen = self.loadImage(filename)
        cv2.imwrite(nombre_archivo,imagen)

def main():
    nueva_imagen = Imagenes(filename = 'images/bote.jpg')
    #print(type(nueva_imagen.image))

#main()