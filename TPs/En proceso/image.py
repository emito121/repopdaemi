import matplotlib.pyplot as plt
import cv2

class Imagenes():
    """Crea instancias en las cuales almacena una imgen (array) y cuenta con métodos para interactuar con esta"""

    def __init__(self, image = None, filename:str = ""):
        """Almacena un array 2D o 3D o lo crea a partir de su directorio y nombre"""

        try:
            self.image = image
            self.fil = self.size()[0] #número de filas
            self.col = self.size()[1] #número de columnas
        
        except:
            self.image = self.loadImage(filename)
            self.fil = self.size()[0] #número de filas
            self.col = self.size()[1] #número de columnas
        
        else:
            print("Se creó la imagen sin problemas")

    def loadImage(self, filename:str):
        '''
        Retorna un array correspondiente a una imagen.

        Argumentos:
            -filename (str): dirección correspondiente a una imagen en el almacenamiento
        '''
        return cv2.imread(filename)

    def showImage(self):
        '''
        Muestra la imagen almacenada en la instancia de clase Imagenes.
        '''
        dpi = 80

        height, width  = self.image.shape[:2]

        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')

        ax.imshow(self.image)

        plt.show()

    def size(self):
        '''
        Retorna una tupla correspondiente a las dimensiones de la imagen de la forma (filas, columnas, capas)
        '''
        return self.image.shape

    def saveImage(self, nombre_archivo:str = 'imagen_nueva.jpg'):
        '''
        Almacena la imagen del objeto actual, preferentemente es un .jpg

        Argumentos:
            -nombre_archivo (str): nombre de la imagen almacenada. Valor pre establecido 'imagen_nueva.jpg'
        '''
        cv2.imwrite(nombre_archivo,self.image)

def main():
    nueva_imagen = Imagenes(filename = 'images/bote.jpg')
    nueva_imagen.showImage()
    #print(type(nueva_imagen.image))

#main()