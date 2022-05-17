from image import Imagenes
import numpy as np
#import seaborn.histplot as plot_histograma
import matplotlib.pyplot as plt
from numpy import log

def ajustarBrillo(imagen:'Imagenes', escalar:int = 2):
    '''
    Retorna un objeto Imagenes con el brillo ajustado.

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se le desea ajustar el brillo
        -escalar (int): valor por el cual ajustar el brillo. Debe ser entre -255 y 255. Valor pre establecido 2.
    '''

    if escalar <= 255 and escalar >= -255:

        imagen_brillo = Imagenes(image = imagen.image*escalar) #numpy power
        return imagen_brillo
        
    #     matriz = imagen.image
    #     filas = imagen.fil
    #     columnas = imagen.col

    #     for canal in range(3):
    #         for fila in range(filas):
    #             for columna in range(columnas):
    #                 matriz[fila][columna][canal] = (matriz[fila][columna][canal])*escalar

    #     imagen_brillo = Imagenes(image = matriz)
    #     return imagen_brillo
    else:
        raise ValueError('El escalar se escapa del límite entre -255 y 255')

def getHistograma(imagen:'Imagenes'):
    '''
    Retorna una matriz correspondiente al histograma de los canales RGB (repeticiones, canal).

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se desea obtener sus histogramas, debe ser RGB
    '''

    matriz = imagen.image
    filas = imagen.fil
    columnas = imagen.col
    histograma = np.zeros((256, 3))

    for canal in range(3):
        for fila in range(filas):
            for columna in range(columnas):
                histograma[matriz[fila][columna][canal]][canal] += 1

    return histograma

def plot_histograma(imagen:'Imagenes', rango:int = 256, canal:int = 1):
    '''
    Grafica el histograma correspondiente a un canal de la instancia Imagenes actual

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se desea graficar un histograma, debe ser RGB
        -rango (int): cantidad de valores diferentes de píxeles en la matriz de imagen. Valor pre establecido 256.
        -canal (int): canal del que se desea graficar su histograma. 0 para canal rojo, 1 para canal verde, 
        2 para canal azul. Valor pre establecido 1.
    '''

    histogram = getHistograma(imagen)[:,canal]

    plt.bar(x = list(range(rango)), height = histogram)
    plt.title(f'Histograma del canal {canal}')
    plt.xlabel('Píxel')
    plt.ylabel('Cantidad')
    plt.show()

def getChannels(imagen:'Imagenes'):
    '''
    Retorna 3 instancias de la Imagenes correspondientes a cada canal de una sola clase Imagenes.

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes de la que desean obtener sus canales, debe ser RGB
    '''
    matriz = imagen.image

    red_channel = Imagenes(image = matriz[:,:,0])
    green_channel = Imagenes(image = matriz[:,:,1])
    blue_channel = Imagenes(image = matriz[:,:,2])

    return red_channel, green_channel, blue_channel

def contraste(pixel:float, alpha:float = 1):
    '''
    Retorna el contraste correspondiente a un pixel de entrada y un factor alpha.

    Argumentos:
        -pixel (float): valor correspondiente a un píxel de la imagen.
        -alpha (float): factor correspondiente al ajuste de contraste. Valor pre establecido 1.
    '''
    return alpha*(pixel - 128) + 128

def ajustarContraste(imagen:'Imagenes', alpha:float = 1):
    '''
    Retorna un objeto de clase Imagenes al cual se le hace un ajuste de contraste según una constante alpha

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se desea ajustar su contraste
        -alpha (float): factor de contraste
    '''
    matriz = imagen.image
    filas = imagen.fil
    columnas = imagen.col

    for canal in range(3):
        for fila in range(filas):
            for columna in range(columnas):
                matriz[fila][columna][canal] = contraste(matriz[fila][columna][canal], alpha)

    contraste_ajustado = Imagenes(matriz)

    return contraste_ajustado

def logaritmo(matriz:'numpy.ndarray', c:float):
    '''
    Retorna una matriz con la transformación logarítmica

    Argumentos:
        -matriz (np.array): matriz correspondiente a una imagen
        -c (float): factor utiizado
    '''
    return c*log(1 + matriz)

def aplicarLog(imagen:'Imagenes'):
    '''
    Retorna un objeto de clase Imagenes al cual se le hace un ajuste de contraste según una constante alpha

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se desea aplicar la transformación 
                            logarítmica
    '''

    matriz = imagen.image

    for canal in range(3):
        pixel_vmax = np.amax(matriz[:,:,canal])
        c = 255 / (log(1 + pixel_vmax))
        matriz[:,:,canal] = logaritmo(matriz[:,:,canal], c)

    image_clog = Imagenes(matriz)

    return image_clog

def ajustarGamma(imagen:'Imagenes', gam:float = 1/4):
    '''
    Retorna un objeto de clase Imagenes al cual se le hace un ajuste gamma

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se desea ajustar su gamma
        -gam (float): factor de ajuste gamma
    '''    

    if gam >= 0:
        matriz = imagen.image

        pixel_vmax = np.amax(matriz)
        c = 255 / (pixel_vmax**(gam))
        matriz = c*(matriz**(gam))

        image_gamma = Imagenes(matriz)

        return image_gamma

    else:
        raise ValueError('El gamma seleccionado debe ser mayor a 0!')

def zero_padding(imagen, kernel_size):
    pass
    
def aplicarKernel(imagen:'Imagenes', kernel:'numpy.ndarray'):
    '''
    Retorna un objeto de clase Imagenes con un filtrado a partir de la aplicacion de un kernel

    Argumentos:
        -imagen (Imagenes): instancia de la clase Imagenes a la que se le desea aplicar el kernel
        -kernel (numpy array): mascara a aplicar sobre la imagen
    '''
    
    matriz = imagen.image
    filas = imagen.fil
    columnas = imagen.col

    kfilas = kernel.shape[0]
    kcolumnas = kernel.shape[1]

    convolucion = np.zeros((filas, columnas, 3))
    kernel = np.reshape(kernel, (9)) 

    for canal in range(3):
        for fila in range(filas):
            for columna in range(columnas):
                try:
                    afiltrar = np.reshape(matriz[fila:fila+kfilas,columna:columna+kcolumnas,canal], (9))
                    convolucion[fila][columna][canal] = np.dot(afiltrar, kernel)
                except:
                    pass
    
    im_convolucion = Imagenes(convolucion)

    return im_convolucion

def main():
    nueva_imagen = Imagenes(filename = 'images/cerebral-angiography_thumb.jpg')
    nueva_imagen.showImage()

    imagen_brillo = ajustarBrillo(nueva_imagen)
    imagen_brillo.showImage()

    hist = getHistograma(nueva_imagen)

    plot_histograma(nueva_imagen, 256, 1)
    
    r, g, b = getChannels(nueva_imagen)
    print(type(r))

    contraste_ajustado = ajustarContraste(nueva_imagen, 3)
    contraste_ajustado.showImage()

    image_clog = aplicarLog(nueva_imagen)
    image_clog.showImage()

    image_gamma = ajustarGamma(nueva_imagen, 2)
    image_gamma.showImage()

    kernel = np.array([(1, 2, 1),
                       (0, 0, 0),
                       (-1, -2, -1)], dtype = 'int8')# kernel top sobel

    image_filtrada = aplicarKernel(nueva_imagen, kernel)
    image_filtrada.showImage()

main()