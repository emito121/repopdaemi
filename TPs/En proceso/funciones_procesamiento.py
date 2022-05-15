from image import Imagenes
import numpy as np
#import seaborn.histplot as plot_histograma
import matplotlib.pyplot as plt
from numpy import log

def ajustarBrillo(imagen, escalar = 1):
    
    if escalar <= 255 and escalar >= -255:
        matriz = imagen.image
        for i in range(3):
            matriz[:,:,i] = matriz[:,:,i]*escalar
        print(matriz.shape)
        imagen_brillo = Imagenes(image = matriz)
        return imagen_brillo
    else:
        raise ValueError('El escalar se escapa del límite entre -255 y 255')

def getHistograma(imagen):
    
    matriz = imagen.image
    filas = imagen.fil
    columnas = imagen.col
    histograma = np.zeros((256, 3))

    for canal in range(3):
        for fila in range(filas):
            for columna in range(columnas):
                histograma[matriz[fila][columna][canal]][canal] += 1

    return histograma

def plot_histograma(imagen, rango, canal):

    histogram = getHistograma(imagen)[:,canal]

    plt.bar(x = list(range(rango)), height = histogram)
    plt.title(f'Histograma del canal {canal}')
    plt.xlabel('Píxel')
    plt.ylabel('Cantidad')
    plt.show()

def getChannels(imagen):

    matriz = imagen.image

    red_channel = Imagenes(image = matriz[:,:,0])
    green_channel = Imagenes(image = matriz[:,:,1])
    blue_channel = Imagenes(image = matriz[:,:,2])

    return red_channel, green_channel, blue_channel

def contraste(pixel, alpha):
    return alpha*(pixel - 128) + 128

def ajustarContraste(imagen, alpha):
    
    matriz = imagen.image
    filas = imagen.fil
    columnas = imagen.col

    for canal in range(3):
        for fila in range(filas):
            for columna in range(columnas):
                matriz[fila][columna][canal] = contraste(matriz[fila][columna][canal], alpha)

    contraste_ajustado = Imagenes(matriz)

    return contraste_ajustado

def logaritmo(matriz, c):
    return c*log(1 + matriz)

def aplicarLog(imagen):
    
    matriz = imagen.image
    # filas = imagen.fil
    # columnas = imagen.col

    for canal in range(3):
        pixel_vmax = np.amax(matriz[:,:,canal])
        c = 255 / (log(1 + pixel_vmax))
        matriz[:,:,canal] = logaritmo(matriz[:,:,canal], c) #sacandole provecho a numpy, aunque los for tambien sirven :)
        # for fila in range(filas):
        #     for columna in range(columnas):
        #         matriz[fila][columna][canal] = logaritmo(matriz[fila][columna][canal], c)

    image_clog = Imagenes(matriz)

    return image_clog

def ajustarGamma(imagen, gam):
    
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
    
def aplicarKernel(imagen, kernel):

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
                # print(np.dot(afiltrar, kernel))
                # break
                # try:
                #     convolucion[fila][columna][canal] = np.dot(matriz[fila:fila+kfilas,columna:columna+kcolumnas,canal], kernel)
                # except:
                #     pass
    
    im_convolucion = Imagenes(convolucion)

    return im_convolucion

def main():
    nueva_imagen = Imagenes(filename = 'images/bote.jpg')
    #nueva_imagen.showImage()
    #imagen_brillo = ajustarBrillo(nueva_imagen, 3)
    #ajustarBrillo(nueva_imagen, 1)
    #imagen_brillo.showImage()

    #hist = getHistograma(nueva_imagen)

    #plot_histograma(nueva_imagen, 256, 1)
    
    # r, g, b = getChannels(nueva_imagen)
    # print(type(r))

    # contraste_ajustado = ajustarContraste(nueva_imagen, 3)
    # contraste_ajustado.showImage()

    # image_clog = aplicarLog(nueva_imagen)
    # image_clog.showImage()

    # image_gamma = ajustarGamma(nueva_imagen, 2)
    # image_gamma.showImage()

    kernel = np.array([(1, 2, 1),
                       (0, 0, 0),
                       (-1, -2, -1)], dtype = 'int8')

    image_filtrada = aplicarKernel(nueva_imagen, kernel)
    image_filtrada.showImage()

# main()