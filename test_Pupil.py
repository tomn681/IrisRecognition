from unittest import TestCase
import cv2
import Image
import Pupil
import Iris
import Polar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np

path = '/media/dela/1TB/A Universidad/Electrica/VIII Sem/Procesamiento Digital de Imagenes/IrisRec/IrisRecognition/IrisTestImages/'
imageList = ['iris4.bmp', 'iris5.bmp', 'iris6.bmp', 'iris7.bmp',
             'iris8.bmp', 'iris9.bmp', 'iris10.bmp', 'iris11.bmp']


class TestPupil(TestCase):

    def test_get_iris(self):
        image_name = imageList[random.randint(0, len(imageList) - 1)]
        img = Image.Image(path + image_name, binary_threshold=50, kernel_size=(5, 5))
        img2 = Image.Image(path + image_name, binary_threshold=110, kernel_size=(7, 7))
        img3 = Image.Image(path + image_name)
        pupil = Pupil.Pupil(img)
        pupil_circle = pupil.get_pupil()[0]
        iris = Iris.Iris(img2, pupil_circle[:2], min_radius=pupil_circle[2])
        iris_circle = iris.get_iris()
        if iris_circle is None:
            iris_circle = ((None, 100, 100), 0)
        iris_circle = iris_circle[0]
        img2 = Polar.Polar(img3, iris_circle, pupil_circle)
        img2.image.size()
        img2.image.clean_image()
        img2.img_polar()
        img2.print()
        img2.get_features(10, range(3, 6, 2))
