import Image
import Pupil
import Iris
import Polar
import random
import os
import numpy as np
from pathlib import Path

data_path = '/media/dela/1TB/A Universidad/Electrica/VIII Sem/Procesamiento Digital de Imagenes/IrisRec/IrisRecognition/IrisTestImages/'
imageList = ['iris4.bmp', 'iris5.bmp', 'iris6.bmp', 'iris7.bmp',
             'iris8.bmp', 'iris9.bmp', 'iris10.bmp', 'iris11.bmp']


def get_features(image_path):
    img = Image.Image(image_path, binary_threshold=50, kernel_size=(5, 5))
    img2 = Image.Image(image_path, binary_threshold=110, kernel_size=(7, 7))
    img3 = Image.Image(image_path)
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
    features = img2.get_features(6, range(3, 14, 2))
    return features


def train(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith(".bmp") and not filename.endswith("1.bmp"):
                f_name = 'features/' + filename[:-4] + '.txt'
                with open(f_name, 'w+') as file:
                    try:
                        file_path = os.path.join(subdir, filename)
                        file.write('{}\n'.format(Path(file_path).stem))
                        lista = get_features(file_path).tolist()
                        for item in lista:
                            file.write('{}\n'.format(item))
                    except Exception:
                        print('File: {} error'.format(filename))


def hamming_distance(feature1, feature2):
    return int(np.linalg.norm(feature1 - feature2))


def detect(path, detect_image_name):
    names, distance = [], []
    for file_name in os.listdir('./features/'):
        with open('./features/' + file_name, 'r') as file:
            try:
                features = get_features(os.path.join(path, detect_image_name))
                list_features = []
                lines = file.readlines()
                for line in lines:
                    if line != '':
                        list_features.append(line.replace('\n', ''))
                distance.append(hamming_distance(np.asarray(list_features[1:302], dtype=int), features))
                names.append(list_features[0])
            except Exception:
                pass #print('File: {} error'.format(detect_image_name))
    if not distance:
        return "PepsiCola"
    i = np.argmin(distance)
    return names[i]


if __name__ == '__main__':
    for train_data in (True, False):
        image_name = imageList[random.randint(0, len(imageList) - 1)]
        if train_data:
            train(data_path)
        else:
            correct = 0
            total = 0
            for subdir, dirs, files in os.walk(data_path):
                for filename in files:
                    if filename.endswith("1.bmp"):
                        name = detect(subdir, filename)
                        print('Real class: ', filename[:-6], 'Detected class: ', name[:-2])
                        if filename[:-6] == name[:-2]:
                            correct += 1
                        total += 1
            print('Accuracy: ', correct/total)
