import numpy as np
import matplotlib.pyplot as plt
import cv2


class Iris:
    def __init__(self, image, center, min_radius=10):
        self.image = image
        self.center = center
        self.min_radius = min_radius

    def hough_transform(self, canny_min, canny_max, min_radius):
        self.image.grayscale()
        self.image.hough_preprocessing(canny_min, canny_max)
        rows = self.image.size()[0]
        circles = cv2.HoughCircles(self.image.image, cv2.HOUGH_GRADIENT,
                                   1, rows / 8, param1=500, param2=20,
                                   minRadius=min_radius, maxRadius=100)
        return circles[0] if circles is not None else None

    def print_circles(self, circles):
        fig = plt.figure(figsize=(15, 15))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(self.image.backup_image, center, radius, (255, 255, 255), 1)
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.image.image, cmap='gray')
        plt.title('canny_image')
        fig.add_subplot(1, 2, 2)
        plt.imshow(self.image.backup_image, cmap='gray')
        plt.title('detected_circle')
        plt.show()

    def get_iris(self):
        circles = self.hough_transform(canny_min=500, canny_max=600, min_radius=int(self.min_radius))
        return circles
