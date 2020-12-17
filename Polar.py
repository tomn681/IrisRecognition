import numpy as np
import matplotlib.pyplot as plt
import cv2


def polar_cartesian(r, theta, center):
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


class Polar:
    def __init__(self, image, iris_circle, pupil_circle):
        self.image = image
        self.iris_circle = iris_circle
        self.pupil_circle = pupil_circle

    def img_polar(self, phase_width=3000):
        center = (int(self.pupil_circle[0]), int(self.pupil_circle[1]))
        initial_radius = int(self.pupil_circle[2])
        final_radius = int(self.iris_circle[2])
        if initial_radius is None:
            initial_radius = 0
        theta, r = np.meshgrid(np.linspace(0, 2 * np.pi, phase_width),
                               np.arange(initial_radius, final_radius))
        x_cart, y_cart = polar_cartesian(r, theta, center)
        x_cart = x_cart.astype(int)
        y_cart = y_cart.astype(int)
        if self.image.image.ndim == 3:
            polar_img = self.image.image[y_cart, x_cart, :]
            polar_img = np.reshape(polar_img, (final_radius - initial_radius, phase_width, 3))
        else:
            polar_img = self.image[y_cart, x_cart]
            polar_img = np.reshape(polar_img, (final_radius - initial_radius, phase_width))
        self.image.image = polar_img

    def print(self):
        plt.imshow(self.image.image, cmap='gray')
        plt.title('Polar-Cartesian Image')
        plt.show()

    def gabor_filtering(self, kernel_size, theta, wavelength):
        kernel_filter = cv2.getGaborKernel(kernel_size, sigma=1.0, theta=theta, lambd=wavelength, gamma=0)
        return cv2.filter2D(self.image.image, -1, kernel_filter)

    def get_features(self, n_filters_theta, wl_range):
        angle = np.pi / n_filters_theta
        features = np.zeros((1,), dtype=int)
        for wavelength in wl_range:
            for i in range(n_filters_theta):
                gabor = self.gabor_filtering((21, 21), i * angle, wavelength)
                new_features = (np.average(gabor), np.mean(gabor),
                                np.median(gabor), np.std(gabor))
                for feature in new_features:
                    features = np.concatenate((features, feature.astype(int).flatten()))
        return features
