import cv2

BLUR_SIGMA = 0


class Image:
    def __init__(self, image_path, canny_under_threshold=400, canny_upper_threshold=500, binary_threshold=50,
                 kernel_size=(5, 5)):
        self.binary_threshold = binary_threshold
        self.canny_upper_threshold = canny_upper_threshold
        self.canny_under_threshold = canny_under_threshold
        image = cv2.imread(image_path)
        self.image = image
        self.backup_image = image
        self.kernel_size = kernel_size

    def grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.backup_image = self.image

    def hough_preprocessing(self, canny_min, canny_max):
        thresh, binary_image = cv2.threshold(self.image, self.binary_threshold, maxval=255, type=cv2.THRESH_BINARY)
        blurred_image = cv2.GaussianBlur(binary_image, self.kernel_size, BLUR_SIGMA)
        self.image = cv2.Canny(blurred_image, canny_min, canny_max)

    def size(self):
        return self.image.shape

    def clean_image(self):
        thresh, binary_mask_1 = cv2.threshold(self.image, 100, 255, cv2.THRESH_BINARY)
        thresh, binary_mask_2 = cv2.threshold(self.image, 170, 255, cv2.THRESH_BINARY_INV)
        binary_mask = binary_mask_1 & binary_mask_2
        self.image = self.image * binary_mask
