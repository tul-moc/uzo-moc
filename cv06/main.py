import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_spectrum(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    fft2_image = np.fft.fft2(gray_image)
    fft_shift_image = np.fft.fftshift(fft2_image)
    return np.log(np.abs(fft_shift_image))

def plot(robot_gray, method_image, method_name):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(robot_gray, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title("Spectrum")
    plt.imshow(get_spectrum(robot_gray))
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title(method_name)
    plt.imshow(method_image)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Spectrum")
    plt.imshow(get_spectrum(method_image))
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def laplace_method(robot_gray):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplace_image = cv2.filter2D(robot_gray, cv2.CV_16S, kernel)
    plot(robot_gray, laplace_image, "Laplace")


def sobel_method(robot_gray):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = cv2.filter2D(robot_gray, cv2.CV_64F, kernel_x)
    gradient_y = cv2.filter2D(robot_gray, cv2.CV_64F, kernel_y)

    sobel_image = np.sqrt(gradient_x**2 + gradient_y**2)

    plot(robot_gray, sobel_image, "Sobel")


def kirsch_method(robot_gray):
    kernel_x = np.array([[-5, 3, 3], [-5, 0, 3], [-5, 3, 3]])
    kernel_y = np.array([[3, 3, 3],[3, 0, 3], [-5, -5, -5]])

    gradient_x = cv2.filter2D(robot_gray, cv2.CV_64F, kernel_x)
    gradient_y = cv2.filter2D(robot_gray, cv2.CV_64F, kernel_y)

    kirsch_image = np.sqrt(gradient_x**2 + gradient_y**2)

    plot(robot_gray, kirsch_image, "Kirsch")


def main():
    robot_image = cv2.imread("cv04c_robotC.bmp")
    robot_gray = cv2.cvtColor(robot_image, cv2.COLOR_BGR2GRAY)

    laplace_method(robot_gray)
    sobel_method(robot_gray)
    kirsch_method(robot_gray)

if __name__ == "__main__":
    plt.close("all")
    plt.rcParams["image.cmap"] = "jet"
    main()