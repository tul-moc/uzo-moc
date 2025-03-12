import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(original_gray, average_gray, mask_gray, median_gray):
    images = {
        "Original": original_gray,
        "Average": average_gray,
        "Mask": mask_gray,
        "Median": median_gray,
    }

    _, axs = plt.subplots(4, 3, figsize=(12, 8))

    for i, (title, image) in enumerate(images.items()):
        axs[i, 0].imshow(image, cmap='gray')
        axs[i, 0].set_title(title)
        axs[i, 0].axis('off')

        spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))))
        spectrum_plot = axs[i, 1].imshow(spectrum, cmap='jet')
        axs[i, 1].set_title("Spectrum")
        axs[i, 1].axis('off')
        plt.colorbar(spectrum_plot, ax=axs[i, 1], fraction=0.046, pad=0.04)

    axs[3, 2].axis('off')  
    axs[2, 2].axis('off')
    axs[1, 2].axis('off')
    axs[0, 2].axis('off')
    plt.tight_layout()
    plt.show()


def basic_average_method(image):
    """
    a) odstraňte šum pomocí metody prostého průměrování 
    """
    kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(image, -1, kernel)


def increasing_mask_method(image):
    """
    b) odstraňte šum pomocí metody s rotující maskou
    """
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    masks = [
        np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1]]),
        np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]]),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]]),
        np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1]]),
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]]),
        np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]]),
        np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
        np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]])
    ]
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    
    for x in range(height):
        for y in range(width):
            min_variance = float('inf')
            best_mean = 0
            
            for mask in masks:
                neighborhood = padded_image[x:x+3, y:y+3]
                mask_values = neighborhood[mask == 1]
                variance = np.var(mask_values)
                if variance < min_variance:
                    min_variance = variance
                    best_mean = np.mean(mask_values)
            result[x, y] = best_mean
    
    return result.astype(np.uint8)


def median_method(image):
    """
    c) odstraňte šum pomocí mediánu
    """
    return cv2.medianBlur(image, 3)

def main():
    pss_image = cv2.imread("cv05_PSS.bmp")
    robot_image = cv2.imread("cv05_robotS.bmp")

    pss_gray = cv2.cvtColor(pss_image, cv2.COLOR_BGR2GRAY)
    robot_gray = cv2.cvtColor(robot_image, cv2.COLOR_BGR2GRAY)

    pss_avg = basic_average_method(pss_gray)
    pss_mask = increasing_mask_method(pss_gray)
    pss_median = median_method(pss_gray)

    robot_avg = basic_average_method(robot_gray)
    robot_mask = increasing_mask_method(robot_gray)
    robot_median = median_method(robot_gray)

    plot(robot_gray, robot_avg, robot_mask, robot_median)
    plot(pss_gray, pss_avg, pss_mask, pss_median)

if __name__ == "__main__":
    plt.close("all")
    plt.rcParams["image.cmap"] = "jet"
    main()