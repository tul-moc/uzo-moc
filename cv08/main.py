import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(original, segmentation, filtered, result, segmentation_cmap="gray", filtered_cmap="gray"):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(segmentation, cmap=segmentation_cmap)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(filtered, cmap=filtered_cmap)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def image_threshold(image):
    _, binary = cv2.threshold(image.astype(np.uint8), 104, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def open_method(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    filtered = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    filtered = (filtered > 0).astype(np.uint8)
    return filtered


def close_method(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    filtered = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    filtered = (filtered > 0).astype(np.uint8)
    return filtered


def calculate_red_channel(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R = np.float32(img_rgb[:, :, 0])
    G = np.float32(img_rgb[:, :, 1])
    B = np.float32(img_rgb[:, :, 2])
    denom = (R + G + B) + 1e-6
    red_channel = 255 - ((R * 255) / denom)
    inverted_red_channel = cv2.bitwise_not(red_channel.astype(np.uint8))
    return inverted_red_channel


def color_regions(binary):
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    infos = []
    for i in range(1, num_labels):
        infos.append(centroids[i])
    return infos


def draw_centroids(image, infos):
    centroids = image.copy()
    for info in infos:
        x, y = map(int, info)
        cv2.drawMarker(centroids, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)
    return centroids


def process_first_image(image_path):
    segmentation = cv2.imread(image_path)
    gray = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)

    binary = image_threshold(inverted)
    binary_filtered = open_method(binary)
    infos = color_regions(binary_filtered)
    centroids = draw_centroids(segmentation, infos)
    centroids_rgb = cv2.cvtColor(centroids, cv2.COLOR_BGR2RGB)

    plot(segmentation, binary, binary_filtered, centroids_rgb, segmentation_cmap="gray")

def process_second_image(image_path):
    segmentation = cv2.imread(image_path)

    channel = calculate_red_channel(segmentation)
    binary = image_threshold(channel)
    binary_filtered = close_method(binary)
    infos = color_regions(binary_filtered)
    centroids = draw_centroids(segmentation, infos)

    rgb_centroids = cv2.cvtColor(centroids, cv2.COLOR_BGR2RGB)

    plot(segmentation, binary_filtered, channel, rgb_centroids, segmentation_cmap="gray", filtered_cmap="jet")

if __name__ == "__main__":
    plt.close("all")
    process_first_image("images/cv08_im1.bmp")
    process_second_image("images/cv08_im2.bmp")
