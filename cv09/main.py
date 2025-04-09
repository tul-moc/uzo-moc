import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(histogram_original, histogram_top_hat, original, top_hat, centroids, rice_count):    
    plt.figure(figsize=(18, 12))

    plt.subplot(231)
    plt.title('hist. orig. image')
    plt.bar(range(256), histogram_original.ravel(), color='blue', width=0.5)
    plt.xlim([0, 256])
    plt.ylabel('#')
    
    plt.subplot(232)
    plt.title('hist. top-hat image')
    plt.bar(range(256), histogram_top_hat.ravel(), color='blue', width=0.5)
    plt.xlim([0, 256])
    plt.ylabel('#')
    
    plt.subplot(233)
    plt.title('seg. orig. image')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(234)
    plt.title('seg. top-hat image')
    plt.imshow(top_hat, cmap='gray')
    plt.axis('off')
    
    plt.subplot(235)
    plt.title('Number of rice grains: ' + str(rice_count))
    plt.imshow(cv2.cvtColor(centroids, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()


def top_hat(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def treshold_image(channel):
    _, binary = cv2.threshold(channel.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def color_regions(thresh):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    count = 0
    valid_centroids = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 90:
            count += 1
            valid_centroids.append(centroids[i])
    return count, valid_centroids


def draw_centroids(image, centroids):
    image_with_centroids = image.copy()
    for x, y in centroids:
        x, y = int(round(x)), int(round(y))
        cv2.drawMarker(image_with_centroids, (x, y), (0, 0, 255), 
                       markerType=cv2.MARKER_STAR, markerSize=8, 
                       thickness=1, line_type=cv2.LINE_AA)
    return image_with_centroids


def main(image_path):
    rice_image = cv2.imread(image_path)
    gray_rice = cv2.cvtColor(rice_image, cv2.COLOR_BGR2GRAY)
    top_hat_image = top_hat(gray_rice)

    histogram_original = cv2.calcHist([gray_rice], [0], None, [256], [0, 256])
    histogram_top_hat = cv2.calcHist([top_hat_image], [0], None, [256], [0, 256])

    rice_treshold = treshold_image(gray_rice)
    top_hat_treshold = treshold_image(top_hat_image)

    rice_count, rice_info = color_regions(top_hat_treshold)
    rice_centroids = draw_centroids(rice_image, rice_info)

    plot(histogram_original, histogram_top_hat, rice_treshold, top_hat_treshold, rice_centroids, rice_count)


if __name__ == "__main__":
    plt.close("all")
    main("images/cv09_rice.bmp")
