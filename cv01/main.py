import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = ['im01.jpg', 'im02.jpg', 'im03.jpg', 'im04.jpg', 'im05.jpg', 'im06.jpg', 'im07.jpg', 'im08.jpg', 'im09.jpg']
path = 'img/'

def create_grayscale_histograms(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    return histogram

def compare_image_distance(input_path):
    input_image = cv2.imread(input_path)
    input_histogram = create_grayscale_histograms(input_image)

    compared_distances = []
    for image_name in images:
        imread_image = cv2.imread(path + image_name)
        img_histogram = create_grayscale_histograms(imread_image)
        compared_distance = cv2.compareHist(input_histogram, img_histogram, cv2.HISTCMP_BHATTACHARYYA)
        compared_distances.append((image_name, compared_distance))

    sorted_distances = sorted(compared_distances, key=lambda x: x[1])
    return sorted_distances
        
if __name__ == '__main__':
    result = []
    for image_name in images:
        result.append((image_name, compare_image_distance(path + image_name)))

    n = len(images)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(3*n, 3*n))

    for row_idx, (input_img_name, sorted_imgs) in enumerate(result):
        for col_idx, (filename, distance) in enumerate(sorted_imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(mpimg.imread(path + filename), cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()