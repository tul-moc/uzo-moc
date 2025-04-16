import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_images(unknown_img, recognized_img):
    plt.figure(figsize=(8, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(unknown_img)
    plt.title('Unknown Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(recognized_img)
    plt.title('Similar Image')
    plt.axis('off')

    plt.show()


def load_sample_images(path="images/sample"):
    image_files = glob.glob(os.path.join(path, "*.bmp"))
    
    images_gray = []
    images_color = []
    
    for file in image_files:
        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        images_color.append(img)
        images_gray.append(img_gray)
    
    return images_color, images_gray


def flatten_images(images):
    flattened = [img.flatten() for img in images]
    return np.array(flattened).T


def project_image(img_vector, mean_img, eigenvectors):
    return np.dot(eigenvectors.T, (img_vector - mean_img).flatten())


def compute_pca(data_matrix):
    mean_img = np.mean(data_matrix, axis=1, keepdims=True)

    A = data_matrix - mean_img
    L = np.dot(A.T, A)

    _, eigenvectors_small = np.linalg.eig(L)    
    eigenvectors = np.dot(A, eigenvectors_small)
    
    return mean_img, eigenvectors


def prepare_gallery(sample_path="images/sample"):
    images_color, images_gray = load_sample_images(sample_path)
    X = flatten_images(images_gray)
    mean_img, eigenvectors = compute_pca(X)
    
    projections = []
    for i in range(X.shape[1]):
        proj = np.dot(eigenvectors.T, (X[:, i].reshape(-1, 1) - mean_img))
        projections.append(proj.flatten())
    projections = np.array(projections)
    
    return images_color, mean_img, eigenvectors, projections


def recognize_unknown(unknown_image_path, mean_img, eigenvectors, projections):
    unknown_img = cv2.imread(unknown_image_path)
    unknown_gray = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
    unknown_vector = unknown_gray.flatten().reshape(-1, 1)
    
    unknown_proj = project_image(unknown_vector, mean_img, eigenvectors)
    
    min_dist = np.inf
    best_match_index = -1
    for i, proj in enumerate(projections):
        dist = np.linalg.norm(proj - unknown_proj)
        if dist < min_dist:
            min_dist = dist
            best_match_index = i

    return unknown_img, best_match_index


def main(unknown_image_path, sample_path="images/sample"):
    images_color, mean_img, eigenvectors, projections = prepare_gallery(sample_path)
    
    unknown_img, best_match_index = recognize_unknown(unknown_image_path, mean_img, eigenvectors, projections)
    
    best_match_img = images_color[best_match_index]

    unknown_rgb = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2RGB)
    recognized_rgb = cv2.cvtColor(best_match_img, cv2.COLOR_BGR2RGB)
    plot_images(unknown_rgb, recognized_rgb)


if __name__ == "__main__":
    plt.close("all")
    main("images/unknown.bmp")
