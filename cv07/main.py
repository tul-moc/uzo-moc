import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(image, cmap, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def get_green_channel(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R = np.float32(image_rgb[:, :, 0])
    G = np.float32(image_rgb[:, :, 1])
    B = np.float32(image_rgb[:, :, 2])
    return 255 - ((G * 255) / (R + G + B))


def threshold_image(image):
    _, binary_image = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def custom_connected_components(binary_image):
    h, w = binary_image.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    coins = []
    
    for i in range(h):
        for j in range(w):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                stack = [(i, j)]
                labels[i, j] = current_label
                pixels = []
                while stack:
                    ci, cj = stack.pop()
                    pixels.append((ci, cj))
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if binary_image[ni, nj] == 255 and labels[ni, nj] == 0:
                                labels[ni, nj] = current_label
                                stack.append((ni, nj))
                area = len(pixels)
                centroid = (sum(j for i, j in pixels) / area,
                            sum(i for i, j in pixels) / area)
                coins.append({
                    'label': current_label,
                    'area': area,
                    'centroid': centroid,
                    'type': "5" if area > 4000 else "1"
                })
                current_label += 1
    return labels, coins


def draw_centroids(image, coins):
    centroids = image.copy()
    total_value = 0
    for coin in coins:
        x, y = map(int, coin['centroid'])
        cv2.circle(centroids, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(centroids, f"{coin['type']}", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        total_value += int(coin['type'])
    return centroids, total_value


def main():
    segmentation_image = cv2.imread("images/cv07_segmentace.bmp")

    green_channel = get_green_channel(segmentation_image)
    binary_image = threshold_image(green_channel)
    labels, coins = custom_connected_components(binary_image)
    centroids, coins_value = draw_centroids(segmentation_image, coins)

    segmentation_rgb = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB)
    centroids_rgb = cv2.cvtColor(centroids, cv2.COLOR_BGR2RGB)

    plot(segmentation_rgb, None, 'Originální obrázek')
    plot(green_channel, 'gray', 'Zelená složka')
    plot(binary_image, 'gray', 'Binární obraz')
    plot(labels, 'nipy_spectral', 'Označené oblasti')
    plot(centroids_rgb, None, 'Detekované mince')

    print(f"Hodnota mincí: {coins_value} CZK")


if __name__ == "__main__":
    plt.close("all")
    main()
