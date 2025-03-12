import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

input_images = ['im01.jpg', 'im02.jpg', 'im03.jpg', 'im04.jpg', 'im05.jpg', 'im06.jpg', 'im07.jpg', 'im08.jpg', 'im09.jpg']
path = 'images/'

def load_images():
    result = []
    for image in input_images:
      result.append(cv2.imread(path + image))
    return result

def convert_to_gray(images):
    gray_images = []
    for image in images:
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return gray_images

def convert_to_dcp(gray_images):
    dcp_images = []
    for gray_image in gray_images:
        dcp_images.append(dctn(gray_image))
    return dcp_images

def limit_dct(dcts, limit):
    limited_dcts = np.zeros_like(dcts)
    limited_dcts[:limit, :limit] = dcts[:limit, :limit]
    return limited_dcts

def convert_filtr(filtr, gray_image, fft2_shift):
    gray = cv2.cvtColor(filtr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (gray_image.shape[1], gray_image.shape[0]))
    return fft2_shift * resized

def get_convolution(conv):
    shifted = np.fft.ifftshift(conv)
    abs =  np.abs(np.fft.ifft2(shifted))
    return abs / np.max(abs)

def plot_cv01(fft2, fft2_shift):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("FFT2")
    plt.imshow(np.log(np.abs(fft2)))

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(fft2_shift)))

    plt.show()

def plot_cv02(fft2, fft2_shift, result1, result2):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("FFT2")
    plt.imshow(np.log(np.abs(fft2)))

    plt.subplot(2, 2, 3)
    plt.imshow(np.log(np.abs(fft2_shift)))

    plt.subplot(2, 2, 2)
    plt.title("Results")
    plt.imshow(result1, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.imshow(result2, cmap='gray')

    plt.show()

def plot_cv03(gray, dcts):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Gray")
    plt.imshow(gray, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("DctS")
    plt.imshow(np.log(np.abs(dcts)))

    plt.show()

def plot_cv04(gray_10, dctn_10, gray_30, dctn_30, gray_50, dctn_50):
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 2, 1)
    plt.title("Gray")
    plt.imshow(gray_10, cmap='gray')

    plt.subplot(3, 2, 2)
    plt.title("DctS")
    plt.imshow(np.log(np.abs(dctn_10)))

    plt.subplot(3, 2, 3)
    plt.imshow(gray_30, cmap='gray')

    plt.subplot(3, 2, 4)
    plt.imshow(np.log(np.abs(dctn_30)))

    plt.subplot(3, 2, 5)
    plt.imshow(gray_50, cmap='gray')

    plt.subplot(3, 2, 6)
    plt.imshow(np.log(np.abs(dctn_50)))

    plt.show()

def filt_dp_hp(gray, fft2_shifted):
    gray_dp1 = cv2.imread("cv04c_filtDP.bmp")
    gray_dp2 = cv2.imread("cv04c_filtDP1.bmp")
    gray_hp1 = cv2.imread("cv04c_filtHP.bmp")
    gray_hp2 = cv2.imread("cv04c_filtHP1.bmp")

    conv_dp1 = convert_filtr(gray_dp1, gray, fft2_shifted)
    conv_dp2 = convert_filtr(gray_dp2, gray, fft2_shifted)
    conv_hp1 = convert_filtr(gray_hp1, gray, fft2_shifted)
    conv_hp2 = convert_filtr(gray_hp2, gray, fft2_shifted)

    im_dp1 = get_convolution(conv_dp1)
    im_dp2 = get_convolution(conv_dp2)
    im_hp1 = get_convolution(conv_hp1)
    im_hp2 = get_convolution(conv_hp2)

    return im_dp1, im_dp2, im_hp1, im_hp2, conv_dp1, conv_dp2, conv_hp1, conv_hp2

def cv01_cv04():
    image = cv2.imread("cv04c_robotC.bmp")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft2 = np.fft.fft2(gray)
    fft2_shifted = np.fft.fftshift(fft2)
    
    plot_cv01(fft2, fft2_shifted)

    im_dp1, im_dp2, im_hp1, im_hp2, conv_dp1, conv_dp2, conv_hp1, conv_hp2 = filt_dp_hp(gray, fft2_shifted)

    plot_cv02(conv_dp1, conv_dp2, im_dp1, im_dp2)
    plot_cv02(conv_hp1, conv_hp2, im_hp1, im_hp2)

    dcts = dctn(gray)

    plot_cv03(gray, dcts)

    dctn_10 = limit_dct(dcts, 10)
    dctn_30 = limit_dct(dcts, 30)
    dctn_50 = limit_dct(dcts, 50)

    gray_10 = idctn(dctn_10)
    gray_30 = idctn(dctn_30)
    gray_50 = idctn(dctn_50)
    plot_cv04(gray_10, dctn_10, gray_30, dctn_30, gray_50, dctn_50)

def cv05():
    images = load_images()
    gray_images = convert_to_gray(images)
    dcp_images = convert_to_dcp(gray_images)

    limited_dcts = [limit_dct(dcp, 5) for dcp in dcp_images]
    limited_imgs = [idctn(dct) for dct in limited_dcts]
    image_differences = []
    for img in limited_imgs:
        differences = []
        for img2 in limited_imgs:
            img = cv2.resize(img, (250, 250))
            img2 = cv2.resize(img2, (250, 250))
            difference = np.sum(np.abs(img - img2))
            differences.append(difference)
        image_differences.append(differences)

    sorted_images = []
    sorted_differences = []
    for i, diffs in enumerate(image_differences):
        sorted_indices = np.argsort(diffs)
        sorted_images.append([images[idx] for idx in sorted_indices])
        sorted_differences.append([diffs[idx] for idx in sorted_indices])

    _, axes = plt.subplots(9, 9, figsize=(15, 15))
    for i, row in enumerate(sorted_images):
        for j, img in enumerate(row):
            axes[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #axes[i, j].text(10, 20, f"{sorted_differences[i][j]:.2f}", color="black")
            axes[i, j].axis('off')
    plt.show()

if __name__ == "__main__":
    plt.close("all")
    plt.rcParams["image.cmap"] = "jet"
    cv01_cv04()
    cv05()