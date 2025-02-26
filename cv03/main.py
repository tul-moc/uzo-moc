import cv2
import numpy as np

image = cv2.imread("./cv03_robot.bmp")
h, w = image.shape[:2]
angle = 25

def rotate_image_manual():
    center_x = w / 2
    center_y = h / 2
    
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    new_w = int(abs(h * sin_angle) + abs(w * cos_angle))
    new_h = int(abs(h * cos_angle) + abs(w * sin_angle))
    
    new_center = (new_w // 2, new_h // 2)
    
    rotated = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    
    for y in range(new_h):
        for x in range(new_w):
            x_translated = x - new_center[0]
            y_translated = y - new_center[1]
            
            x_rotated = int(x_translated * cos_angle + y_translated * sin_angle + center_x)
            y_rotated = int(-x_translated * sin_angle + y_translated * cos_angle + center_y)
            
            if 0 <= x_rotated < w and 0 <= y_rotated < h:
                rotated[y, x] = image[round(y_rotated), round(x_rotated)]
    return rotated


if __name__ == "__main__":
    rotated = rotate_image_manual()
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
