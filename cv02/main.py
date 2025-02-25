import cv2 as cv
import numpy as np

cap = cv.VideoCapture("./cv02_hrnecek.mp4")
model = cv.imread("./cv02_vzor_hrnecek.bmp")

model_hsv = cv.cvtColor(model, cv.COLOR_BGR2HSV)
roi_hist = cv.calcHist([model_hsv], [0], None, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)


def update_tracking_window(frame, track_window, max_iter=10, epsilon=1):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    back_proj = cv.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)
    
    x, y, w, h = track_window
    
    for _ in range(max_iter):
        window_back_proj = np.zeros_like(back_proj)
        window_back_proj[y:y+h, x:x+w] = back_proj[y:y+h, x:x+w]
        
        moment = cv.moments(window_back_proj)
        if moment["m00"] == 0:
            break
        
        moment_x = int(moment["m10"] / moment["m00"])
        moment_y = int(moment["m01"] / moment["m00"])
        
        new_x = max(0, min(moment_x - w // 2, frame.shape[1] - w))
        new_y = max(0, min(moment_y - h // 2, frame.shape[0] - h))
        
        if abs(new_x - x) < epsilon and abs(new_y - y) < epsilon:
            x, y = new_x, new_y
            break
        
        x, y = new_x, new_y

    return (x, y, w, h)

def get_default_track_window(frame):
    frame_height, frame_width = frame.shape[:2]
    model_height, model_width = model.shape[:2]
    init_x = frame_width // 2 - model_width // 2
    init_y = frame_height // 2 - model_height // 2
    return (init_x, init_y, model_width, model_height)

def main():
    ret, frame = cap.read()
    if not ret:
        return
    
    track_window = get_default_track_window(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        track_window = update_tracking_window(frame, track_window)
        x, y, w, h = track_window
        
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow("Image", frame)
        
        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
