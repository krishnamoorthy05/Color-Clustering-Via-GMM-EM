import cv2
import numpy as np

name = "detectbuoy.avi"
cap = cv2.VideoCapture(name)

points = []

def get_image():
    image = cv2.imread('frame 3.6 sec.jpg')
    return image

def crop_contour(points):
    print("In crop contour")
    image = get_image()
    
    if len(points) < 3:
        print("Need at least 3 points to define a contour.")
        return
    
    new_list = np.array(points, dtype=np.int32)
    mask = np.zeros_like(image, dtype=np.uint8)
    
    cv2.drawContours(mask, [new_list], -1, (255, 255, 255), -1)
    mask = cv2.bitwise_not(mask)
    final = cv2.bitwise_and(image, mask)
       
    x, y, _ = np.where(final != 0)
    if len(x) == 0 or len(y) == 0:
        print("No non-zero pixels found. Please select points correctly.")
        return
    
    TL_x, TL_y = np.min(x), np.min(y)
    BR_x, BR_y = np.max(x), np.max(y)
    
    # Adjust crop boundaries to ensure the region fits within the image dimensions
    TL_x = max(TL_x - 20, 0)
    TL_y = max(TL_y - 20, 0)
    BR_x = min(BR_x + 20, image.shape[0])
    BR_y = min(BR_y + 20, image.shape[1])
    
    cropped = final[TL_x:BR_x, TL_y:BR_y]
    cv2.imwrite("cropped_buoy.jpg", cropped)
    cv2.imshow("Cropped", cropped)
    cv2.waitKey(0)

def click_and_crop(event, x, y, flag, params):
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))        
        print(points)
        
        if len(points) >= 2:
            cv2.line(image, points[-1], points[-2], (0, 0, 0), 1)
            cv2.imshow("name", image)
        if len(points) >= 3:
            cv2.polylines(image, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.imshow("name", image)
        if len(points) >= 3:
            crop_contour(points)

image = get_image()
cv2.imshow("name", image)
cv2.setMouseCallback("name", click_and_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
