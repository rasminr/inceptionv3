import numpy as np
import cv2

cap = cv2.VideoCapture(0)


    # Capture frame-by-frame
ret, frame = cap.read()

cv2.imwrite("frame0.jpg",frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
