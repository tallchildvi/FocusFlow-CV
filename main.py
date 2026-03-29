import cv2

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: no frame")
        break
    
    cv2.imshow("AI Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()