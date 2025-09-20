import cv2
import numpy as np
import tensorflow as tf

tflite_model_path = r" "
labels_path = r" "

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print("Loaded labels:", class_names)

cap = cv2.VideoCapture(1)  
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("Pill Detection + Classification", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pill Detection + Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if w > 30 and h > 30:
            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (128,128)).astype(np.float32)/255.0
            img_input = np.expand_dims(roi_resized, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_idx = np.argmax(output_data)
            pred_class = class_names[pred_idx]
            conf = np.max(output_data)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{pred_class} {conf*100:.1f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("Pill Detection + Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

