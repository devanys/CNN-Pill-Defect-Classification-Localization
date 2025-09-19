# =================== Realtime Pill Detection + Classification (Fullscreen) ===================
import cv2
import numpy as np
import tensorflow as tf

# ---------------- Load TFLite Model ----------------
tflite_model_path = r"C:\Users\Devan\Downloads\Zuhair\cnn_mobilenetv2_retrain2.tflite"
labels_path = r"C:\Users\Devan\Downloads\Zuhair\labes.txt"

# Load interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print("Loaded labels:", class_names)

# ---------------- Open Camera ----------------
cap = cv2.VideoCapture(1)  # ganti index kamera sesuai kebutuhan
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Buat jendela fullscreen
cv2.namedWindow("Pill Detection + Classification", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pill Detection + Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- Preprocessing untuk deteksi pil sederhana ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Cari kontur (calon pil)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # filter ukuran biar tidak kebaca noise kecil
        if w > 30 and h > 30:
            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (128,128)).astype(np.float32)/255.0
            img_input = np.expand_dims(roi_resized, axis=0)

            # prediksi dengan model
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_idx = np.argmax(output_data)
            pred_class = class_names[pred_idx]
            conf = np.max(output_data)

            # gambar bounding box + label
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{pred_class} {conf*100:.1f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Tampilkan fullscreen
    cv2.imshow("Pill Detection + Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
