import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import time

# Đọc mô hình đã lưu
gender_model = load_model('models/gender_model.h5')
age_model = load_model('models/age_model.h5')

# Hàm tiền xử lý ảnh
def preprocess_image(img):
    img = Image.fromarray(img)
    img = img.resize((128, 128), Image.LANCZOS)
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0
    return img

# Hàm xử lý ảnh từ tệp
def process_uploaded_image():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()  # Dừng camera khi tải ảnh

    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(128, 128))

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_processed = preprocess_image(face)

            gender_prediction = gender_model.predict(face_processed)[0][0]
            gender = 'Nam' if gender_prediction > 0.5 else 'Nữ'

            age_prediction = age_model.predict(face_processed)[0][0]
            age = int(age_prediction)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f'Gioi tinh: {gender}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f'Tuoi: {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

# Hàm xử lý video từ camera
def start_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()  # Dừng ảnh tải trước khi mở camera

    cap = cv2.VideoCapture(0)
    process_video()

def process_video():
    global last_age, last_gender, last_update
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể truy cập camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(128, 128))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_processed = preprocess_image(face)

            gender_prediction = gender_model.predict(face_processed)[0][0]
            gender = 'Nam' if gender_prediction > 0.5 else 'Nữ'

            age_prediction = age_model.predict(face_processed)[0][0]
            age = int(age_prediction)

            if time.time() - last_update > 1:
                if last_age is None or abs(age - last_age) > 2:
                    last_age = age
                    last_gender = gender
                    last_update = time.time()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Gioi tinh: {last_gender}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f'Tuoi: {last_age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        panel.configure(image=frame)
        panel.image = frame

        window.update_idletasks()
        window.update()

# Giao diện người dùng với tkinter
window = tk.Tk()
window.title("Dự Đoán Tuổi và Giới Tính AI")
window.geometry('800x600')
window.config(bg='#f0f0f0')

frame_buttons = tk.Frame(window, bg='#f0f0f0')
frame_buttons.pack(side=tk.TOP, pady=20)

button_upload = tk.Button(frame_buttons, text="Tải Ảnh", command=process_uploaded_image, bg='#4CAF50', fg='white', font=('Arial', 12), width=15)
button_upload.pack(side=tk.LEFT, padx=20)

button_camera = tk.Button(frame_buttons, text="Mở Camera", command=start_camera, bg='#2196F3', fg='white', font=('Arial', 12), width=15)
button_camera.pack(side=tk.LEFT, padx=20)

panel = tk.Label(window, bg='#f0f0f0')
panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Biến toàn cục
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = None
last_age = None
last_gender = None
last_update = time.time()

window.mainloop()

# Giải phóng camera và đóng cửa sổ khi kết thúc
if cap is not None and cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
