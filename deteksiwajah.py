#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2

# Memuat file Haar cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk mendeteksi wajah dalam frame
def detect_faces_from_webcam():
    # Membuka webcam (biasanya device index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam.")
        return

    while True:
        # Membaca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari webcam.")
            break
        
        # Mengubah frame ke skala abu-abu
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah dalam frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Menandai wajah yang terdeteksi dengan persegi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Menampilkan frame dengan wajah yang terdeteksi
        cv2.imshow('Detected Faces', frame)
        
        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Melepaskan webcam dan menutup semua jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

# Memanggil fungsi untuk mendeteksi wajah dari webcam
detect_faces_from_webcam()


# In[ ]:




