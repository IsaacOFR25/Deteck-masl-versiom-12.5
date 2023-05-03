#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#version 2.0 con interfaz de ingreso
import cv2
import os
import mediapipe as mp
from playsound import playsound
import tkinter as tk

mp_face_detection = mp.solutions.face_detection
LABELS = [" Con_mascarilla", "Sin_mascarilla"]

# Leer el modeto
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

def detect_mask():
    global status_label
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if ret == False: break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections is not None:
                for detection in results.detections:
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin= int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width  * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)

                    if xmin< 0 and ymin < 0:
                        continue

                    face_image = frame[ymin : ymin  + h, xmin : xmin +w]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    result = face_mask.predict(face_image)
                    if result[1] < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[result[0]] == "Sin_mascarilla"else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin-25), 2, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[result[0]])

            cv2.imshow("Detector de mascarilla", frame)
            k = (cv2.waitKey(1))
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#codigo optimizado 3.0
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

def detect_mask():
    global status_label
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width  * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0:
                        continue

                    face_image = frame[ymin:ymin+h, xmin:xmin+w]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        color = (0, 0, 255) if label == 1 else (0, 255, 0)
                        cv2.putText(frame, LABELS[label], (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                        status_label.config(text=LABELS[label])

            cv2.imshow("Detector de mascarilla", frame)
            k = cv2.waitKey(1)
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#codigo optimizado 4.0
import cv2
import os
import mediapipe as mp
from playsound import playsound
import tkinter as tk

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

def detect_mask():
    global status_label
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 and ymin < 0:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label])

            cv2.imshow("Detector de mascarilla", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#codigo optimizado 5.0
import cv2
import os
import mediapipe as mp
from playsound import playsound
import tkinter as tk

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

# Agregar el texto en un Label
author_label = tk.Label(root, text="©Isaac Francisco Ortega Romero©", font=("Helvetica", 12))
author_label.pack(side="bottom")

def detect_mask():
    global status_label
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 and ymin < 0:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label])
                        
            cv2.namedWindow("Detector de mascarilla", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Detector de mascarilla", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Detector de mascarilla", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#version 6.0
import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

progress = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate")
progress.pack()

def detect_mask():
    global status_label, progress
    status_label.config(text="Iniciando detección...")
    progress["value"] = 0
    progress["maximum"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 and ymin < 0:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label])

            cv2.imshow("Detector de mascarilla", frame)
            progress["value"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#Version 7.0 que parte de la 4, es mas estable
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

def detect_mask():
    global status_label
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label])

            cv2.imshow("Detector de mascarilla", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#Version 8.0
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

def detect_mask():
    global status_label
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label])

            cv2.imshow("Detector de mascarilla", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


#version 9.0 
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label])

            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.configure(bg='#f2f2f2')

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16), bg='#f2f2f2')
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12), bg='#f2f2f2')
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12), bg='#f2f2f2')
status_label.pack()

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2,)
                        status_label.config(text=LABELS[label], fg='#ff0000' if LABELS[label] == "Sin_mascarilla" else '#00ff00')

            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Helvetica", 12))
button1.pack(pady=10)

root.mainloop()


# In[ ]:


import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16))
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12))
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12))
status_label.pack()

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...")
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin_mascarilla" else '#008000')

            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.pack(pady=10)


root.mainloop()


# In[ ]:


#version 10.0
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.configure(bg='#f2f2f2')

label1 = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 16), bg='#f2f2f2')
label1.pack()

label2 = tk.Label(root, text="Estado: ", font=("Helvetica", 12), bg='#f2f2f2')
label2.pack()

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 12), bg='#f2f2f2')
status_label.pack()

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin_mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin_mascarilla" else '#008000')

            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.pack(pady=10)


root.mainloop()


# In[ ]:


#Version 11.0
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x620")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Helvetica", 20, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Helvetica", 16), bg='#f2f2f2')
status_label.pack(pady=10)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.pack(pady=10)


root.mainloop()


# In[ ]:


#version 12.0 creo que es la definitiva :)
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x680")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 22, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [Isaac Francisco Ortega Romero]")
        
        
button1 = tk.Button(root, text="Iniciar Detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.pack(pady=10, side=tk.TOP)

button2 = tk.Button(root, text="Acerca de", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=about)
button2.pack(pady=10, side=tk.TOP)


root.mainloop()


# In[ ]:


#version 12.05 creo que es la definitiva :)
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x700")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 26, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [Isaac Francisco Ortega Romero]")

def pausa():        
         messagebox.showinfo("Pausa en la detección","Haz pausado al detección, para reanudar solo da click en aceptar")
        
button_frame = tk.Frame(root, bg='#f2f2f2')
button_frame.pack(pady=10)

button1 = tk.Button(button_frame, text="Iniciar detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.grid(row=0, column=0, padx=10, columnspan=2)

button3 = tk.Button(button_frame, text="Pausar detección", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=pausa)
button3.grid(row=0, column=2, padx=10, columnspan=2)

button2 = tk.Button(button_frame, text="Acerca de", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=about)
button2.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")


root.mainloop()


# In[ ]:


#version 12.05 creo que es la definitiva :)
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x700")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 26, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [equipo]")

def pausa():        
         messagebox.showinfo("Pausa en la detección","Haz pausado al detección, para reanudar solo da click en aceptar")
        
button_frame = tk.Frame(root, bg='#f2f2f2')
button_frame.pack(pady=10)

button1 = tk.Button(button_frame, text="Iniciar detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.grid(row=0, column=0, padx=10, columnspan=2)

button3 = tk.Button(button_frame, text="Pausar detección", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=pausa)
button3.grid(row=0, column=2, padx=10, columnspan=2)

button2 = tk.Button(button_frame, text="Acerca de", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=about)
button2.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")


root.mainloop()


# In[ ]:


#Version 13.0
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x750")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 26, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_face():
    global status_label, canvas
    status_label.config(text="Iniciando detección facial...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 0, 0), 3)
                    
            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [Isaac Francisco Ortega Romero]")

def pausa():        
         messagebox.showinfo("Pausa en la detección","Haz pausado al detección, para reanudar solo da click en aceptar")
        
button_frame = tk.Frame(root, bg='#f2f2f2')
button_frame.pack(pady=10)

button1 = tk.Button(button_frame, text="Iniciar detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.grid(row=0, column=0, padx=10, columnspan=2)

button2 = tk.Button(button_frame, text="Acerca de", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=about)
button2.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

button3 = tk.Button(button_frame, text="Pausar detección", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=pausa)
button3.grid(row=0, column=2, padx=10, columnspan=2)

button4 = tk.Button(button_frame, text="Iniciar", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=detect_face)
button4.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

root.mainloop()


# In[ ]:


#Version 13.5
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x750")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 26, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_face():
    global status_label, canvas
    status_label.config(text="Iniciando detección facial", fg='#2271b3')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 255, 0), 3)
                    
            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección...", fg='#ff6600')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        #playsound('C:/Users/luke2/Downloads/alarma.mp3')
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [Isaac Francisco Ortega Romero]")

def pausa():        
         messagebox.showinfo("Pausa en la detección","Haz pausado al detección, para reanudar solo da click en aceptar")
        
button_frame = tk.Frame(root, bg='#f2f2f2')
button_frame.pack(pady=10)

button1 = tk.Button(button_frame, text="Iniciar detección", command=detect_mask, font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF')
button1.grid(row=0, column=0, padx=10, columnspan=2)

button2 = tk.Button(button_frame, text="Acerca de", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=about)
button2.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

button3 = tk.Button(button_frame, text="Pausar detección", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=pausa)
button3.grid(row=0, column=2, padx=10, columnspan=2)

button4 = tk.Button(button_frame, text="Iniciar", font=("Arial", 14, "bold"), bg='#6495ED', fg='#FFFFFF',command=detect_face)
button4.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

root.mainloop()


# In[2]:


#Version 13.0
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x700")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 26, "bold"), bg='#6495ED', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_face():
    global status_label, canvas
    status_label.config(text="Iniciando detección facial", fg='#2271b3')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 255, 0), 3)
                    
            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección", fg='#E74C3C')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [Isaac Francisco Ortega Romero]")

def pausa():        
         messagebox.showinfo("Pausa en la detección","Haz pausado al detección, para reanudar solo da click en aceptar")

#Crear un contenedor Frame
button_frame = tk.Frame(root, bg='#f2f2f2')
button_frame.pack(side=tk.TOP)

# Crear los botones y agregarlos al contenedor
detect_face_button = tk.Button(button_frame, text="Detectar rostro", font=("Arial", 12), command=detect_face, width=5, height=2)
detect_face_button.pack(side=tk.LEFT, padx=10, pady=10)

detect_mask_button = tk.Button(button_frame, text="Detectar mascarilla", font=("Arial", 12), command=detect_mask, width=5, height=2)
detect_mask_button.pack(side=tk.LEFT, padx=10, pady=10)

pause = tk.Button(button_frame, text="Pausar detección", font=("Arial", 12), command=pausa, width=5, height=2)
pause.pack(side=tk.LEFT, padx=10, pady=10)

about_button = tk.Button(button_frame, text="Acerca de", font=("Arial", 12), command=about, width=5, height=2)
about_button.pack(side=tk.LEFT, padx=10, pady=10)

root.mainloop()


# In[5]:


#Version 14.0
import os
import cv2
import mediapipe as mp
from playsound import playsound
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import messagebox

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con mascarilla", "Sin mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Detector de Mascarilla")
root.geometry("680x680")
root.resizable(False, False)
root.configure(bg='#f2f2f2')

title_label = tk.Label(root, text="Detector de Mascarilla", font=("Arial", 26, "bold"), bg='#2596be', fg='#FFFFFF')
title_label.pack(fill='x')

status_label = tk.Label(root, text="Desconocido", font=("Arial", 16), bg='#f2f2f2')
status_label.pack(pady=10,side=tk.TOP)

# Crear un canvas donde se mostrará el frame de la cámara
canvas = tk.Canvas(root, width=640, height=480, bg='#000000')
canvas.pack()

def detect_face():
    global status_label, canvas
    status_label.config(text="Iniciando detección facial", fg='#2271b3')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), (255, 255, 0), 3)
                    
            # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def detect_mask():
    global status_label, canvas
    status_label.config(text="Iniciando detección", fg='#E74C3C')
    with mp_face_detection.FaceDetection(
        min_detection_confidence = 0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    xmin = int(bbox.xmin * width)
                    ymin = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)

                    if xmin < 0 or ymin < 0 or w < 1 or h < 1:
                        continue

                    face_image = cv2.cvtColor(frame[ymin : ymin  + h, xmin : xmin +w], cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation = cv2.INTER_CUBIC)

                    label, confidence = face_mask.predict(face_image)
                    if confidence < 150:
                        color = (0,0,255) if LABELS[label] == "Sin mascarilla" else (0,255,0)
                        cv2.putText(frame, "{}".format(LABELS[label]), (xmin, ymin-25), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 3)
                        status_label.config(text=LABELS[label], fg='#B22222' if LABELS[label] == "Sin mascarilla" else '#008000')

           # Mostrar el frame en el canvas
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = ImageTk.PhotoImage(image=Image.fromarray(img))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.img = img
            root.update()

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def about():
    messagebox.showinfo("Acerca de", "Este programa utiliza la biblioteca OpenCV y la biblioteca de mediapipe para detectar rostros y detectar si el usuario está usando una mascarilla o no. El programa muestra la cámara en vivo y utiliza un modelo de aprendizaje automático previamente entrenado para determinar si se está usando una mascarilla. \n\nSi el usuario no tiene una mascarilla, se muestra una etiqueta ""Sin mascarilla"" en el video en vivo y se activa una alarma de sonido. \nEste programa se puede utilizar para ayudar a garantizar el cumplimiento de los requisitos de uso de mascarillas en lugares públicos.\n\nDesarrollado por: [Isaac Francisco Ortega Romero]")

def pausa():        
         messagebox.showinfo("Pausa en la detección","Haz pausado al detección, para reanudar solo da click en aceptar")

# Crear un contenedor Frame
button_frame = tk.Frame(root, bg='#f2f2f2')
button_frame.pack(side=tk.TOP)

detect_face_button = tk.Button(button_frame, text="Detectar rostro", font=("Arial", 10, "bold"), command=detect_face, bg="#C4DEF6", fg="#008CBA", width=15, height=2)
detect_face_button.pack(side=tk.LEFT, padx=10, pady=10)

detect_mask_button = tk.Button(button_frame, text="Detectar mascarilla", font=("Arial", 10, "bold"), command=detect_mask, bg="#BEE7D2", fg="#4CAF50", width=15, height=2)
detect_mask_button.pack(side=tk.LEFT, padx=10, pady=10)

pause = tk.Button(button_frame, text="Pausar detección", font=("Arial", 10, "bold"), command=pausa, bg="#F7D7DA", fg="#f44336", width=15, height=2)
pause.pack(side=tk.LEFT, padx=10, pady=10)

about_button = tk.Button(button_frame, text="Acerca de", font=("Arial", 10, "bold"), command=about, bg="#B19CD9", fg="white", width=15, height=2)
about_button.pack(side=tk.LEFT, padx=10, pady=10)


root.mainloop()


# In[ ]:




