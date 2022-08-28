from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
import pyzbar.pyzbar as pyzbar
from tkinter.ttk import *
import openpyxl



def button_hand():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
          success, image = cap.read()
          if not success:
            continue
          image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
          image.flags.writeable = False
          results = hands.process(image)
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                  image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          cv2.imshow('Hands Detection - Press ESC to exit the window', image)
          if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    cap.release()


def button_face():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing_face = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
          for detection in results.detections:
             mp_drawing_face.draw_detection(image, detection)
        cv2.imshow('Face Detection - Press ESC to exit the window', image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break


def button_qrcode():
    flag=0
    pres={}
    cam = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()
    print(type(detector))
    print("above is detecetor")
    while True:
        _, img = cam.read()
        decodedObjects = pyzbar.decode(img)
        for obj in decodedObjects:
            if flag>0:
                if temp==obj.data:
                    break
            print(obj.data[2:-1])
            pres[obj.data] = "present"
            flag=flag+1
            temp=obj.data
        cv2.imshow("Scan the QR Code - Press ESC to exit the window", img)
        if cv2.waitKey(1) & 0xFF == 27:
            print("\n")
            for (i, j) in pres.items():
                print ("{0} : {1} \n".format(i, j))
            cam.release()
            
            cv2.destroyAllWindows()
            break
            
root=Tk()

root.geometry('350x350')

root.bind('<Escape>', lambda e: root.destroy())            
root['background']='#80D0C7'
style = Style()


style.configure('TButton', font =('calibri', 20, 'bold'),borderwidth = '4')


btn1=Button(root, text="Hand detection", command=button_hand ,cursor="gobbler")
btn1.grid(row = 3, column = 3, pady = 30, padx = 10)

btn2=Button(root, text="QR Code", command=button_qrcode,cursor="gobbler")
btn2.grid(row = 4, column = 3, pady = 30, padx = 100)

btn3=Button(root, text="Face detection", command=button_face ,cursor="gobbler")
btn3.grid(row = 5, column = 3, pady = 30, padx = 100)


root.mainloop()
print("Successfully exited")
