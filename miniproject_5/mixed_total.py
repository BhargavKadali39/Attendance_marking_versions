import os
from pathlib import Path
from tkinter import *
from tkinter.ttk import *

import cv2
import numpy as np
from PIL import Image

#use python idle 3.9 only for me


# pip install tk opencv-python pathlib numpy Pillow
# pip install opencv-python opencv-contrib-python cv2operator cv

# Creates a instance for tkinter to work with
root=Tk()

# Window size
root.geometry('400x400')


#add all files as modules in here

def Generate_The_DataSet():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Start the video camera
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Get the userId and userName
    print("Follow the roll number order to avoid error\nEnter the id and name of the person: ")
    userId = input()
    userName = input()

    # Initially image Count is = 1
    count = 1

    # Function to save the image
    def saveImage(image, userName, userId, imgId):

        # Create a folder with the name as userName

        Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
        # Save the images inside the previously created folder
        cv2.imwrite(f"dataset/{userName}/{userId}_{imgId}.jpg",image)
        print(f"[INFO] Image {imgId} has been saved in folder : {userName}")


    print("[INFO] Video Capture is now starting please stay still...")

    while True:
        # Capture the frame/image
        _, img = vc.read()

        # Copy the original Image
        originalImg = img.copy()

        # Get the gray version of our image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the coordinates of the location of the face in the picture
        faces = faceCascade.detectMultiScale(gray_img,
                                            scaleFactor=1.2,
                                            minNeighbors=5,
                                            minSize=(50, 50))
        # Draw a rectangle at the location of the coordinates
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            coords = [x, y, w, h]

        # Show the image
        cv2.imshow("Identified Face - Press s to save and q to exit the window", img)

        # Wait for user keypress
        key = cv2.waitKey(1) & 0xFF

        # Check if the pressed key is 'k' or 'q'
        if key == ord('s') and count <= 10:
            roi_img = originalImg[coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]]
            saveImage(roi_img, userName, userId, count)
            count += 1
        elif key in [ord('s'), ord('q')]:
            break

    print(f"[INFO] Dataset has been created for {userName}")

    # Stop the video camera
    vc.release()
    # Close all Windows
    cv2.destroyAllWindows()




def Train_The_Model():
    path = []

    names = list(os.listdir("dataset"))
    # Get the path to all the images
    for name in names:
        for image in os.listdir(f"dataset/{name}"):
            path_string = os.path.join(f"dataset/{name}", image)
            path.append(path_string)


    faces = []
    ids = []

    # For each image create a numpy array and add it to faces list
    for img_path in path:
        image = Image.open(img_path).convert("L")

        imgNp = np.array(image, "uint8")


        id=int(img_path.split("/")[1].split("\\")[1].split("_")[0])
        # id = int(img_path.split("/")[2].split("_")[0])

        faces.append(imgNp)
        ids.append(id)

    # Convert the ids to numpy array and add it to ids list
    ids = np.array(ids)

    print("[INFO] Created faces and names Numpy Arrays")
    print("[INFO] Initializing the Classifier")

    # Make sure contrib is installed
    # The command is pip install opencv-contrib-python

    # Call the recognizer
    trainer = cv2.face.LBPHFaceRecognizer_create()
    # Give the faces and ids numpy arrays
    trainer.train(faces, ids)
    # Write the generated model to a yml file
    trainer.write("training.yml")

    print("[INFO] Training Done")




# dataset/{userName}/{userId}_{imgId}.jpg"

def Face_Recognition():  # sourcery skip: use-fstring-for-concatenation

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Call the trained model yml file to recognize faces
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("training.yml")



    # Names corresponding to each id
    names = list(os.listdir("dataset"))
    imgids= []
    # list_names=names.tolist()
    for i in range(len(names)):
        # print(names)
        p=list(os.listdir(f"dataset/{names[i]}"))
        imgids.append(p[i].split("_")[0])
    hoo = {imgids[i]: names[i]  for i in range(len(names))}
    #-----------------------
    while True:
        # true_count=0
        # true_val=""

        _, img = video_capture.read()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        # Predicting the face and get the id
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, _ = recognizer.predict(gray_image[y : y + h, x : x + w]) #use any variable instead of _ for confidence
            # id
            # print(hoo)
            # print(id)
            if id:    
                cv2.putText(
                    img,
                    hoo[str(id)],
                    (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # if(true_count>20):
                #     print(f"marked present for {hoo[str(id)]}")
                #     true_count=0
                #     break
                # hoo[str(id)]=hoo[str(id)]
                # if(true_val==hoo[str(id)]):
                #     true_count+=1
                # else:
                #     true_val=hoo[str(id)]
                #     true_count=0


                # if(true_val==id):
                #     true_count+=1
                # else:
                #     true_val = hoo[str(id)]


            else:
                cv2.putText(
                    img,
                    "Unknown",
                    (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("recoginising - Press q to exit the window", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_capture.release()
    cv2.destroyAllWindows()

# def Attendace_register():
#     with open("D:\\miniproj images\\studentData.csv")  as inputFile:
#         csvReader = csv.reader(inputFile, dialect='excel')
#         for row,cols in csvReader:
#             print(row)
#             print(cols)

# Window styles
root['background']='#80D0C7'
style = Style()
style.configure('TButton', font =('calibri', 20, 'bold'),borderwidth = '4')

# Buttons which are displayed

btn1=Button(root, text="Generate Dataset", command=Generate_The_DataSet ,cursor="gobbler")
btn1.grid(row = 3, column = 3, pady = 40, padx = 20) 

btn2=Button(root, text="Train Model", command=Train_The_Model,cursor="gobbler")
btn2.grid(row = 4, column = 3, pady = 30, padx = 100)

btn3=Button(root, text="Face Recognition", command=Face_Recognition ,cursor="gobbler")
btn3.grid(row = 5, column = 3, pady = 30, padx = 100)

# Mainloop to stitch all the tkinter components into a single component and run it using gui

# qrval=""
root.mainloop()
cv2.destroyAllWindows()

