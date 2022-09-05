# Attendance_marking_versions


<h1>miniproject_0</h1>
intital steps

three step process  
capture  
train  
recognise  
test model with some changes according to our model requirement.

<h1>miniproject_1</h1>
Improvised the files and code

implemented them into a single runnable py file

<h1>miniproject_2</h1>
Integrated all module

<h1>miniproject_3</h1>
resolved some minor bugs for required modules and added comments

<h1>miniproject_4</h1>
Started integrating the xls 


<h1>Other info</h1>
  <h2>Face recognition module issue</h2>
   Have some typical difficulties with numpy list indexing and retriving details of the face .
   while using numpy list follow the below code in face recog module
   
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, _ = recognizer.predict(gray_image[y : y + h, x : x + w])
            if id:
                print(names,id)
                # cv2.imshow( img)
                cv2.putText(
                    img,
                    names[id-2],
                    (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
