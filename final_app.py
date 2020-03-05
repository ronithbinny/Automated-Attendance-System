import pandas as pd
import numpy as np
import datetime
import cv2
import os
import face_recognition

encodings = []
names = []
name = []
    
def add_student() :
    # Student Details :
    student_name = input("Enter Students Full Name : ").strip().title()
    student_number = input("Enter Student Number : ").strip()
    
    # Make Folder :
    parent_dir = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Attendance\Data Base"
    path = os.path.join(parent_dir,student_name)
    os.mkdir(path)
    
    # Turn On Cam :
    cap = cv2.VideoCapture(0)
    
    print(" Please Turn Your Face around in different directions : ")
    
    x = 0
    i = 0
    while(True) :
        # Capture Frame :
        ret,frame = cap.read()
        # Converting to Grey Scale :
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display :
        cv2.imshow("Frame",grey)
        cv2.waitKey(1)
        # Saving Img :
        if i in range(0,500,5) :
            filename = "{}\{}-{}-{}.jpg".format(path,student_name,student_number,i)
            cv2.imwrite(filename,grey)
            print("Please Turn Your Face around in different directions : Snap - {}".format(x + 1))
            x = x + 1
        i = i + 1
        # Closing Loop :
        if i == 1001 :
            break
        
    # Release the capture :
    cap.release()
    cv2.destroyAllWindows()
    
    print("Thank You! Your Photos are stored at {}".format(path))
    
    
def training() :  
    # Training directory
    path = r'C:\Users\Ronith\Desktop\Data Science 2020\Projects\Attendance\Data Base/'
    
    train_dir = os.listdir(path)
    
    for persons in train_dir:
        pix = os.listdir( path + persons)
        
        for person_img in pix :
            face = face_recognition.load_image_file(path + persons + "/" + person_img)
            
            face_bounding_boxes = face_recognition.face_locations(face)
            
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(persons)
            else:
                print(persons + "/" + person_img + " was skipped and can't be used for training")
    
def attendance() :
    # Turn On Cam :
    cap = cv2.VideoCapture(0)
    
    ret,frame = cap.read()
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Release the capture :
    cap.release()
    cv2.destroyAllWindows()
    
    # Date :
    x = datetime.datetime.now()
    date = x.strftime("%d")
    month = x.strftime("%b")
    year = x.strftime("%Y")
    hour = x.strftime("%H")
    minute = x.strftime("%M")
    
    path_1 = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Attendance\Class Photo"
    
    filename = "{}\{}-{}-{}-{}.{}.jpg".format(path_1,date,month,year,hour,minute)
    cv2.imwrite(filename, grey)
    
    # Predicting :
    
    # SVM Classifier :
    from sklearn import svm
    clf = svm.SVC(gamma = 'scale')
    clf.fit(encodings, names)
    
    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file(filename)
    
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    print("Number of faces detected: ", no)
    
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        name.append(clf.predict([test_image_enc]))
    
    na = []
    for u in name :
        if u in na :
            pass
        else :
            na.append(u)

    # Converting to CSV :   
    sub = pd.DataFrame({"Students Present in Class" : na})
    path_2 = r"C:\Users\Ronith\Desktop\Data Science 2020\Projects\Attendance\Attendance CSV"
    filename = "{}\{}-{}-{}-{}.{}.csv".format(path_2,date,month,year,hour,minute)
    sub.to_csv(filename)

while True :
    # New Studdent :
    new_stud = input("Any new Student to be added (Y/N) : ").strip().title()
    if new_stud == "Y" :
        add_student()
    else :
        break
# Train the images :
training()

print(" Automated Attendance Begins ----- Thank you!")

while True :
    
    x1 = datetime.datetime.now()
    ho = x1.strftime("%H")
    mi = x1.strftime("%M")
    
    if (ho == '18' and mi == '55') or (ho == '19' and mi == '05') :
        # Attendance :
        attendance()
    else :
        print(ho,mi)
        name = []