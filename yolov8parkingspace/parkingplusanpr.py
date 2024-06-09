import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import easyocr

# Loads pre-trained YOLOv8 model for object detection
model=YOLO('yolov8s.pt')

# Used to print the mouse coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {
    "AN": "Andaman and Nicobar",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odissa",
    "PY": "Pondicherry",
    "PN": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "CG": "Chhattisgarh",
    "TS": "Telangana",
    "JH": "Jharkhand",
    "UK": "Uttarakhand"
}

# Opens video file for processing
cap=cv2.VideoCapture('plate.mp4')

# Reads from a text file containing names of object classes that the model can detect
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   
# Coordinates of 12 different parking spots
area1=[(52,364),(30,417),(73,412),(88,369)]

area2=[(105,353),(86,428),(137,427),(146,358)]

area3=[(159,354),(150,427),(204,425),(203,353)]

area4=[(217,352),(219,422),(273,418),(261,347)]

area5=[(274,345),(286,417),(338,415),(321,345)]

area6=[(336,343),(357,410),(409,408),(382,340)]

area7=[(396,338),(426,404),(479,399),(439,334)]

area8=[(458,333),(494,397),(543,390),(495,330)]

area9=[(511,327),(557,388),(603,383),(549,324)]

area10=[(564,323),(615,381),(654,372),(596,315)]

area11=[(616,316),(666,369),(703,363),(642,312)]

area12=[(674,311),(730,360),(764,355),(707,308)]

# areas = [area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11, area12]


def extract_num(frame):
    reader = easyocr.Reader(['en'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in nplate:
        wT, hT, cT = frame.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = frame[y + a:y + h - a, x + b:x + w - b, :]
        
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Read the text on the plate using EasyOCR
        result = reader.readtext(plate)
        if result:
            read = result[0][1]
        else:
            read = ""
        
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(frame, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(frame, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


# Main Loop until Esc
while True:  
    # Reads frame from the video  
    ret,frame = cap.read()
    if not ret:
        break
    # time.sleep(1)
    # resizes the frame to a specific size.
    frame=cv2.resize(frame,(1020,500))
    # runs object detection on the frame using the YOLOv8 model.
    results=model.predict(frame)
    # extracts bounding box information from the model's output. 
    a=results[0].boxes.data
    # converts bounding box data into a pandas DataFrame for easier processing.
    px=pd.DataFrame(a).astype("float")    

    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    list7=[]
    list8=[]
    list9=[]
    list10=[]
    list11=[]
    list12=[]
    
    # Iterate through detected objects
    for index,row in px.iterrows():

        # Represents coordinates of the bounding box
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            
            extract_num(frame)

            results1=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if results1>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list1.append(c)
               cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            
            results2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if results2>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list2.append(c)
            
            results3=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
            if results3>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list3.append(c)   
            results4=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
            if results4>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list4.append(c)  
            results5=cv2.pointPolygonTest(np.array(area5,np.int32),((cx,cy)),False)
            if results5>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list5.append(c)  
            results6=cv2.pointPolygonTest(np.array(area6,np.int32),((cx,cy)),False)
            if results6>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list6.append(c)  
            results7=cv2.pointPolygonTest(np.array(area7,np.int32),((cx,cy)),False)
            if results7>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list7.append(c)   
            results8=cv2.pointPolygonTest(np.array(area8,np.int32),((cx,cy)),False)
            if results8>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list8.append(c)  
            results9=cv2.pointPolygonTest(np.array(area9,np.int32),((cx,cy)),False)
            if results9>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list9.append(c)  
            results10=cv2.pointPolygonTest(np.array(area10,np.int32),((cx,cy)),False)
            if results10>=0:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
                list10.append(c)     
            results11=cv2.pointPolygonTest(np.array(area11,np.int32),((cx,cy)),False)
            if results11>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list11.append(c)    
            results12=cv2.pointPolygonTest(np.array(area12,np.int32),((cx,cy)),False)
            if results12>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list12.append(c)
              
    a1=(len(list1))
    a2=(len(list2))       
    a3=(len(list3))    
    a4=(len(list4))
    a5=(len(list5))
    a6=(len(list6)) 
    a7=(len(list7))
    a8=(len(list8)) 
    a9=(len(list9))
    a10=(len(list10))
    a11=(len(list11))
    a12=(len(list12))
    o=(a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12)
    space=(12-o)
    print(space)
    if a1==1:
        # cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a2==1:
        # cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a3==1:
        # cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a4==1:
        # cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a5==1:
        # cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a6==1:
        # cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
    if a7==1:
        # cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('7'),(456,414),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area7,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('7'),(456,414),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a8==1:
        # cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('8'),(527,406),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area8,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('8'),(527,406),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)  
    if a9==1:
        # cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('9'),(591,398),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area9,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('9'),(591,398),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a10==1:
        # cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('10'),(649,384),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area10,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('10'),(649,384),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a11==1:
        # cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area11,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('11'),(697,377),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a12==1:
        # cv2.polylines(frame,[np.array(area12,np.int32)],True,(0,0,255),2)
        cv2.putText(frame,str('12'),(752,371),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        # cv2.polylines(frame,[np.array(area12,np.int32)],True,(0,255,0),2)
        cv2.putText(frame,str('12'),(752,371),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)

    cv2.putText(frame,str(space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
#stream.stop()
