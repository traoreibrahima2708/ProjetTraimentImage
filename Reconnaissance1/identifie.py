
# Identification

import cv2
import pickle
import numpy as np
import common as c
#Fichier permetant d'identifier le visage (haarrcascade)
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create() #recoonaisseur

#lecture du fichier d'entrainement
recognizer.read("trainner.yml")# lire le fichier creer dans l'apprentissage
id_image=0
color_info=(255, 255, 255)
color_ko=(0, 0, 255)
color_ok=(0, 255, 0)
# lire le fichier labels.pickle
with open("labels.pickle", "rb") as f:
    og_labels=pickle.load(f) 
    labels={v:k for k, v in og_labels.items()}

#cap=cv2.VideoCapture("nom de la video, Hu.mp4")
    
cap = cv2.VideoCapture(0)
while True:
    ret, frame=cap.read() # recupere l'image 
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=4, minSize=(c.min_size, c.min_size)) # faux positive
    for (x, y, w, h) in faces:
        roi_gray=gray[y:y+h, x:x+w] #image recuperee de la webcam juste le visage avec les coordonnée x,,y , h, w -redimensionnement 
        
        #id_, conf=recognizer.predict(cv2.resize(roi_gray, (c.min_size, c.min_size)))
        id_, conf=recognizer.predict(roi_gray) # on lance la fonction de prediction et qui va renvoyer 2 variables :id et conf( indice de confiance)
        print(id_)
        print(conf)
        
        if conf<=95: # plus indice de conf est bas ==>id est bonne  et plus indice est elevee moins c'est sûr d'elle dans la prediction (valeur general :80-100)
             color=color_ok# couleur verte
             name=labels[id_]# on cherche dans labels id 
        else:
            color=color_ko # couleur rouge
            name="Inconnu" # id est inconnu
        label=name+" "+'{:5.2f}'.format(conf)
        print(label)
        #
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)#afficher le label et coef de confiance
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)# afficher la couleur 
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark) # fps vitesse 
    cv2.putText(frame, "FPS: {:05.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color_info, 2)
    cv2.imshow('Demonstrateur', frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    if key==ord('a'):
        for cpt in range(100): # stock 100 images dans le dataSet
            ret, frame=cap.read()

cv2.destroyAllWindows()
print("Fin")
