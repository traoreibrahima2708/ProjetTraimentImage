#Programme d'apprentissage

import cv2
import os # permet de parcourir une arborescence
import numpy as np
import pickle

#Initialisation des variables
image_dir="./images/" #Variables dans laquelle ya mes images
current_id=0
label_ids={}
x_train=[]
y_labels=[]

for root, dirs, files in os.walk(image_dir):
    #print(root, dirs,files)
    print(image_dir)
    
    
    # on parcours le repertoir root ( ya les variables les dossier)  avec os.walk
    if len(files): 
        label=root.split("/")[-1] #recuperer le nom dans la variable root selon le caracter / avec la fonction split
                                    #avec -1 recupere le dernier element
        for file in files: # parcourir tout les fichiers files
            if file.endswith("png") or file.endswith("jpg"): # recuperer le fichier qui se termine par png ou jpg
                path=os.path.join(root, file) # concatener root et filles 
                if not label in label_ids: # le nom deja present dans id sinon ajoute le dans le tableau label_ids
                    label_ids[label]=current_id
                    current_id+=1
                id_=label_ids[label]  # en fonction du nom de label on recuper id
                image=cv2.imread(path, cv2.IMREAD_GRAYSCALE) # lire le fichier pour ajouter à x_train
                x_train.append(image) #stock toutes les images -comprend chacune des images qu'on a lu
                y_labels.append(id_) # ajoueter id dans labels - contient les identités des differentes images ci-dessus
        print(y_labels)
        print(label_ids)
print(x_train[1])
#cv2.imshow('Demonstrateur', x_train[1])
#print(id_)
                
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)# enregistre id avec le vrai label dans le fichier pickle
    
print(f)
#Creation de deux tableaux 
x_train=np.array(x_train)
y_labels=np.array(y_labels)
recognizer=cv2.face.LBPHFaceRecognizer_create()# LBPHF fonction dediée à la reconnaissance de visage
recognizer.train(x_train, y_labels)# avec chaque image il ya une identité
recognizer.save("trainner.yml") #on enregistre l'entrainement dans trainner.yml et dans le fichier .yml ladans on a tout les caracteristiques de different visage
print(trainner.yml)
