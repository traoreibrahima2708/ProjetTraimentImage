
import cv2

detector=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cam=cv2.VideoCapture(0)
Id=input(['enter votre id'])
sampleNum=0
while(True):
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #
        sampleNum=sampleNum+1
        #
        cv2.imwrite("dataSet/."+ Id +' .'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    # break if the

    elif sampleNum>20:
        breakc
cam.release()
cv2.destroyAllWindows()
