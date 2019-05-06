import cv2 as cv
    
face_path='D:\Anaconda\envs\py2\Lib\site-packages\opencv-3.2.0-np111py27_204\Library\etc\haarcascades\haarcascade_frontalface_alt.xml'
eye_path ='D:\Anaconda\envs\py2\Lib\site-packages\opencv-3.2.0-np111py27_204\Library\etc\haarcascades\haarcascade_eye.xml'
upperbody_path='D:\Anaconda\envs\py2\Lib\site-packages\opencv-3.2.0-np111py27_204\Library\etc\haarcascades\haarcascade_upperbody.xml'

face_class = cv.CascadeClassifier(face_path)
eye_class  = cv.CascadeClassifier(eye_path)
upperbody  = cv.CascadeClassifier(upperbody_path)

#green-face.blue-eye,red-upperbody
    
#real-time capture
capture  = cv.VideoCapture(0)
while True:
    ret,frame = capture.read()
    
    if not ret:
        print "VideoCapture read no frame"
        break
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY )
    

    
    detected_faces = face_class.detectMultiScale(gray_img,1.1,minNeighbors=5,minSize=(30,30),flags=cv.CASCADE_SCALE_IMAGE)
    for (y,x,w,h) in detected_faces :
        cv.rectangle(frame,(y,x),(y+w,x+h),(34,245,2),2)
        cv.putText(frame,'FACE',(y,x),1,2,(255,0,0),3)
        roi_gray = gray_img[x:x+h,y:y+w]
        roi_color= frame[x:x+h,y:y+w]
        
        detected_eyes = eye_class.detectMultiScale(roi_gray,1.16,minNeighbors=35,minSize=(25,25),flags=cv.CASCADE_SCALE_IMAGE) 
        for (ey,ex,ew,eh) in detected_eyes:
            cv.rectangle(roi_color,(ey,ex),(ey+ew,ex+eh),(234,2,12),2)
            cv.putText(frame,'EYE',(y+ey,x+ex),1,2,(0,0,250),2)
    detected_upperbody = upperbody.detectMultiScale(gray_img,1.09,minNeighbors=5,floatags=cv.CASCADE_SCALE_IMAGE)
    for (uy,ux,uw,uh) in detected_upperbody:
        cv.rectangle(frame,(uy,ux),(uy+uw,ux+uh),(0,0,242),2)  
        cv.putText(frame,'UPPER_BODY',(uy,ux),1,2,(10,24,70),2)
    
    cv.imshow('Detect',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

capture.release()
cv.destroyAllWindows()


