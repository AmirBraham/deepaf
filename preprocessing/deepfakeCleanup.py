import cv2
import random
import os

NB_CHECK = 15
THRESHOLD = 0.1

def detectFace(video):
    if os.path.exists("random_frame.jpg"):
        os.remove("random_frame.jpg")
    vidcap = cv2.VideoCapture(video)
    totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    randomFrameNumber=random.randint(0, totalFrames)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("random_frame.jpg", image)
    else:
        return False
    img = cv2.imread('random_frame.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    if(len(faces) == 0):
        return False

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if(len(eyes) <= 1):
            return False
    return True
if __name__ == "__main__":
    VIDEOS_PATH = "../dataset-fake"
    for file in os.listdir(VIDEOS_PATH):
        if file.endswith(".mp4"):
            video = os.path.join(VIDEOS_PATH, file)
            check = 0
            for _ in range(NB_CHECK):
                check+=int(detectFace(video))
            if check/NB_CHECK<THRESHOLD:
                print("Removing",video)
                os.remove(video)