

# import required libraries
import cv2
from os import listdir
from os.path import isfile, join


def main():
    celeb_cropped = "../dataset-real/celebcropped/"
    celeb_path = "../dataset-real/celebfaces/"
    files = [f for f in listdir(celeb_path) if isfile(join(celeb_path, f))]
    for f in files:
        # read the input image
        img = cv2.imread(celeb_path + f)
        width , height,_ = img.shape
        # convert to grayscale of each frames
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # read the haarcascade to detect the faces in an image
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        # detects faces in the input image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #print('Number of detected faces:', len(faces))

        # loop over all detected faces
        if len(faces) > 0:
            for i, (x,y,w,h) in enumerate(faces):
                # To draw a rectangle in a face
                padding = 15
                face = img[max(0,y-padding):min(y+h + padding,width),max(0,x-padding):min(x+w+padding,height)]
               
                cv2.imwrite(f'{celeb_cropped}{f}_cropped.jpg', face)
                #print(f'{celeb_cropped}{f}_cropped.jpg')


if __name__ == "__main__":
   main()