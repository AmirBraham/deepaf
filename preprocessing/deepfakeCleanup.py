from os import remove
from os import listdir
from os.path import isfile, join
import face_alignment
import cv2

def main():
    DEEPFAKE_DIRECTORY = "../dataset-fake"
    files = [f for f in listdir(DEEPFAKE_DIRECTORY) if isfile(join(DEEPFAKE_DIRECTORY, f))]
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    for f in files:
        vidcap = cv2.VideoCapture(join(DEEPFAKE_DIRECTORY,f))
        success,image = vidcap.read()
        if(success):
            landmarks = fa.get_landmarks(image)
            if landmarks is None :
               print(f'No faces detected {join(DEEPFAKE_DIRECTORY,f)}')
               remove(join(DEEPFAKE_DIRECTORY,f))

if __name__ == "__main__":
    main()