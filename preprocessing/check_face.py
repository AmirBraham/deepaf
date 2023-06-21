import face_alignment
import cv2
from os import listdir, remove
from os.path import join
import sys



def cleanup():
   folder = sys.argv[1]
   video_folder="/home/pafvideo/video-preprocessing-master/vox/" + folder
   fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
   for f in listdir(video_folder):
      if f.endswith(".mp4"):
         vidcap = cv2.VideoCapture(join(video_folder,f))
         success,image = vidcap.read()
         if success:
            landmarks = fa.get_landmarks(image)
            if landmarks is None :
               print(f'No faces detected {join(video_folder,f)}')
               remove(join(video_folder,f))

if __name__ == "__main__":
   cleanup()