import face_alignment
import cv2
from os import listdir
from os.path import join
import torch
video_folder="/home/pafvideo/video-preprocessing-master/vox/test"

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

for f in listdir(video_folder):
   if f.endswith(".mp4"):
      vidcap = cv2.VideoCapture(join(video_folder,f))
      success,image = vidcap.read()
      landmarks = fa.get_landmarks(image)
      if landmarks is None :
         print(f'No faces detected {join(video_folder,f)}')