from os import listdir
from os.path import join
import random

FOLDER = "/home/pafvideo/deepaf/dataset-real"
NB_DEEPFAKES = 10

faces = listdir(join(FOLDER,"faces"))
len_faces = len(faces)
random.shuffle(faces)
videos = listdir(join(FOLDER,"train"))
len_videos = len(videos)
random.shuffle(videos)
for i in range(NB_DEEPFAKES):
    random_face = join(join(FOLDER,"faces"),faces[i%len_faces])
    random_video = join(join(FOLDER,"train"),videos[i%len_videos])
    print(random_face, random_video)