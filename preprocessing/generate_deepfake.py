from os import listdir
from os.path import join
import random

FOLDER = "/home/pafvideo/deepaf/dataset-real"
NB_DEEPFAKES = 10

faces = listdir(join(FOLDER,"faces"))
random.shuffle(faces)
videos = listdir(join(FOLDER,"train"))
random.shuffle(videos)
for i in range(NB_DEEPFAKES):
    random_face = join(join(FOLDER,"faces"),faces[i])
    random_video = join(join(FOLDER,"train"),videos[i])
    print(random_face, random_video)