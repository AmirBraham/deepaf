from os import listdir
from os.path import join
import random

FOLDER = "/home/pafvideo/deepaf/dataset-real"
NB_DEEPFAKES = 10

faces = listdir(join(FOLDER,"faces"))
videos = listdir(join(FOLDER,"train"))
for i in range(NB_DEEPFAKES):
    random_face = join(join(FOLDER,"faces"),random.choice(faces))
    random_video = join(join(FOLDER,"train"),random.choice(videos))
    print(random_face, random_video)