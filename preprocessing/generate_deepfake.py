from os import listdir
from os.path import join
import random

FOLDER = "/home/pafvideo/deepaf/dataset-real"

faces = listdir(join(FOLDER,"faces"))
print(join(join(FOLDER,"faces"),random.choice(faces)))