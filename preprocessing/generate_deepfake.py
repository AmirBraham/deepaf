from os import listdir
from os.path import join
import random

FOLDER = "/home/pafvideo/deepaf/dataset-real"

faces = listdir(join(FOLDER,"faces"))
print(join(faces,str(random.choice(faces))))