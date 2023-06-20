from os import listdir
from os.path import join
import os

video_folder = "./videos"

onlymp4 = []

for f in listdir(video_folder):
    if f.endswith(".mp4"):
        onlymp4.append(join(video_folder, f))

for f in onlymp4:
    out = os.popen("ffmpeg -v error -i "+f+" -f null - 2>&1").read()
    if len(out)!=0:
        print(f + " is corrupted")
    else:
        print(f + " not corrupted")