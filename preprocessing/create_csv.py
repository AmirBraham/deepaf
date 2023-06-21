import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
import random as rd
fichiers1 = [f for f in listdir("./dataset-real/test") if isfile(join("./dataset-real/test", f))]
fichiers2 = [f for f in listdir("./dataset-real/train") if isfile(join("./dataset-real/train", f))]
fichiers3 = [f for f in listdir("./dataset-fake") if isfile(join("./dataset-fake", f))]
with open('dataset.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(["file_name", "label","status","subfolder"])
  for fichier in fichiers1:
    if (rd.random() < 0.2):
      writer.writerow([fichier, "1", "test","dataset-real/test"])
    else:
      writer.writerow([fichier, "1", "train","dataset-real/test"])
  for fichier in fichiers2:
    if (rd.random() < 0.2):
      writer.writerow([fichier, "1", "test","dataset-real/train"])
    else:
      writer.writerow([fichier, "1", "train","dataset-real/train"])
  for fichier in fichiers3:
    if (rd.random() < 0.2):
      writer.writerow([fichier, "0", "test","dataset-fake"])
    else:
      writer.writerow([fichier, "0", "train","dataset-fake"])
