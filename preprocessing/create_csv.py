import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
import random as rd
fichiers1 = [f for f in listdir("./dataset-real/test") if isfile(join("./test", f))]
fichiers2 = [f for f in listdir("./dataset-real/train") if isfile(join("./train", f))]
fichiers3 = [f for f in listdir("./dataset-fake") if isfile(join("./test", f))]
with open('dataset.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(["file_name", "label","status"])
  for fichier in fichiers1:
    if (rd.random() < 0.2):
      writer.writerow([fichier, "1", "test"])
    else:
      writer.writerow([fichier, "1", "train"])
  for fichier in fichiers2:
    if (rd.random() < 0.2):
      writer.writerow([fichier, "1", "test"])
    else:
      writer.writerow([fichier, "1", "train"])
  for fichier in fichier3:
    if (rd.random() < 0.2):
      writer.writerow([fichier, "0", "test"])
    else:
      writer.writerow([fichier, "0", "train"])