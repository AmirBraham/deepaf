import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
import random as rd
fichiers1 = [f for f in listdir("./test") if isfile(join("./test", f))]
fichiers2 = [f for f in listdir("./train") if isfile(join("./train", f))]
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
