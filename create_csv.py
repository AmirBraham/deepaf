import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
fichiers1 = [f for f in listdir("./vox/test") if isfile(join("./vox/test", f))]
fichiers2 = [f for f in listdir("./vox/train") if isfile(join("./vox/train", f))]
with open('dataset.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(["file_name", "label"])
  for fichier in fichiers1:
    writer.writerow([fichier, "1"])
  for fichier in fichiers2:
    writer.writerow([fichier, "1"])