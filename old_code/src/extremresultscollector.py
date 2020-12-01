from cv2 import imread
import os
import math
import shutil
import csv
worst_file = "../data_new/20181110_15-13-32_new_data_5_worst.csv"
worst_dir = "../data_new/worst/"
im_dir= "../data_new/Data_train/"
with open(worst_file) as csv_file:
    labels_reader = csv.reader(csv_file, delimiter=',')
    rows = [row for row in labels_reader]
    filenames = [row[3] for row in rows]
    labels = [row[0] for row in rows]
    predictions = [row[1] for row in rows]
if not os._exists(worst_dir):
    os.mkdir(worst_dir)
for i, fn in enumerate(filenames):
    shutil.copyfile(im_dir+fn, worst_dir+str(i)+"_"+str(labels[i])+"_"+str(predictions[i])+"_"+fn)