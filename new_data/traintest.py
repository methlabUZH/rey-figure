from sklearn.model_selection import train_test_split
import csv
import os
import shutil
import numpy as np


RANDOM_STATE= 123
test_size = 0.2

def create_directory(path, empty_dir = False):
    """ Creates directory if it doesnt exist yet, optionally deleting all files in there """
    if not os.path.exists(path):
        os.makedirs(path)

    if empty_dir:
        shutil.rmtree(path)
        os.makedirs(path)

    try:
        os.chmod(path, 0o777)
    except PermissionError:
        pass
train_indices = []
test_indices = []
tobesplit_indices = []
missing_indices = []
label_path = "Data2018-11-23.csv"
train_dir = "Data_train/"
test_dir = "Data_test/"
all_dir = "uploadFinal/"
check_folder = "../data/raw/"
test_folder = ["Mexico_test/","Mexico_NA_test/","Brugger_Daten_test/","Colombia_test/","Colombia_NA_test/"]
train_folder = ["Mexico/","Mexico_NA/","Brugger_Daten/","Colombia/","Colombia_NA/"]
with open(label_path) as csv_file:
    labels_reader = csv.reader(csv_file, delimiter=',')
    rows = [row for row in labels_reader]
    rows = rows[1:]
    filenames = [row[5] for row in rows]
filenames = np.unique(filenames)
create_directory(test_dir)
create_directory(train_dir)


for i ,fn in enumerate(filenames):
    exists = False
    moved = False
    if os.path.exists(test_dir + fn):
        test_indices.append(i)
        moved = True
        exists =True
    if os.path.exists(train_dir + fn):
        train_indices.append(i)
        moved = True
        exists = True
    if not moved:
        if os.path.exists(all_dir+fn):
            exists = True
            for tf in test_folder:
                if os.path.exists(check_folder+tf+fn):
                    shutil.move(all_dir + filenames[i], test_dir + filenames[i])
                    test_indices.append(i)
                    moved = True
            if not moved:
                for tf in train_folder:
                    if os.path.exists(check_folder + tf + fn):
                        shutil.move(all_dir + filenames[i], train_dir + filenames[i])
                        train_indices.append(i)
                        moved = True
                if not moved:
                        tobesplit_indices.append(i)
    if not exists:
        missing_indices.append(i)





if np.shape(tobesplit_indices)[0]>0:
    test_size = ((np.shape(test_indices)[0]+np.shape(train_indices)[0]+np.shape(tobesplit_indices)[0])*test_size-np.shape(test_indices)[0])/np.shape(tobesplit_indices)[0]
    add_train, add_test = train_test_split(tobesplit_indices, test_size=test_size, random_state=RANDOM_STATE)

    if np.shape(add_test)[0]>0:
        for i in add_test:
            shutil.move(all_dir + filenames[i], test_dir + filenames[i])
    if np.shape(add_train)[0]>0:
        for i in add_train:
            shutil.move(all_dir + filenames[i], train_dir + filenames[i])

if np.shape(missing_indices)[0]>0:
    mfn = filenames[missing_indices]
    with open('missing_images.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(mfn))








