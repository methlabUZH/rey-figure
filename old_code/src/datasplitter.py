from sklearn.model_selection import train_test_split
import csv
from cv2 import imread
import os
import math
import shutil
DEBUG=True
RANDOM_STATE=123

RAW = [
    {'name': "Mexico", 'images' : "../preprocessing/raw/Mexico_NA/", 'labels': "../preprocessing/raw/mexico_NA.csv"},
    {'name': "Colombia", 'images': "../preprocessing/raw/Colombia_NA/", 'labels': "../preprocessing/raw/colombia_NA.csv"},
]
def is_number(s):
    try:
        number = float(s)
        return not math.isnan(number)
    except ValueError:
        return False

def load_raw_data_NA_split(image_path, label_path):
    if(DEBUG):
        print("Loading dataset "+image_path)
    #read labels
    with open(label_path) as csv_file:
        labels_reader = csv.reader(csv_file, delimiter=',')
        rows = [row for row in labels_reader]
        filenames = [row[0] for row in rows]
        labels = [row[1] for row in rows]

    # find indices of valid labels (ones that can be converted to numbers and are not NaN)
    valid_indices = [index for index, label in enumerate(labels)]
    invalid_indices = []
    valid_labels = [labels[i] for i in valid_indices]

    # extract filenames corresponding to valid indices
    valid_filenames = [f for ind, f in enumerate(filenames) if (ind in valid_indices)]

    if len(invalid_indices)>0:
        invalid_filenames = [f for ind, f in enumerate(filenames) if (ind in invalid_indices)]
        invalid_labels = [labels[i] for i in invalid_indices]
        with open(label_path[:-4] + "_NA" + '.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            writer.writerows(zip(invalid_filenames, invalid_labels))
        csv_file.close()
        NA_path = (image_path[:-1] + "_NA/")
        if not os.path.exists(NA_path):
            os.mkdir(NA_path)

        for f in invalid_filenames:
            shutil.move(image_path+f, NA_path+f)


    # read images
    valid_paths = [image_path + f for f in valid_filenames]
    images = [imread(file) for file in valid_paths]


    return images, valid_labels, valid_filenames
for set in RAW:
    images, labels, filenames = load_raw_data_NA_split(set['images'], set['labels'])

    test_size=0.2
    image_train, image_test, labels_train, labels_test, filenames_train, filenames_test=train_test_split(images, labels, filenames, test_size=test_size, random_state=RANDOM_STATE)
    with open(set['labels'][:-4] + "_test" + '.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(filenames_test, labels_test))
    csv_file.close()
    test_path = (set['images'][:-1] + "_test/")
    with open(set['labels'][:-4] + "_trainval" + '.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(filenames_train, labels_train))
    csv_file.close()

    test_path = (set['images'][:-1] + "_test/")
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    for f in filenames_test:
        shutil.move(set['images'] + f, test_path + f)






