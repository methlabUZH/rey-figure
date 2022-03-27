import numpy as np
from config import MODEL_PATH, DATA_DIR, LABEL_FORMAT
from model import CNN
import csv
from dataloader import DATA

# TODO: adapt such that it works for all formats ('one-per-item', ...), currently only works for 'one'
# restore commit in result.txt to ensure loading model parameters works


def restore(pathname):
    model = CNN('restored', 0)
    model.restore_model(pathname)
    return model


def getTestImages(foldername):
    path = DATA_DIR + foldername + '/'
    filenames = np.genfromtxt('../preprocessing/' + foldername + '.csv', dtype='|U64')
    return path, filenames


def cv_predict(data, modelpath):
    print("Running cv predict for modelpath: {}...".format(modelpath))
    final = np.empty((data.images.shape[0], 10))

    for i in range(10):
        model = restore(modelpath + '/model_fold{}.ckpt'.format(i))

        temp = model.predict(data.images)

        temp = np.asarray(temp[:, 0])

        final[:, i] = temp

    result = np.mean(final, axis=1)
    result = np.reshape(result, (-1, 1))
    result = np.concatenate((result, np.reshape(np.var(final, axis=1), (-1, 1))), axis=1)

    print("Writing result file... ")
    with open(modelpath + '/result.csv', 'w') as csv_file_2:
        labels_writer = csv.writer(csv_file_2, delimiter=',')
        labels_writer.writerow(["filename"] + ["mean"] + ["var"])
        # binned_predictions = binned_predictions.tolist()
        # binned_labels = binned_labels.tolist()
        labels_writer.writerows(zip((file.filename for file in data.files), result[:, 0], result[:, 1]))


if __name__ == '__main__':
    cv_predict(DATA, MODEL_PATH)
