import numpy as np
from model import CNN
import csv
import time
from dataloader import DATA_TEST





def restore(pathname):
    model = CNN('restored', 0)
    model.restore_model(pathname)
    return model


def ensemble_predict(data, modelpaths,run):
    final = np.empty((data.images.shape[0], 10))

    for i, modelpath in enumerate(modelpaths):
        print("Running test prediction for modelpath: {}...".format(modelpath))
        model = restore(modelpath + '.ckpt')
        temp = model.predict(data.images)
        temp = np.asarray(temp[:, 18])
        final[:, i] = temp

    print("Writing result file... ")
    with open('test_results_'+run+'.csv', 'w') as csv_file_2:
        labels_writer = csv.writer(csv_file_2, delimiter=',')
        labels_writer.writerow(["filename", "true label", "prediction", "all predictions"])
        for i in range(data.images.shape[0]):
            labels_writer.writerow([data.files[i].filename, data.labels[i,-1], np.mean(final[i,:])] + list(final[i,:]))


if __name__ == '__main__':
    #runs = ['20181211_13-28-29_ALL_final_train_10times0','20181211_13-28-19_ALL_final_train_10times1', '20181211_13-28-28_ALL_final_train_10times2','20181211_13-28-28_ALL_final_train_10times3','20181211_13-28-30_ALL_final_train_10times4']
    runs = ['20181215_12-39-51_ALL_final_binning0','20181215_12-39-54_ALL_final_binning1', '20181215_12-40-03_ALL_final_binning2','20181215_12-39-54_ALL_final_binning3']

    for run in runs:
        basepath = "/home/jheitz/dslab/models/"+run+"/"
        modelpaths = [basepath+'model_fold'+str(i) for i in range(10)]
        ensemble_predict(DATA_TEST, modelpaths,run)
