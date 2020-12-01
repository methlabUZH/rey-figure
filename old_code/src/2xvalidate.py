from train import cross_validate
from sklearn.model_selection import KFold
from dataloader import DATA
from cv_results import cv_predict


kf = KFold(n_splits=10)
for i, (train_idx, val_idx) in enumerate(kf.split(DATA.images)):
    data_train, data_val = DATA.split(train_idx, val_idx)
    print("Starting cross validation run number {}...".format(i))
    res, runname = cross_validate(data_train)
    print("Calling cv_predict...")
    cv_predict(data_val, '../models/' + runname)
    print("Finished calling cv_predict...")
