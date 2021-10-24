from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from rocf_scoring.data_preprocessing.loading_data import load_raw_data
from rocf_scoring.data_preprocessing.preprocess import preprocess, BIN_LOCATIONS
from rocf_scoring.helpers.helpers import check_if_in_range

class ROCFDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, figures, labels, files, csv_preprocessed=None, csv_raw=None):
        """
        Args:
            data_preprocessing (): Path to the csv file with annotations.
            csv_preprocessed (string, optional): csv file containing information about preprocessed data_preprocessing
                                                 (paths, labels, ...)
            csv_raw (string, optional): csv file containing information about raw data_preprocessing (paths, labels, ...)
        """
        self.figures = figures
        self.labels = labels
        self.files = files

        if csv_preprocessed:
            pass

        if csv_raw:
            pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        _figures = self.figures[idx]
        _labels = self.labels[idx]
        _files = self.files[idx]

        final_score = _labels[-1]
        one_hot_label = np.zeros(len(BIN_LOCATIONS))

        for i, _range in enumerate(BIN_LOCATIONS):
            is_location = check_if_in_range(final_score, _range[1], _range[0])
            if is_location:
                one_hot_label[i] = 1

        preprocessed_img = preprocess(_figures.getImage())[np.newaxis, :]

        return preprocessed_img, _labels, one_hot_label

    # def __str__(self):
    #     return "DATA SET: {}\nData shape:{}\nLabels shape: {}\nFiles shape: {}\n"\
    #         .format(self.name, self.images.shape, self.labels.shape, len(self.files))


if __name__=="__main__":
    figures, labels, files = load_raw_data()
    dataset = ROCFDataset(figures, labels, files)
    dataloader = DataLoader(dataset,
                            batch_size=4, shuffle=True,
                            num_workers=4)

    for (samples, _labels, one_hot_labels) in dataloader:

        _labels = torch.stack(_labels)
        _labels = torch.transpose(_labels, 0, 1)

        print("data_preprocessing: ", samples.size())
        print("labels: ", _labels)
        print("one hot: ", one_hot_labels)
