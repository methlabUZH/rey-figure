# from torch.dataloaders.data_preprocessing import Dataset
# import torch
# import numpy as np
# import os
#
# from constants import DATA_DIR, BIN_LOCATIONS_DENSE, BIN_LOCATIONS


# class ROCFDatasetRegression(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, df_metadata=None):
#         """
#         Args:
#
#         """
#         self._df_metadata = df_metadata
#
#         # Regress on median scores
#         self._labels = np.asarray(self._df_metadata["regression_label"])
#
#         self._path_npy = np.asarray(self._df_metadata["path_npy"])
#
#         for i in range(len(self._path_npy)):
#             self._path_npy[i] = os.path.join(DATA_DIR, self._path_npy[i])
#
#     def __len__(self):
#         return len(self._labels)
#
#     def __getitem__(self, idx):
#
#         img = np.load(self._path_npy[idx])[np.newaxis, :]
#         label = torch.from_numpy(np.asarray(self._labels[idx])).type('torch.FloatTensor')
#
#         return img, label

# class ROCFDatasetClassification(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, df_metadata=None, binning='default'):
#         """
#         Args:
#
#         """
#         self._df_metadata = df_metadata
#         self._binning = binning
#
#         if binning == "dense":
#             self._labels = np.asarray(self._df_metadata["classification_label_dense"])
#         else:
#             self._labels = np.asarray(self._df_metadata["classification_label"])
#
#         self._path_npy = np.asarray(self._df_metadata["path_npy"])
#
#         for i in range(len(self._path_npy)):
#             self._path_npy[i] = os.path.join(DATA_DIR, self._path_npy[i])
#
#     def __len__(self):
#         return len(self._labels)
#
#     def __getitem__(self, idx):
#
#         img = np.load(self._path_npy[idx])[np.newaxis, :]
#
#         label = self._labels[idx]
#
#         if self._binning == "dense":
#             one_hot_label = np.zeros(len(BIN_LOCATIONS_DENSE))
#         else:
#             one_hot_label = np.zeros(len(BIN_LOCATIONS))
#
#         one_hot_label[label] = 1
#         one_hot_label = torch.from_numpy(np.asarray(one_hot_label)).type('torch.FloatTensor')
#
#         return img, label, one_hot_label
#
#
#
