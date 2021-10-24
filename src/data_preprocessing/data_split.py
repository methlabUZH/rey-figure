# import os
# import random
# import shutil
# from tqdm import tqdm
#
#
# def random_split(raw_data_dir, data_dir):
#     source_folder = raw_data_dir
#     train_folder = os.path.join(data_dir, 'train/')
#     test_folder = os.path.join(data_dir, 'test/')
#
#     if not os.path.exists(train_folder):
#         os.makedirs(train_folder)
#
#     if not os.path.exists(test_folder):
#         os.makedirs(test_folder)
#
#     files = os.listdir(source_folder)
#     num_files = len(files)
#
#     # select 20% test data_preprocessing randomly
#     test_files = random.sample(files, num_files // 5)
#     train_files = [f for f in files if f not in test_files]
#
#     # copy to test folder
#     for file_name in tqdm(test_files):
#         shutil.copy(os.path.join(source_folder, file_name), test_folder)
#
#     # copy to train folder
#     for file_name in tqdm(train_files):
#         shutil.copy(os.path.join(source_folder, file_name), train_folder)
