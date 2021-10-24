import os
import pandas as pd

LABEL_FORMAT_ONE_PER_ITEM = 'one-per-item'
LABEL_FORMAT_THREE_PER_ITEM = 'three-per-item'


def join_ground_truth_files(labels_root: str) -> pd.DataFrame:
    """
    find all csv files in labels_root and join them into one pandas dataframe
    returns:
        pd.DataFrame with merged user rating data_preprocessing without duplicates
    """
    all_label_files = os.listdir(labels_root)

    dataframes = []
    for fn in all_label_files:

        if fn.startswith('.'):
            continue

        try:
            df = pd.read_csv(os.path.join(labels_root, fn))
            print(f'successfully loaded {os.path.join(labels_root, fn)}')
        except Exception:  # noqa
            print(f'failed to load {os.path.join(labels_root, fn)}')
            continue

        dataframes.append(df)

    # concatenate dataframes
    df = pd.concat(dataframes, ignore_index=True, sort=False)
    df = df.drop_duplicates()

    # cast type
    df = df.astype({'FILE': str})

    # include column with figure_id
    df['figure_id'] = df.apply(lambda row: os.path.splitext(row['FILE'])[0], axis=1)

    return df

# def load_raw_data(mode='train', label_format=LABEL_FORMAT_ONE_PER_ITEM):
#     if label_format not in [LABEL_FORMAT_ONE_PER_ITEM, LABEL_FORMAT_THREE_PER_ITEM]:
#         raise ValueError(
#             f'label_format must be one of [{LABEL_FORMAT_ONE_PER_ITEM},{LABEL_FORMAT_THREE_PER_ITEM}]! ' +
#             f'got {label_format}')
#
#     raw_figures = {}
#     with open(os.path.join(DATA_DIR, LABEL_FILE)) as csv_file:
#         labels_reader = csv.reader(csv_file, delimiter=',')
#
#         rows = [row for row in labels_reader]
#         rows = rows[1:]  # first is header
#
#         for ind, row in enumerate(rows):
#
#             if not is_number(row[11]) or float(row[11]) >= 0.8:
#                 filename = row[5]
#                 assessment_id = row[0]
#
#                 # add this assessment / figure if in correct (train or test) directory
#                 if os.path.exists(DATA_DIR + "{}/".format(mode) + filename):
#
#                     if filename not in raw_figures:
#                         raw_figures[filename] = ReyFigure(filename)
#
#                     assessment = raw_figures[filename].get_assessment(assessment_id)
#                     assessment.add_item(item_id=row[6],
#                                         score=row[7],
#                                         visible=row[8],
#                                         right_place=row[9],
#                                         drawn_correctly=row[10])
#
#     # convert dictionary of figures to list
#     figures = [raw_figures[fig] for fig in raw_figures if raw_figures[fig].has_valid_assessment()]
#
#     # get list of labels
#     if label_format == 'one-per-item':
#         labels = [fig.get_median_score_per_item() + [fig.get_median_score()] for fig in figures]
#
#     elif label_format == 'three-per-item':
#         labels = [fig.get_median_part_score_per_item() + [fig.get_median_score()] for fig in figures]
#
#     else:
#         labels = [fig.get_median_score() for fig in figures]
#
#     # get list of files
#     directory = DATA_DIR + "{}/".format(mode)
#     print("directory: ", directory)
#     files = [File(fig.filename, directory) for fig in figures]
#
#     return figures, labels, files

# def load_meta_data(csv_metadata):
#     print(f'[INFO] Loading metadata from {csv_metadata}')
#     meta_df = pd.read_csv(csv_metadata)
#     meta_df['regression_label'] = meta_df['regression_label'].apply(literal_eval)
#     return meta_df
