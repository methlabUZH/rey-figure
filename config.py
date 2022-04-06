"""
Training and evaluation settings 
"""
config = dict()

"""
Data related settings 
"""
# config['image_size'] = '232 300'  # 116 150 or 232 300
config['data_root'] = {
    '116 150': 'serialized-data/data_116x150-seed_1',
    '232 300': 'serialized-data/data_232x300-seed_1',
    '348 450': 'serialized-data/data_348x450-seed_1'
}

"""
Model related settings 
"""
config['model'] = 'multilabel-classifier'  # multi-label classifier or regressor
config['model_type'] = 'v2'  # v1 or v2 for regressor

"""
Results and logging 
"""
config['results_dir'] = '../spaceml-results'  # './results' # set to model directory for evaluation  # noqa
