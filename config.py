"""
Training and evaluation settings 
"""
config = dict()

"""
Data related settings 
"""
config['image_size'] = '232 300'  # 116 150 or 232 300
config['data_root'] = './data/serialized-data/data_232x300-seed_1'

"""
Model related settings 
"""
config['model'] = 'multilabel-classifier'  # multi-label classifier or regressor
config['model_type'] = 'v2'  # v1 or v2 for regressor

"""
Results and logging 
"""
config['results_dir'] = '/home/ubuntu/projects/rey-figure/results/data_232x300-seed_1/final-bigsize/rey-multilabel-classifier'  # './results' # set to model directory for evaluation  # noqa
