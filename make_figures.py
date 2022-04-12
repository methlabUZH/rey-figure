"""
resolution progression
"""
from constants import SQUARED_ERROR, ABSOLUTE_ERROR
from src.analyze.plot_resolution_progression import make_plot as resolution_plot

RES_DIR_78x100 = './results/spaceml-results/data_78x100-seed_1/final/rey-multilabel-classifier'
RES_DIR_116x150 = './results/spaceml-results/data_116x150-seed_1/final/rey-multilabel-classifier'
RES_DIR_232x300 = './results/spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier'
RES_DIR_348x450 = './results/spaceml-results/data_348x450-seed_1/final/rey-multilabel-classifier'

resolution_plot([
    ('78x100', RES_DIR_78x100), ('116x150', RES_DIR_116x150),
    ('232x300', RES_DIR_232x300), ('348x450', RES_DIR_348x450),
], pmeasure=ABSOLUTE_ERROR, save_as='./results/figures/paper/resolution_progression_mae.pdf')

resolution_plot([
    ('78x100', RES_DIR_78x100), ('116x150', RES_DIR_116x150),
    ('232x300', RES_DIR_232x300), ('348x450', RES_DIR_348x450),
], pmeasure=SQUARED_ERROR, save_as='./results/figures/paper/resolution_progression_mse.pdf')

"""
robustness plots
"""
from src.analyze.plot_semantic_robustness2 import make_plot as robustness_plot

"""
Data Progression Plots
"""
# TODO...

"""
Ablation Study
"""
# TODO...
