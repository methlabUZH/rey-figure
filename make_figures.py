"""
resolution progression
"""
from src.analyze.plot_resolution_progression import make_plot as resolution_plots

resolution_plots([
    ('78x100', './results/spaceml-results/data_78x100-seed_1/final/rey-multilabel-classifier'),
    ('116x150', './results/spaceml-results/data_116x150-seed_1/final/rey-multilabel-classifier'),
    ('232x300', './results/spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier')
])
