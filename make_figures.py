import warnings
warnings.warn("figures are still under development! might be buggy...")

# which = 'resolution'
# which = 'robustness'
# which = 'data-progression'
which = 'ablation-study'
# which = 'human-comparison'

if which == 'resolution':
    """
    resolution progression
    """
    from constants import SQUARED_ERROR, ABSOLUTE_ERROR
    from src.analyze.plot_resolution_progression import make_plot as resolution_plot

    RES_DIR_78x100 = './results/spaceml-results/data_78x100-seed_1/final/rey-multilabel-classifier'
    RES_DIR_116x150 = './results/spaceml-results/data_116x150-seed_1/final/rey-multilabel-classifier'
    RES_DIR_232x300 = './results/spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier'
    RES_DIR_348x450 = './results/spaceml-results/data_348x450-seed_1/final/rey-multilabel-classifier'

    # mean absolute error
    resolution_plot([
        ('78x100', RES_DIR_78x100), ('116x150', RES_DIR_116x150),
        ('232x300', RES_DIR_232x300), ('348x450', RES_DIR_348x450),
    ], pmeasure=ABSOLUTE_ERROR, save_as='./results/figures/paper/resolution_progression_mae.pdf')

    # mean squared error
    resolution_plot([
        ('78x100', RES_DIR_78x100), ('116x150', RES_DIR_116x150),
        ('232x300', RES_DIR_232x300), ('348x450', RES_DIR_348x450),
    ], pmeasure=SQUARED_ERROR, save_as='./results/figures/paper/resolution_progression_mse.pdf')

if which == 'robustness':
    """
    robustness plots
    """
    from constants import TF_ROTATION, TF_PERSPECTIVE, TF_CONTRAST, TF_BRIGHTNESS, ABSOLUTE_ERROR
    from src.analyze.plot_semantic_robustness import make_plot as robustness_plot

    RES_DIR_232x300 = './results/spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier'
    RES_DIR_232x300_AUG = './results/spaceml-results/data_232x300-seed_1/final-aug/rey-multilabel-classifier'

    # rotations
    angles = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    angles = [(a1, a2) for a1, a2 in zip(angles[:-1], angles[1:])]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_ROTATION,
                    transform_params=angles, pmeasure=ABSOLUTE_ERROR, save_as=None)

    # perspective
    distortions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_PERSPECTIVE,
                    transform_params=distortions, pmeasure=ABSOLUTE_ERROR, save_as=None)

    # contrast
    contrast = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_CONTRAST,
                    transform_params=contrast, pmeasure=ABSOLUTE_ERROR, save_as=None)

    # brightness decrease
    brightness_decr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_BRIGHTNESS, xlabel='Brightness Reduction',
                    transform_params=brightness_decr, pmeasure=ABSOLUTE_ERROR, save_as=None)

    # brightness increase
    brightness_incr = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_BRIGHTNESS, xlabel='Brightness Increase',
                    transform_params=brightness_incr, pmeasure=ABSOLUTE_ERROR, save_as=None)

if which == 'data-progression':
    """
    Data Progression Plots
    """
    from constants import ABSOLUTE_ERROR, SQUARED_ERROR
    from src.analyze.plot_data_progression import make_plot as data_progression_plot

    dir_pattern = './results/spaceml-results/data-progression/{}-data_116x150-seed_1/final/rey-multilabel-classifier'
    RES_DIR_1000 = dir_pattern.format(1000)
    RES_DIR_2000 = dir_pattern.format(2000)
    RES_DIR_4000 = dir_pattern.format(4000)
    RES_DIR_6000 = dir_pattern.format(6000)
    RES_DIR_10000 = dir_pattern.format(10000)
    RES_DIR_15000 = dir_pattern.format(15000)

    data_progression_plot([
        (RES_DIR_1000, '1k'), (RES_DIR_2000, '2k'), (RES_DIR_4000, '4k'),
        (RES_DIR_6000, '6k'), (RES_DIR_10000, '10k'), (RES_DIR_15000, '15k'),
    ], pmeasure=ABSOLUTE_ERROR, save_as=None)

    quit()

if which == 'ablation-study':
    """
    Ablation Study
    """
    from src.analyze.plot_ablation_study import make_plot as ablation_study_plot

    # TODO...

if which == 'human-comparison':
    """
    Human Comparison Plot
    """
    from src.analyze.plt_human_comparison import make_plot as human_comparison_plot

    # TODO...
