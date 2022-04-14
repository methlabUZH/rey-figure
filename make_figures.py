RESOLUTION_FIG = 'resolution'
ROBUSTNESS_FIG = 'robustness'
HUMAN_COMP_FIG = 'human-comparison'
DATA_PROGR_FIG = 'data-progression'
ABLATIONST_FIG = 'ablation-study'

DO_PLOT = [
    RESOLUTION_FIG,
    ROBUSTNESS_FIG,
    HUMAN_COMP_FIG,
    DATA_PROGR_FIG,
    # ABLATIONST_FIG TODO ...
]

MAIN_SAVE_AS = './results/figures/paper/{}.pdf'

if RESOLUTION_FIG in DO_PLOT:
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
    ], pmeasure=ABSOLUTE_ERROR, save_as=MAIN_SAVE_AS.format('resolution_progression_mae'))

    # mean squared error
    resolution_plot([
        ('78x100', RES_DIR_78x100), ('116x150', RES_DIR_116x150),
        ('232x300', RES_DIR_232x300), ('348x450', RES_DIR_348x450),
    ], pmeasure=SQUARED_ERROR, save_as=MAIN_SAVE_AS.format('resolution_progression_mse'))

if ROBUSTNESS_FIG in DO_PLOT:
    """
    robustness plots
    """
    from constants import TF_ROTATION, TF_PERSPECTIVE, TF_CONTRAST, TF_BRIGHTNESS, ABSOLUTE_ERROR
    from src.analyze.plot_semantic_robustness import make_plot as robustness_plot

    RES_DIR_232x300 = './results/spaceml-results/data_232x300-seed_1/final/rey-multilabel-classifier'
    RES_DIR_232x300_AUG = './results/spaceml-results/data_232x300-seed_1/final-aug/rey-multilabel-classifier'

    # rotations
    angles = [(float(a1), float(a2)) for a1, a2 in zip(range(0, 45, 5), range(5, 50, 5))]  # angle bucket: (from, to)
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_ROTATION,
                    transform_params=angles, pmeasure=ABSOLUTE_ERROR,
                    save_as=MAIN_SAVE_AS.format('robustness_rotation'))

    # perspective
    distortions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_PERSPECTIVE,
                    transform_params=distortions, pmeasure=ABSOLUTE_ERROR,
                    save_as=MAIN_SAVE_AS.format('robustness_perspective'))

    # contrast
    contrast = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_CONTRAST,
                    transform_params=contrast, pmeasure=ABSOLUTE_ERROR,
                    save_as=MAIN_SAVE_AS.format('robustness_contrast'))

    # brightness decrease
    brightness_decr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_BRIGHTNESS, xlabel='Brightness Reduction',
                    transform_params=brightness_decr, pmeasure=ABSOLUTE_ERROR,
                    save_as=MAIN_SAVE_AS.format('robustness_brightness_decrease'))

    # brightness increase
    brightness_incr = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    robustness_plot(RES_DIR_232x300, RES_DIR_232x300_AUG, TF_BRIGHTNESS, xlabel='Brightness Increase',
                    transform_params=brightness_incr, pmeasure=ABSOLUTE_ERROR,
                    save_as=MAIN_SAVE_AS.format('robustness_brightness_increase'))

if DATA_PROGR_FIG in DO_PLOT:
    """
    Data Progression Plots
    """
    from constants import ABSOLUTE_ERROR, SQUARED_ERROR
    from src.analyze.plot_data_progression import make_plot as data_progression_plot

    ####################################################################################################################
    # Debugging ....
    import warnings

    warnings.warn("models for data progression figures have not completed training!")

    DIR_PATTERN1 = './results/spaceml-results/data-progression-116x150/{}-data_116x150-seed_1/'
    DIR_PATTERN1 += 'final{}/rey-multilabel-classifier'
    K_LIST = [1000, 2000, 4000, 6000, 10000, 15000]

    DIR_PATTERN2 = './results/spaceml-results/data-progression-232x300/{}-data_232x300-seed_1/'
    DIR_PATTERN2 += 'final{}/rey-multilabel-classifier'
    ####################################################################################################################

    # DIR_PATTERN1 = './results/spaceml-results/data-progression-232x300/{}-data_232x300-seed_1/'
    # DIR_PATTERN1 += 'final{}/rey-multilabel-classifier'
    # K_LIST = [1000] + list(range(2000, 16001, 2000))]

    dir_configs = [
        {
            'label': 'CNN without Data Augmentation', 'ls': '--', 'marker': 'o', 'color_idx': 0,
            'res-dirs': [(DIR_PATTERN1.format(k, ''), str(k // 1000) + 'k') for k in K_LIST]},
        {
            'label': 'CNN with Data Augmentation', 'ls': '-.', 'marker': 'd', 'color_idx': 1,
            'res-dirs': [(DIR_PATTERN2.format(k, '-aug'), str(k // 1000) + 'k') for k in K_LIST]}
    ]

    data_progression_plot(dir_configs=dir_configs, pmeasure=ABSOLUTE_ERROR,
                          save_as=MAIN_SAVE_AS.format('data_progression'))

if ABLATIONST_FIG in DO_PLOT:
    """
    Ablation Study
    """
    from src.analyze.plot_ablation_study import make_plot as ablation_study_plot

    # TODO...

if HUMAN_COMP_FIG in DO_PLOT:
    """
    Human Comparison Plot
    """
    from src.analyze.plot_human_comparison2 import make_plot as human_comparison_plot

    human_comparison_plot('./results/spaceml-results/data_232x300-seed_1/final-aug/rey-multilabel-classifier',
                          save_as=MAIN_SAVE_AS.format('huam_comparison'))
