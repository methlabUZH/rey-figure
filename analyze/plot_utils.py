import matplotlib as mpl
import seaborn as sns


def init_mpl(sns_style="whitegrid", colorpalette='muted', fontsize=16, grid_lw=1.0):
    sns.set_style(sns_style)
    sns.set_palette(colorpalette)
    colors = sns.color_palette(colorpalette)
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["grid.linewidth"] = grid_lw / 2.0
    mpl.rcParams["axes.linewidth"] = grid_lw
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.
    mpl.rcParams['ytick.major.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.

    return colors