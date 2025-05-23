# external imports
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, fontManager


PLOT_FORMAT = "pdf"
PLOT_SIZE = 10

COLORS = [
    "#E64B35",  #1
    "#00A087",  #2
    "#3C5488",  #3
    "#FFA70F",  #4
"#208BB5",  #35
    "#ED7287",  #5
    "#6AC882",  #6
    "#FF7C2B",  #7
    "#6A5A96",  #8
"#D3C518",  #28
    "#23A250",  #9
    "#5478D1",  #10
    "#A68026",  #11
    "#DF5C93",  #12
    "#41ABC4",  #13
    "#A04B4B",  #14
    "#7F7F7F",  #15
    "#995000",  #16
    "#925099",  #17
    "#E07A00",  #18
    "#E22B2B",  #19
    "#A5D324",  #20
    "#813BC9",  #21
    "#00CC65",  #22
    "#C91C77",  #23
    "#2A437C",  #24
    "#326924",  #25
    "#F15A22",  #26
    "#0069B5",  #27
    "#D3C518",  #28
    "#603813",  #29
    "#00AFA6",  #30
    "#C898AE",  #31
    "#A19700",  #32
    "#6B278B",  #33
    "#E07F48",  #34
    "#208BB5",  #35
    "#68111E",  #36
    "#A35563",  #37
    "#335C64",  #38
    "#E3BC3D",  #39
    "#3E6549",  #40
]

to_rgba = mpl.colors.ColorConverter().to_rgba
PLOT_COLORS = [to_rgba(c, 0.85) for c in COLORS]
COLOR_MAP = LinearSegmentedColormap.from_list("my_cmap", ["#00A087", "#208BB5", "#3C5488"], N=256)

# https://matplotlib.org/stable/users/explain/customizing.html

mpl.rcParams['lines.linewidth'] = PLOT_SIZE / 2 #2.5

mpl.rcParams['axes.linewidth'] = 0 #PLOT_SIZE / 10
mpl.rcParams['axes.labelsize'] = 4 * PLOT_SIZE # 3
mpl.rcParams['axes.labelpad'] = 3 * PLOT_SIZE
mpl.rcParams['axes.titlesize'] = 4 * PLOT_SIZE # 3
mpl.rcParams['axes.titlepad'] = 3 * PLOT_SIZE


mpl.rcParams['xtick.labelsize'] = 4 * PLOT_SIZE # 3
mpl.rcParams['ytick.labelsize'] = 4 * PLOT_SIZE # 3
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0

mpl.rcParams['legend.fontsize'] = 4 * PLOT_SIZE # 2.5
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.edgecolor'] = "#FFFFFF"

mpl.rcParams['grid.color'] = "#E0E0E0"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PLOT_COLORS)

mpl.rcParams['figure.autolayout'] = True

#mpl.rcParams["mathtext.fontset"] = 'Libertinus Math' TODO
try:
    font_path = 'fonts/OptimaLTPro-Roman.ttf'
    fontManager.addfont(font_path)
    myfont = FontProperties(fname=font_path)
    mpl.rcParams['font.serif'] = myfont.get_name()
    mpl.rcParams['font.family'] = myfont.get_name()
except:
    try:
        font_path = '../fonts/OptimaLTPro-Roman.ttf'
        fontManager.addfont(font_path)
        myfont = FontProperties(fname=font_path)
        mpl.rcParams['font.serif'] = myfont.get_name()
        mpl.rcParams['font.family'] = myfont.get_name()
    except:
        pass


def notation(region_name):
    """
    Create an abbreation for a region name.

    Parameters
    ----------
    region_name : str
        The name of the region.
    """

    split_name = region_name.split(" ")
    split_name_except_last = split_name#[:-1]
    split_name_initials = [word[0] for word in split_name_except_last]
    split_name_initials_joined_caps = "".join(split_name_initials).upper()
    math_txt = split_name_initials_joined_caps #+ "_" + split_name[-1][0].upper()
    # return f"$\\mathrm{{{math_txt}}}$"
    return math_txt
