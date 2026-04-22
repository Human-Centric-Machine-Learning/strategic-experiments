"""Plotting utilities: LaTeX-compatible matplotlib styling and figure sizing."""

import matplotlib
import matplotlib.pyplot as plt


def get_fig_dim(width, fraction=1, aspect_ratio=None):
    """Return figure dimensions in inches that avoid LaTeX scaling artifacts.

    Parameters
    ----------
    width : float
        Document textwidth or columnwidth in points.
    fraction : float, optional
        Fraction of ``width`` that the figure should occupy.
    aspect_ratio : float, optional
        Width-to-height ratio. Defaults to the golden ratio.

    Returns
    -------
    tuple
        ``(width_in_inches, height_in_inches)``.
    """
    fig_width_pt  = width * fraction
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        aspect_ratio = (1 + 5 ** 0.5) / 2  # golden ratio

    fig_width_in  = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in / aspect_ratio
    return (fig_width_in, fig_height_in)


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10,
             small_font_size=None, usetex=True):
    """Configure matplotlib's RC params for LaTeX-compatible plotting.

    Call this before creating a figure.

    Parameters
    ----------
    font_serif : str, optional
        Serif font family.
    mathtext_font : str, optional
        Math-text font set.
    font_size : int, optional
        Base font size.
    small_font_size : int, optional
        Size for legends / tick labels. Defaults to ``font_size``.
    usetex : bool, optional
        If True, render text with LaTeX.
    """
    if small_font_size is None:
        small_font_size = font_size

    params = {
        'backend':              'ps',
        'text.latex.preamble':  '\\usepackage{gensymb} \\usepackage{bm}',
        'axes.labelsize':       font_size,
        'axes.titlesize':       font_size,
        'font.size':            font_size,
        'legend.fontsize':      small_font_size,
        'legend.title_fontsize': small_font_size,
        'xtick.labelsize':      small_font_size,
        'ytick.labelsize':      small_font_size,
        'text.usetex':          usetex,
        'font.family':          'serif',
        'font.serif':           font_serif,
        'mathtext.fontset':     mathtext_font,
    }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)
