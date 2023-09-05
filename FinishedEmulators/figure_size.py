# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, dims=[1, 1], fraction=1, golden_ratio=True):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    dims: array
            Array of figure dimensions [panels in y, panels in x]
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    golden_ration: boolean 
            Set golden ratio aspect

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if golden_ratio:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        golden_ratio = (5**.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (dims[0] / dims[1])
    else:
        fig_width_in = fig_width_pt * inches_per_pt
        fig_height_in = fig_width_in / dims[0] * dims[1]


    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

