import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional
from matplotlib.patches import Rectangle

# generamos un gráfico que reciba un cl binneado, dos tensores ell_min y ell_max que representen los bordes de cada bin,
# los errores de cada bin, y un cl no binneado, y a partir de eso genera un gráfico del tal forma que grafiques el
# cl no bineado y el binneado lo grafiques como bloques transparentes (rectangulos) con el error como barra de error
def plot_cl(
    cl: torch.Tensor,
    ell: torch.Tensor,
    ell_min: torch.Tensor,
    ell_max: torch.Tensor,
    cl_err: torch.Tensor,
    cl_clean: torch.Tensor,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    # quiero que el ell_unbinned parta de 2
    ell_unbinned = torch.arange(len(cl_clean)) + 2*torch.ones_like(cl_clean)
    plt.plot(ell_unbinned, cl_clean, color='C1', label='camb')
    ax.scatter(ell, cl, color='C0', s=0.1, zorder=3, label='obs')
    for i in range(len(ell)):
        x0 = ell_min[i].item()
        x1 = ell_max[i].item()
        y0 = (cl[i] - cl_err[i]).item()
        y1 = (cl[i] + cl_err[i]).item()
        ax.fill_between(
            [x0, x1],
            [y0, y0],
            [y1, y1],
            color='C0',
            alpha=0.3,
            linewidth=0
        )
    ax.set_xlabel(r'$\ell$', fontsize=12)
    ax.set_ylabel(r'$C_\ell$', fontsize=12)
    ax.legend()
    return fig

def plot_cmb_map(
    map_data: torch.Tensor, 
    c_min: float = -400, 
    c_max: float = 400, 
    x_width: Optional[float] = None, 
    y_width: Optional[float] = None
) -> plt.Figure:
    map_np = map_data.numpy() if not isinstance(map_data, np.ndarray) else map_data
    print("map mean:", np.mean(map_np), "map rms:", np.std(map_np))

    if x_width is None:
        x_width = (2**10) * 0.5 / 60
    if y_width is None:
        y_width = (2**10) * 0.5 / 60

    plt.gcf().set_size_inches(x_width, y_width)
    im = plt.imshow(map_np, interpolation='bilinear', origin='lower', cmap=cm.RdBu_r)
    im.set_clim(c_min, c_max)
    im.set_extent([0, x_width, 0, y_width])
    plt.ylabel(r"Angle $[^\circ]$")
    plt.xlabel(r"Angle $[^\circ]$")

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Temperature [$\mu $K]")

    fig = plt.gcf()
    return fig

