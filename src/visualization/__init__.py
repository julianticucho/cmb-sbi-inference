import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'high-vis'])
plt.rcParams['figure.dpi'] = 300

from .diagnostic import plot_ppc, plot_hpd, plot_sbc, plot_tarp
from .inference import correlation_matrix
from .simulation import plot_cmb_map

__all__ = [
    'plot_ppc',
    'plot_hpd', 
    'plot_sbc',
    'plot_tarp',
    'correlation_matrix',
    'plot_cmb_map'
]
