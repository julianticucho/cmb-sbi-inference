import torch
from ..contracts.base_step import BaseStep


class MultipoleBinningStep(BaseStep):
    """Uniformly bin a spectrum by multipole intervals.

    Provides :meth:`get_lmin` and :meth:`get_lmax` returning the inclusive
    lower and upper ℓ for each bin.
    """

    def __init__(self, l_min: int, l_max: int, n_bins: int = 200):
        super().__init__("MultipoleBinningStep")
        self.l_min = l_min
        self.l_max = l_max
        self.n_bins = n_bins
        self._compute_bins()

    def _compute_bins(self):
        """Compute inclusive lmin/lmax tensors for ``n_bins`` uniform bins.

        The total number of multipoles is ``l_max - l_min + 1``. We distribute the
        remainder evenly among the first bins so that bin sizes differ by at most one.
        """
        total = self.l_max - self.l_min + 1
        base, rem = divmod(total, self.n_bins)
        edges = []
        cur = self.l_min
        for i in range(self.n_bins):
            size = base + (1 if i < rem else 0)
            start = cur
            end = cur + size - 1
            edges.append((start, end))
            cur = end + 1
        # store as tensors
        self.lmin = torch.tensor([s for s, _ in edges], dtype=torch.long)
        self.lmax = torch.tensor([e for _, e in edges], dtype=torch.long)

    def get_lmin(self) -> torch.Tensor:
        """Return lower ℓ (inclusive) for each bin."""
        return self.lmin

    def get_lmax(self) -> torch.Tensor:
        """Return upper ℓ (inclusive) for each bin."""
        return self.lmax

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Bin a 1‑D spectrum ``x`` using the pre‑computed edges.

        The input length must match the full ℓ range ``l_max - l_min + 1``.
        """
        if x.ndim != 1:
            raise ValueError("Input must be a 1‑D tensor")
        expected_len = self.l_max - self.l_min + 1
        if x.shape[0] != expected_len:
            raise ValueError(
                f"Input length {x.shape[0]} does not match expected ℓ range size {expected_len}."
            )
        binned = []
        for lo, hi in zip(self.lmin, self.lmax):
            start = int(lo.item() - self.l_min)
            end = int(hi.item() - self.l_min) + 1  # inclusive slice
            binned.append(x[start:end].mean())
        return torch.stack(binned)
