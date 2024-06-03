import time
import warnings

import qutip
from qutip.ui.progressbar import BaseProgressBar

qutip_maj_ver = int(qutip.__version__[0])
if qutip_maj_ver >= 5:
    warnings.warn("use native: from qutip.ui.progressbar import TqdmProgressBar")


class TqdmProgressBar(BaseProgressBar):
    """
    A progress bar using tqdm module
    """

    def __init__(self, iterations=0, chunk_size=10):
        from tqdm.auto import tqdm

        self.tqdm = tqdm

    def start(self, iterations, **kwargs):
        self.pbar = self.tqdm(total=iterations, ncols=80, **kwargs)
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self, n=None):
        self.pbar.update()

    def finished(self):
        self.pbar.close()
        self.t_done = time.time()
