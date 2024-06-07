import copy
import itertools
import json
import math
from dataclasses import asdict, dataclass

import joblib
import numpy as np
from scipy.integrate import quad
from qutip import Qobj, fock, tensor, ket2dm
from qutip.entropy import negativity
from tqdm import tqdm

import bec


def comb(n: int, k: int):
    return math.comb(n, k)


def projection_on_qubit_state(q: int, alpha: float, n: int):
    """See Eq. 20."""
    return (
        1j ** (n - q)
        * np.exp(1j * n * alpha / 2)
        * math.sqrt(comb(n, q))
        * math.cos(alpha / 2) ** q
        * math.sin(alpha / 2) ** (n - q)
    )


def _projection_on_z_fock_state_lm(l: int, m: int, q: int, k: int, n: int):
    return (
        (-1) ** (n - q - m)
        * comb(q, l)
        * comb(n - q, m)
        * math.sqrt(
            math.factorial(l + m)
            * math.factorial(n - l - m)
            / math.factorial(q)
            / math.factorial(n - q)
            / 2**n
        )
    )


def projection_on_z_fock_state(q: int, k: int, n: int):
    """See Eq. 21."""
    if None in (q, k, n):
        raise TypeError("expected integer, not 'None'")

    return sum(
        _projection_on_z_fock_state_lm(l, m, q, k, n)
        for l, m in itertools.product(range(q + 1), range(n - q + 1))
        if k == (l + m)
    )


def omega(t: float, q: int, j: int, k: tuple[int], phase: float, n: int):
    if j % 2 == 1:
        return projection_on_qubit_state(q, (k[j - 1] - k[j + 1]) * t + phase, n)
    return (
        np.exp(1j * k[j] * phase)
        * math.sqrt(comb(n, k[j]))
        * projection_on_z_fock_state(q, k[j], n)
    )


def k_state(i: int, k: int, m: int, n: int):
    return bec.fock_state_constructor(bec.BEC_Qubits.init_default(n, 0), m, i=i, k=k)


def f_state(t: float, q: tuple[int], m: int, n: int, phase: float = 0, focked=True):
    """
    Return final state, see eq. 11 and eq. 12.

    Args:
        t: time of evolution
        k: measured values
        p: some project state number
        m: number of qubits in chain
        n: number of bosons in qubit
    """

    if len(q) < (m - 2):
        raise ValueError("too few measured sites")
    q = (None,) + tuple(q) + (None,)

    if m % 2 == 0:
        norm = 2 ** (m * n / 4)
    else:
        norm = 2 ** ((m + 1) * n / 4)

    fock_range = range(n + 1)
    k_ranges = [fock_range if i % 2 == 0 else [None] for i in range(m)]
    if m % 2 == 0:
        k_ranges[-1] = fock_range  # reveal coherent state via fock states
    k_sets = list(itertools.product(*k_ranges))

    model = None
    if not focked:
        model = bec.BEC_Qubits.init_default(n, 0)

    return (
        sum(
            f_state_coeff(t, q, k, m, n, phase)
            * (
                tensor(fock(n + 1, k[0]), fock(n + 1, k[m - 1]))
                if focked
                else (
                    bec.fock_state_constructor(model, n=2, i=0, k=k[0])
                    * bec.fock_state_constructor(model, n=2, i=1, k=k[m - 1])
                    * bec.vacuum_state(model)
                )
            )
            for k in k_sets
        )
        / norm
    )


def f_state_norm(t: float, q: tuple[int], m: int, n: int, phase: float = 0):
    if len(q) < (m - 2):
        raise ValueError("too few measured sites")
    q = (None,) + tuple(q) + (None,)

    if m % 2 == 0:
        norm = 2 ** (m * n / 4)
    else:
        norm = 2 ** ((m + 1) * n / 4)

    fock_range = range(n + 1)
    s = 0
    for k1, km in itertools.product(fock_range, fock_range):
        k_ranges = (
            [[k1]]
            + [fock_range if i % 2 == 0 else [None] for i in range(1, m - 1)]
            + [[km]]
        )
        k_sets = list(itertools.product(*k_ranges))
        s += abs(sum(f_state_coeff(t, q, k, m, n, phase) for k in k_sets)) ** 2
    return np.sqrt(s) / norm


def f_state_decoherence_part(k, q, n, phase):
    x = math.prod(1 if k_ is None else (2 * k_ - n) for k_ in k)
    return np.exp(1j * x * phase)


def f_state_coeff(
    t: float, q: tuple[int], k: tuple[int], m: int, n: int, phase: float = 0
):
    coeff = math.prod((omega(t, q[j], j, k, 0, n) for j in range(1, m - 1)))
    coeff *= math.sqrt(comb(n, k[0]))
    coeff *= math.sqrt(comb(n, k[-1]))
    if phase != 0:
        # See eq. 32 in Alexey N Pyrkov and Tim Byrnes 2013 New J. Phys. 15 093019
        # and eq. 25 in Full-Bloch-sphere teleportation of spinor Bose-Einstein condensates and spin ensembles.
        coeff *= f_state_decoherence_part(k, q, n, phase)
    if m % 2 == 0:
        coeff *= 1 / math.sqrt(2) ** n
        coeff *= np.exp(1j * k[-2] * t * k[-1])
    return coeff


def f_state_fid_respect_m2(t: float, q: tuple[int], m: int, n: int, phase: float = 0):
    """
    phase : float
        See eq. 32 in Alexey N Pyrkov and Tim Byrnes 2013 New J. Phys. 15 093019
        and eq. 25 in Full-Bloch-sphere teleportation of spinor Bose-Einstein condensates and spin ensembles.
    """
    if len(q) < (m - 2):
        raise ValueError("too few measured sites")
    q = (None,) + tuple(q) + (None,)
    odd = m % 2 == 1
    norm = None
    if odd:
        norm = 2 ** (n * (m + 5) / 2)
    else:
        norm = 2 ** (n * (m + 2) / 2)

    fock_range = range(n + 1)
    k_ranges = [fock_range if i % 2 == 0 else [None] for i in range(m)]
    if m % 2 == 0 and phase != 0:
        k_ranges[-1] = fock_range
    k_sets = list(itertools.product(*k_ranges))

    return (
        np.abs(
            np.sum(
                comb(n, k[0])
                * (1 if not odd and phase == 0 else comb(n, k[-1]))
                * math.prod((omega(t, q[j], j, k, 0, n) for j in range(1, m - 1)))
                * (
                    (
                        np.exp(-1j * k[0] * k[-1] * t)
                        * (
                            1
                            if phase == 0
                            else f_state_decoherence_part(k, q, n, phase)
                        )
                    )
                    if odd
                    # else np.cos((k[0] - k[-2]) * t / 2) ** n
                    else (
                        ((np.exp(1j * (k[-2] - k[0]) * t) + 1) / 2) ** n
                        if phase == 0
                        else (
                            np.exp(1j * k[-1] * (k[-2] - k[0]) * t)
                            * f_state_decoherence_part(k, q, n, phase)
                            / 2**n
                        )
                    )
                )
                for k in k_sets
            )
        )
        ** 2
        / norm
    )


def negativity_of_dephased_state(
    t: float, q: tuple[int], m: int, n: int, gamma: float = 1, quad_kwargs: dict = {}
):
    if t == 0:
        return 0, 0

    def fun(phase):
        s = f_state(t, q, m, n, phase=phase)
        s /= f_state_norm(t, q, m, n, phase=phase)
        rho = ket2dm(s)
        neg = negativity(rho, 0, method="eigenvalues")
        return np.exp(-(phase**2) / (2 * gamma * t)) * neg

    I, e = quad(fun, -np.inf, np.inf, **quad_kwargs)
    return I / np.sqrt(2 * np.pi * gamma * t), e


def fid_f_state_dephased_respect_m2(
    t: float, q: tuple[int], m: int, n: int, gamma: float = 1
):
    if t == 0:
        return 1, 0

    def fun(phase):
        return (
            np.exp(-(phase**2) / (2 * gamma * t))
            * f_state_fid_respect_m2(t, q, m, n, phase=phase)
            / f_state_norm(t, q, m, n, phase=phase) ** 2
        )

    I, e = quad(fun, -np.inf, np.inf)
    return I / np.sqrt(2 * np.pi * gamma * t), e


def entropy_vn(m, base=2):
    if base != 2:
        raise ValueError("invalid base: {base} != 2")
    eigvals = [v.real for v in np.linalg.eigvals(m)]
    eigvals_sum = sum(eigvals)
    return -sum(
        l / eigvals_sum * math.log2(l / eigvals_sum)
        for l in eigvals
        if eigvals_sum > 1e-13 and l > 1e-13
    )


@dataclass(frozen=True)
class PropagateEntanglementTask:
    n_bosons: int
    n_sites: int
    t_span: tuple[float]
    k_measured: list[int] = None
    projection: int = None

    def __post_init__(self):
        if self.k_measured is None:
            object.__setattr__(
                self,
                "k_measured",
                (self.n_bosons,) * (self.n_sites // 2 - ((self.n_sites + 1) % 2)),
            )
        if self.projection is None and self.n_sites > 3:
            object.__setattr__(self, "projection", self.n_bosons)

    @property
    def t_list(self):
        return list(np.linspace(*self.t_span))

    @property
    def label(self):
        return (
            f"n{self.n_bosons}m{self.n_sites}"
            + (
                f"k{','.join(map(str, self.k_measured))}"
                if len(self.k_measured) > 0
                else ""
            )
            + (f"p{self.projection}" if self.projection is not None else "")
            + f"t{self.t_span[0]};{self.t_span[1]};{self.t_span[2]}"
        )

    def run(self, verbose=True, n_jobs=-2, ncols=80):
        states = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(rho_b)(
                t,
                p=self.projection,
                k=self.k_measured,
                m=self.n_sites,
                n=self.n_bosons,
            )
            for t in tqdm(
                self.t_list, postfix=self.label, disable=not verbose, ncols=ncols
            )
        )
        return PropagateEntanglementResult(task=self, t_list=self.t_list, states=states)


@dataclass(frozen=True)
class PropagateEntanglementResult:
    task: PropagateEntanglementTask
    t_list: list[float]
    states: list[list[list[complex]]]

    def entropies(self, verbose=False):
        return [entropy_vn(s) for s in tqdm(self.states, disable=not verbose)]

    def to_Qobj(self, s: list[list[complex]]):
        from qutip import Qobj

        return Qobj(s, dims=[[self.task.n_bosons + 1]] * 2)

    def reveal_state(self, indx: int):
        rho = 0
        model = bec.BEC_Qubits.init_default(self.task.n_bosons, 0)
        state = self.states[indx]
        for km1, km2 in itertools.combinations_with_replacement(
            range(self.task.n_bosons + 1), 2
        ):
            elem = state[km1][km2]
            km1_vec = bec.fock_state_constructor(
                model, n=1, i=0, k=km1
            ) * bec.vacuum_state(model, n=1)
            km2_vec = bec.fock_state_constructor(
                model, n=1, i=0, k=km2
            ) * bec.vacuum_state(model, n=1)
            rho += km1_vec * km2_vec.dag() * elem
            if km1 != km2:
                rho += km2_vec * km1_vec.dag() * elem.conjugate()
        return rho / 2**self.task.n_bosons  # `... / 2^n` added for Tr{rho} == 1

    @classmethod
    def init_form_dict(cls, dct):
        dct = copy.deepcopy(dct)
        task = PropagateEntanglementTask(**dct.pop("task"))
        return cls(task=task, **dct)

    @staticmethod
    def json_default(obj):
        if isinstance(obj, complex):
            return {"__complex__": [obj.real, obj.imag]}
        if isinstance(obj, tuple):
            return {"__tuple__": obj}
        return obj

    def dump(self, f):
        dct = asdict(self)
        return json.dump(dct, f, default=self.json_default)

    @staticmethod
    def json_object_hook(dct):
        if v := dct.get("__complex__"):
            return complex(*v)
        if v := dct.get("__tuple__"):
            return tuple(v)
        return dct

    @classmethod
    def load(cls, f):
        return cls.init_form_dict(json.load(f, object_hook=cls.json_object_hook))
