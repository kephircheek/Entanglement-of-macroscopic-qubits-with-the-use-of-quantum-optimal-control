import copy
import itertools
import json
import math
from dataclasses import asdict, dataclass

import bec
import joblib
import numpy as np
from qutip import Qobj, fock, tensor
from tqdm import tqdm


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
        return projection_on_qubit_state(q, (k[j - 1] - k[j + 1]) * t, n)
    return (
        np.exp(1j * k[j] * phase)
        * math.sqrt(comb(n, k[j]))
        * projection_on_z_fock_state(q, k[j], n)
    )


def k_state(i: int, k: int, m: int, n: int):
    return bec.fock_state_constructor(bec.BEC_Qubits.init_default(n, 0), m, i=i, k=k)


def f_state(t: float, q: tuple[int], m: int, n: int, focked=True):
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
            f_state_coeff(t, q, k, m, n)
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


def f_state_coeff(t: float, q: tuple[int], k: tuple[int], m: int, n: int):
    coeff = math.prod((omega(t, q[j], j, k, 0, n) for j in range(1, m - 1)))
    coeff *= math.sqrt(comb(n, k[0]))
    coeff *= math.sqrt(comb(n, k[-1]))
    if m % 2 == 0:
        coeff *= 1 / math.sqrt(2) ** n
        coeff *= np.exp(1j * k[-2] * t * k[-1])
    return coeff


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
