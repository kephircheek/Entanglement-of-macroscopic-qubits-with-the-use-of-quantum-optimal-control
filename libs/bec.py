"""

References

1. Alexey N Pyrkov and Tim Byrnes. New J. Phys. 15 093019 (2013)
2. Colombe Y et al. Nature 450, 272 (2007)
3. Daniel Rosseau Qianqian Ha and Tim Byrnes, Phys.Rev.A 90, 052315 (2014)
"""
import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip


@dataclass(frozen=True)
class BEC_Qubits:
    """
    Parameters
    ----------

    coupling_strength :
        Coupling strength of single quantized mode coupled to the transition.

    transition_ampl: float
        Transition amplitude of off-resonant laser.

    transition_freq : float
        Transition frequency.

    resonance_freq : float
        Cavity photon resonance.

    phase: float
        Phase due to propagation of the field through the fiber.

    """

    n_bosons: float
    coupling_strength: float
    transition_ampl: float
    transition_freq: float
    resonance_freq: float
    phase: float
    excitation_level: bool = False
    communication_line: bool = False

    @property
    def G(self):
        "Atom-cavity mode coupling"
        return self.coupling_strength

    @property
    def g(self):
        "Laser coupling"
        return self.transition_ampl

    @property
    def omega(self):
        return self.resonance_freq

    @property
    def omega0(self):
        return self.transition_freq

    @property
    def phi(self):
        return self.phase

    @property
    def Omega(self):
        return self.coupling_strength**2 * self.g**2 / 2 / self.delta**3

    @property
    def delta(self):
        """Detuning between the transition to the excited state and the cavity photon resonance."""
        return self.transition_freq - self.resonance_freq

    @property
    def delta_l(self):
        """Detuning between the laser transition and the b <-> e transition. See [3]."""
        warnings.warn("Means \delta_l = \delta_c = \delta")
        return self.delta

    @property
    def delta_c(self):
        """Detuning between the cavity and the b <-> e transition. See [3]."""
        return self.delta

    @classmethod
    def init_alexey2003(
        cls,
        n_bosons,
        phase,
        single_coupling_strength=1.35e6,
        transition_freq=1e7,
        detuning_param: float = 1,
        excitation_level=False,
    ):
        """
        See Sec. '4. Estimated gate times' in [1] and [2] for details.

        Parameters
        ----------
        n_bosons :
            Number of bosons

        single_coupling_strength :
            Single atom cavity coupling strength
        d :
            Dimensionless detuning parameter.
        """

        coupling_strength = math.sqrt(n_bosons) * single_coupling_strength
        transition_ampl = coupling_strength
        delta = detuning_param * single_coupling_strength * n_bosons
        resonance_freq = transition_freq - delta
        return cls(
            n_bosons=n_bosons,
            coupling_strength=coupling_strength,
            transition_ampl=transition_ampl,
            transition_freq=transition_freq,
            resonance_freq=resonance_freq,
            phase=phase,
            excitation_level=excitation_level,
        )

    @classmethod
    def init_default(cls, n_bosons, phase, excitation_level=False):
        coupling_strength = 1
        delta = 10
        transition_freq = 11
        resonance_freq = transition_freq - delta
        return cls(
            n_bosons=n_bosons,
            coupling_strength=coupling_strength,
            transition_ampl=coupling_strength,
            transition_freq=transition_freq,
            resonance_freq=resonance_freq,
            phase=phase,
            excitation_level=excitation_level,
        )

    @property
    def sublevels(self):
        if self.excitation_level:
            return 3  # Means only 'a', 'b' and 'e'
        return 2  # Means only 'a' and 'b'

    @property
    def communication_line_levels(self):
        return 2


def _build_entire_space(operator, model, n, k, kind):
    """
    Parameters
    ----------
    operator: callable
    Target operator constructor
    model: BEC_Qubits
    Model
    n : int
    Number of qubits.
    k : int
    Qubit number.
    kind : int
    Sublevel number.
    """
    if k is None and n > 1:
        raise NotImplementedError(f"n={n}; k={k}")

    if k is None and n == 1:
        k = 0

    if k > (n - 1):
        raise ValueError(
            f"qubit number out of range {k} > {n - 1}. Counting starts with zero."
        )

    if (k > 1 or n > 2) and model.communication_line:
        raise ValueError("only one or two qubits for communication_line")

    n_qubit_kinds = model.sublevels
    qubits_spaces = [[qutip.identity(model.n_bosons + 1)] * n_qubit_kinds] * n
    communication_line_spaces = (
        [qutip.identity(model.communication_line_levels)]
        if model.communication_line
        else []
    )
    if kind != "c":
        kind_i = {"a": 0, "b": 1, "e": 2}[kind]
        qubit_spaces = (
            kind_i * [qutip.identity(model.n_bosons + 1)]
            + [operator(model.n_bosons + 1)]
            + (n_qubit_kinds - kind_i - 1) * [qutip.identity(model.n_bosons + 1)]
        )
        qubits_spaces[k] = qubit_spaces
    else:
        communication_line_spaces = [operator(model.communication_line_levels)]

    return qutip.tensor(
        *(
            qubits_spaces[0]
            + sum((communication_line_spaces + qs for qs in qubits_spaces[1:]), [])
        )
    )


def _destroy(model: BEC_Qubits, n, k, kind: Literal["a", "b", "e", "c"]):
    """
    Create `k`-th destroy operator of `kind` in full model space of `n` particles.
    Indexing starts from zero.

    Examples:
        >>> _destory(model, n=2, k=0, 'a')  # a_1
        >>> _destory(model, n=2, k=0, 'b')  # b_2
    """
    return _build_entire_space(qutip.destroy, model, n, k, kind=kind)


def a(model, n=1, k=None):
    return _destroy(model, n, k, kind="a")


def b(model, n=1, k=None):
    return _destroy(model, n, k, kind="b")


def e(model, n=1, k=None):
    if model.excitation_level is False:
        raise ValueError("no excitation state in model")
    return _destroy(model, n, k, kind="e")


def c(model, n=1, k=None):
    if k is not None and k != 0:
        raise NotImplementedError("only for single communication line")
    if model.communication_line is False:
        raise ValueError("no communication line in model")
    return _destroy(model, n, k, kind="c")


def na(model, n=1, k=None):
    a_ = a(model, n, k)
    return a_.dag() * a_


def sz(model, n=1, k=None):
    a_ = a(model, n, k)
    b_ = b(model, n, k)
    return a_.dag() * a_ - b_.dag() * b_


def sx(model, n=1, k=None):
    a_ = a(model, n, k)
    b_ = b(model, n, k)
    return a_.dag() * b_ + b_.dag() * a_


def sy(model, n=1, k=None):
    a_ = a(model, n, k)
    b_ = b(model, n, k)
    return 1j * (b_.dag() * a_ - a_.dag() * b_)


def h_eff_total(model, n=2):
    """Return total effective Hamiltonian. See (13) in [1]"""
    if n != 2:
        raise NotImplementedError("only qubit pair")

    omega = (model.n_bosons + 2) * model.Omega * math.cos(
        model.phase
    ) - model.g**2 * model.omega0 / 4 / model.delta
    d = model.Omega * math.cos(model.phase)
    print(f"omega = {omega:.3e}; d = {d:.3e}")
    return -d * sz(model, n, 0) * sz(model, n, 1) + omega * (
        sz(model, n, 0) + sz(model, n, 1)
    )


def h_eff_eq9(model, n=2):
    """Return effective Hamiltonian edition  See (9) in [1]"""
    if n != 2:
        raise NotImplementedError("only qubit pair")
    zz_const = -model.Omega * np.cos(model.phase)
    z_squared_const = model.Omega / 2
    z_const = (
        model.Omega
        * (model.n_bosons * (np.cos(model.phase) - 1) + 2 * np.cos(model.phi))
        - model.g**2 * model.omega0 / model.delta / 2
    )
    return (
        zz_const * sz(model, n=n, k=0) * sz(model, n=n, k=1)
        + z_squared_const
        * (
            sz(model, n=n, k=0) * sz(model, n=n, k=0)
            + sz(model, n=n, k=1) * sz(model, n=n, k=1)
        )
        + z_const * (sz(model, n=n, k=0) + sz(model, n=n, k=1))
    )


def h_eff_edition1(model, n=2):
    """Return effective Hamiltonian. See 1 edition of H_eff in page 6 of Alexey notes."""
    if n != 2:
        raise NotImplementedError("only qubit pair")

    p1 = (
        model.G**2
        * model.g**2
        / model.delta_c
        / model.delta_l**2
        * 2
        * np.cos(model.phi)
        * b(model, n=n, k=0).dag()
        * b(model, n=n, k=1)
        * b(model, n=n, k=1).dag()
        * b(model, n=n, k=0)
    )

    p2 = (
        model.g**2
        / model.delta_l
        * (
            b(model, n=n, k=0).dag() * b(model, n=n, k=0)
            + b(model, n=n, k=1).dag() * b(model, n=n, k=1)
        )
    )

    return p1 + p2


def h_eff_edition3(model, n=2):
    """Return effective Hamiltonian. See 3 edition of H_eff in page 6 of Alexey notes."""
    if n != 2:
        raise NotImplementedError("only qubit pair")

    return (
        (
            model.G**2
            * model.g**2
            / model.delta_c
            / model.delta_l**2
            * 2
            * np.cos(model.phase)
            * b(model, n=n, k=0).dag()
            * b(model, n=n, k=0)
            * b(model, n=n, k=1).dag()
            * b(model, n=n, k=1)
        )
        + (
            model.g**2
            / model.delta_l
            * (
                1
                + 2 * model.G**2 / model.delta_c / model.delta_l * np.cos(model.phase)
            )
            * b(model, n=n, k=0).dag()
            * b(model, n=n, k=0)
        )
        + (model.g**2 / model.delta_l * b(model, n=2, k=1).dag() * b(model, n=2, k=1))
    )


def hamiltonian_eff(model, n=2, zeeman=True, quadratic=True):
    """Return effective Hamiltonian. See (6) in [3]."""
    if n != 2:
        raise NotImplementedError("only qubit pair")

    ham = 0
    zz = -model.Omega * (sz(model, n=n, k=0) * sz(model, n=n, k=1))
    ham += zz

    if quadratic:
        ham += -model.Omega * (
            sz(model, n=n, k=0) * sz(model, n=n, k=0)
            + sz(model, n=n, k=1) * sz(model, n=n, k=1)
        )

    if zeeman:
        omega_ = model.g**2 / 2 / model.delta_l + model.Omega * model.n_bosons / 2
        ham += omega_ * (sz(model, n=n, k=0) + sz(model, n=n, k=1))

    return ham


def hamiltonian_ad(model, n=2):
    """Return effective Hamiltonian. See (5) in [3]."""
    return -2 * model.G**2 * model.g**2 / model.delta_c / model.delta_l**2 * (
        b(model, n, k=0).dag() * b(model, n, k=0)
        + b(model, n, k=1).dag() * b(model, n, k=1)
    ) * (
        b(model, n, k=0).dag() * b(model, n, k=0)
        + b(model, n, k=1).dag() * b(model, n, k=1)
    ) - model.g**2 / model.delta_l * (
        b(model, n, k=0).dag() * b(model, n, k=0)
        + b(model, n, k=1).dag() * b(model, n, k=1)
    )


def hzz(model, n=2):
    if n != 2:
        raise NotImplementedError("only qubit pair")

    return model.Omega * sz(model, n, 0) * sz(model, n, 1)


def h_int_approx(model, n=2):
    """The approx of interaction Hamiltonian from Alexey handwriting notes. See last line in page 5."""
    if model.phi != 0:
        raise ValueError(f"hardcoded for \phi = 0, not {model.phi}")

    if n != 2:
        raise ValueError(f"hardcoded for n = 2, not {n}")

    return (
        model.G**2
        / model.delta_c
        * (
            b(model, n=2, k=0)
            * e(model, n=2, k=0).dag()
            * b(model, n=2, k=1).dag()
            * e(model, n=2, k=1)
            + b(model, n=2, k=0).dag()
            * e(model, n=2, k=0)
            * b(model, n=2, k=1)
            * e(model, n=2, k=1).dag()
        )
        + model.delta_l
        * (
            e(model, n=2, k=0).dag() * e(model, n=2, k=0)
            + e(model, n=2, k=1).dag() * e(model, n=2, k=1)
        )
        # since there are many cases delta_c = delta_l it almost always is zero
        + (model.delta_l - model.delta_c)
        * c(model, n=2, k=0).dag()
        * c(model, n=2, k=0)
    )


def h_int(model, n=2):
    """The interaction Hamiltonian combined from H_CQED and H_f. See eq. (5) in [1]."""
    if n != 2:
        raise NotImplementedError("only qubit pair")

    return (
        model.G
        / np.sqrt(2)
        * (
            e(model, n=n, k=0).dag() * b(model, n=n, k=0) * c(model, n=n, k=0)
            - np.exp(1j * model.phase)
            * e(model, n=n, k=1).dag()
            * b(model, n=n, k=1)
            * c(model, n=n, k=0)
            + (
                c(model, n=n, k=0).dag() * b(model, n=n, k=0).dag() * e(model, n=n, k=0)
                - np.exp(-1j * model.phase)
                * c(model, n=n, k=0).dag()
                * b(model, n=n, k=1).dag()
                * e(model, n=n, k=1)
            )
        )
        + model.omega0
        * (
            e(model, n=n, k=0).dag() * e(model, n=n, k=0)
            + e(model, n=n, k=1).dag() * e(model, n=n, k=1)
        )
        + model.omega * c(model, n=n, k=0).dag() * c(model, n=n, k=0)
    )


def hamiltonian_coupling(model, n=2):
    """Return Hamiltonian coupling the BECs to the common mode. See Eq. (3) in [3]."""
    delta_c = model.delta_c
    return delta_c * (c(model, n=n, k=0).dag() * c(model, n=n, k=0)) + model.G * (
        e(model, n=n, k=0).dag() * b(model, n=n, k=0) * c(model, n=n, k=0)
        + e(model, n=n, k=1).dag() * b(model, n=n, k=1) * c(model, n=n, k=0)
        + b(model, n=n, k=0).dag() * e(model, n=n, k=0) * c(model, n=n, k=0).dag()
        + b(model, n=n, k=1).dag() * e(model, n=n, k=1) * c(model, n=n, k=0).dag()
    )


def hamiltonian_laser_field(model, n=2):
    """Return Hamiltonian of controllable laser field. See Eq. (4) in [3]."""
    delta_l = model.delta_l
    return model.g * (
        e(model, n=n, k=0).dag() * b(model, n=n, k=0)
        + e(model, n=n, k=1).dag() * b(model, n=n, k=1)
        + b(model, n=n, k=0).dag() * e(model, n=n, k=0)
        + b(model, n=n, k=1).dag() * e(model, n=n, k=1)
    )
    +model.delta_l * (
        e(model, n=n, k=0).dag() * e(model, n=n, k=0)
        + e(model, n=n, k=1).dag() * e(model, n=n, k=1)
    )


def vacuum_state(model, n=2):
    return qutip.tensor(
        *(
            model.sublevels * [qutip.fock(model.n_bosons + 1, 0)]
            + (
                [qutip.fock(model.communication_line_levels, 0)]
                if model.communication_line
                else []
            )
            + (n - 1) * model.sublevels * [qutip.fock(model.n_bosons + 1, 0)]
        )
    )


def coherent_state_constructor(
    model, n, k, alpha=1 / math.sqrt(2), beta=1 / math.sqrt(2)
):
    return (
        1
        / math.sqrt(math.factorial(model.n_bosons))
        * (alpha * a(model, n, k).dag() + beta * b(model, n, k).dag()) ** model.n_bosons
    )


def epr_plus_state_constructor(model, n=2):
    if n != 2:
        raise ValueError("wrong number of qubits: {n} != 2")

    norm = math.sqrt(model.n_bosons + 1)
    return (
        sum(
            fock_state_constructor(model, n, i=0, k=k)
            * fock_state_constructor(model, n, i=1, k=k)
            for k in range(model.n_bosons + 1)
        )
        / norm
    )


def fock_state_constructor(model, n, i, k=0):
    """Return operator to create `k`-th eigenstate of Sz,i (Fock states) for `i`-th qubit of `n` qubits from vacuum state."""
    norm = math.sqrt(math.factorial(k) * math.factorial(model.n_bosons - k))
    return (
        1
        / norm
        * a(model, n, i).dag() ** k
        * b(model, n, i).dag() ** (model.n_bosons - k)
    )


def fock_x_state_constructor(model, n, i, k=0):
    """Return operator to create `k`-th eigenstate of Sx,i (Fock states in X basis) for `i`-th qubit of `n` qubits from vacuum state."""
    norm = math.sqrt(math.factorial(k) * math.factorial(model.n_bosons - k))
    return (
        1
        / norm
        * (a(model, n, i).dag() + b(model, n, i).dag()) ** k
        * (a(model, n, i).dag() - b(model, n, i).dag()) ** (model.n_bosons - k)
    )


def state_under_h_zz_teor(model, t):
    """
    Return theoretical evolution state of two BEC qubits under h_zz with superposition initial state.

    See [1].
    """

    def alpha(k, t):
        return np.exp(1j * (model.n_bosons - 2 * k) * model.Omega * t) / math.sqrt(2)

    def beta(k, t):
        return alpha(k, t).conjugate()

    return sum(
        (
            math.sqrt(math.comb(model.n_bosons, k))
            * coherent_state_constructor(model, 2, 0, alpha(k, t), beta(k, t))
            * fock_state_constructor(model, 2, 1, k)
            * vacuum_state(model)
        )
        for k in range(model.n_bosons + 1)
    ) / math.sqrt(2**model.n_bosons)


def coherent_state_focked(model, alpha=None, beta=None):
    alpha = 1 / math.sqrt(2) if alpha is None else alpha
    beta = 1 / math.sqrt(2) if beta is None else beta
    return sum(
        math.sqrt(math.comb(model.n_bosons, k))
        * alpha**k
        * beta ** (model.n_bosons - k)
        * qutip.fock(model.n_bosons + 1, k)
        for k in range(model.n_bosons + 1)
    )


def fock_state_focked(model, n, i, k=0):
    return qutip.tensor(
        *[
            [qutip.identity(model.n_bosons + 1)] * i
            + [qutip.fock(model.n_bosons + 1, k)]
            + [qutip.identity(model.n_bosons + 1)] * (n - i - 1)
        ]
    )


def state_under_h_zz_teor_focked(model, t):
    def alpha(k, t):
        return np.exp(1j * (model.n_bosons - 2 * k) * model.Omega * t) / math.sqrt(2)

    def beta(k, t):
        return alpha(k, t).conjugate()

    return sum(
        (
            math.sqrt(math.comb(model.n_bosons, k))
            * qutip.tensor(
                coherent_state_focked(model, alpha(k, t), beta(k, t)),
                qutip.fock(model.n_bosons + 1, k),
            )
        )
        for k in range(model.n_bosons + 1)
    ) / math.sqrt(2**model.n_bosons)


def state_under_h_zz_reduced_teor_focked(model, t):
    def alpha(k, t):
        return np.exp(1j * (model.n_bosons - 2 * k) * model.Omega * t) / math.sqrt(2)

    def beta(k, t):
        return alpha(k, t).conjugate()

    return (
        sum(
            (
                math.comb(model.n_bosons, k)
                * (v := coherent_state_focked(model, alpha(k, t), beta(k, t)))
                * v.dag()
            )
            for k in range(model.n_bosons + 1)
        )
        / 2**model.n_bosons
    )


def state_under_h_na_exact_focked(model, t):
    return sum(
        (
            math.sqrt(math.comb(model.n_bosons, k))
            * qutip.tensor(
                qutip.fock(model.n_bosons + 1, k),
                coherent_state_focked(
                    model, np.exp(1j * k * t) / math.sqrt(2), 1 / math.sqrt(2)
                ),
            )
        )
        for k in range(model.n_bosons + 1)
    ) / math.sqrt(2**model.n_bosons)


def state_under_h_nn_exact(model, t):
    return sum(
        (
            math.sqrt(math.comb(model.n_bosons, k))
            * fock_state_constructor(model, 2, 0, k)
            * coherent_state_constructor(
                model, 2, 1, np.exp(1j * k * t) / math.sqrt(2), 1 / math.sqrt(2)
            )
            * vacuum_state(model)
        )
        for k in range(model.n_bosons + 1)
    ) / math.sqrt(2**model.n_bosons)
