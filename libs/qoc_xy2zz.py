"""
Пытаемся вращениями получить запутанное состояние ZZ гамильтониана,
вращением XY гамильтониана.
"""

import itertools
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import qutip
from joblib import Parallel, delayed
from qutip.control.pulseoptim import optimize_pulse_unitary
from tqdm import tqdm

import bec


@dataclass(frozen=True)
class OptimizeTask:
    n_bosons: int = 3
    phase: float = 0  # np.pi / 4
    t_target: float = 0.4
    n_ts: int = 100
    init_pulse_type: str = "RND"  # RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
    fid_err_targ: float = 1e-2
    max_iter: int = 500
    max_wall_time: timedelta = timedelta(minutes=5).seconds
    min_grad: float = 1e-5
    method: str = "GRAPE"
    controls: tuple[tuple[str]] = (("x", 0), ("y", 1))

    def H_c_constructor(self):
        i_ = 0
        H_c = []
        olocal = {
            "x": bec.sx,
            "y": bec.sy,
            "z": bec.sz,
            "z^2": lambda model, n, k: bec.sz(self.model, n=n, k=k)
            * bec.sz(self.model, n=n, k=k),
        }
        oglobal = {
            "xx": lambda model, n: bec.sx(self.model, n=n, k=0)
            * bec.sx(self.model, n=n, k=1),
            "yy": lambda model, n: bec.sy(self.model, n=n, k=0)
            * bec.sy(self.model, n=n, k=1),
            "xxyy": lambda model, n: bec.sx(self.model, n=n, k=0)
            * bec.sx(self.model, n=n, k=1)
            + bec.sy(self.model, n=n, k=0) * bec.sy(self.model, n=n, k=1),
        }
        for control in self.controls:
            if isinstance(control, (tuple, list)):
                if len(control) == 1:
                    (o,) = control
                    i = i_
                    i_ += 1
                else:
                    o, i = control
                H_c.append(
                    olocal[o](self.model, n=2, k=i),
                )
            else:
                H_c.append(oglobal[control](self.model, n=2))
        return H_c

    def H_d_constructor(self):
        return bec.sx(self.model, n=2, k=0) * bec.sx(self.model, n=2, k=1) - bec.sy(
            self.model, n=2, k=0
        ) * bec.sy(self.model, n=2, k=1)

    @property
    def model(self):
        return bec.BEC_Qubits.init_default(n_bosons=self.n_bosons, phase=self.phase)

    def psi_initial_constructor(self):
        return (
            bec.coherent_state_constructor(self.model, 2, 0)
            * bec.coherent_state_constructor(self.model, 2, 1)
            * bec.vacuum_state(self.model, n=2)
        )

    def psi_target_constructor(self, t=0.4):
        return bec.state_under_h_zz_teor(self.model, t / self.model.Omega)

    def run(self):
        psi_initial = self.psi_initial_constructor()
        psi_target = self.psi_target_constructor()
        tspan = np.linspace(0, self.t_target, self.n_ts)
        H_d = self.H_d_constructor()
        H_c = self.H_c_constructor()
        if self.method == "GRAPE":
            result = optimize_pulse_unitary(
                H_d,
                H_c,
                psi_initial,
                psi_target,
                self.n_ts,
                self.t_target,
                fid_err_targ=self.fid_err_targ,
                max_iter=self.max_iter,
                max_wall_time=self.max_wall_time,
                min_grad=self.min_grad,
                init_pulse_type=self.init_pulse_type,
                gen_stats=True,
            )
        else:
            raise ValueError(f"not supported method: {method}")

        print(self, result.stats.report(), sep="\n")
        return OptimizeTaskResult(
            task=self,
            fid_err=result.fid_err,
            final_amps=result.final_amps,
        )


@dataclass(frozen=True)
class OptimizeTaskResult:
    task: OptimizeTask
    fid_err: float
    final_amps: list


def optimize(*args, **kwargs):
    task = OptimizeTask(*args, **kwargs)
    result = None

    try:
        result = task.run()
    except Exception as e:
        print("Fail:", e, task)
        return

    script_path = pathlib.Path(__file__)
    script_filename = script_path.stem
    result_filename = str(datetime.now().timestamp()).replace(".", "")
    assets_path = script_path.parent.parent / "assets"
    path = assets_path / script_filename / "results" / f"{result_filename}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "task": result.task.__dict__,  # asdict return empty dict
                "fid_err": result.fid_err,
                "final_amps": result.final_amps.tolist(),
            },
            f,
            indent=2,
        )
    print("Done:", path, task)
    return result


if __name__ == "__main__":
    # p_types = "RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE".split("|")
    p_types = "RND|SINE".split("|")
    # controls_set = [(("y", 0), ("y", 1)), (("x", 0), ("x", 1)), ("xx", "yy")]
    # controls_set = [("xxyy", ("x", 0), ("x", 1)), ("xxyy",)]
    controls_set = [
        (("z^2", 0), ("z^2", 1)),
        (("z^2", 0), ("z^2", 1), ("x", 0), ("x", 1)),
    ]
    t_targets = [0.2, 0.3, 0.4]
    max_iter: int = 1000
    max_wall_time = timedelta(hours=2).seconds
    min_grad: float = 1e-8
    n_ts = 400
    # max_wall_time = 1
    cases = list(itertools.product(p_types, controls_set, t_targets))
    Parallel(n_jobs=-2)(
        delayed(optimize)(
            t_target=t_target,
            controls=controls,
            init_pulse_type=p_type,
            max_wall_time=max_wall_time,
            max_iter=max_iter,
            n_ts=n_ts,
        )
        for p_type, controls, t_target in tqdm(cases)
    )
