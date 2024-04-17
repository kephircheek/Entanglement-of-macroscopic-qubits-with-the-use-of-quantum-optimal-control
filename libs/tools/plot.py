import math


def set_xlabel_as_pi_fraction(ax, fontsize=None):
    ax.set_xticks(
        *zip(
            *(
                (
                    (math.pi / i, rf"$\frac{{\pi}}{{{i}}}$")
                    if i not in [0, 1]
                    else ((0, "0") if i == 0 else (math.pi, r"$\pi$"))
                )
                if isinstance(i, int)
                else (i[0] * math.pi / i[1], rf"$\frac{{{i[0]}\pi}}{{{i[1]}}}$")
                for i in (
                    0,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    (2, 5),
                    (4, 9),
                    2,
                    (4, 7),
                    (2, 3),
                    (4, 5),
                    1,
                )
            )
        ),
        fontsize=fontsize,
    )
