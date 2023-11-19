import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(sys.path[0]).parent / "libs"))


from qoc_xy2zz import OptimizeTask

task = OptimizeTask(
    max_iter=5,
    max_wall_time=1000,
)

t0 = time.time()
result = task.run()
print("Eval time:", time.time() - t0, "seconds")
print(result)
