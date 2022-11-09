import random
from skopt import Optimizer
from skopt.space import Categorical

def random_hp_generator(n_iter, **kwargs):
    hp = {}
    for _ in range(n_iter):
        for key, val in kwargs.items():
            if isinstance(val[0], int):
                hp[key] = random.randint(val[0], val[-1])
            else:
                hp[key] = random.uniform(val[0], val[-1])
        yield hp


def rec_grid(hp, **kwargs):
    if len(kwargs) == 0:
        yield {k: v for k, v in hp.items()}
    else:
        name = list(kwargs.keys())[0]
        p = kwargs.pop(name)
        for i in p:
            hp[name] = i
            yield from rec_grid(hp, **kwargs)


def grid_hp_generator(**kwargs):
    if "n_iter" in kwargs:
        kwargs.pop("n_iter")
    yield from rec_grid({}, **kwargs)


class Bayesian_hp_generator:
    def __init__(self):
        self.iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter >= self.n_iter:
            self.opt = Optimizer(self.hp_ranges)
            self.iter = 0
            raise StopIteration
        self.iter += 1
        self.last_point = self.opt.ask()
        return {i[0]: p for i, p in zip(self.range_list, self.last_point)}

    def __call__(self, *args, **kwargs):
        self.n_iter = kwargs.pop("n_iter")
        if "sub_reservoirs" in kwargs:
            input_scaling = kwargs.pop("input_scaling")
            bias_scaling = kwargs.pop("bias_scaling")
            if "units_sub" in kwargs:
                units_sub = kwargs.pop("units_sub")
            for i in range(kwargs["sub_reservoirs"][0]):
                kwargs[f"input_scaling{i+1}"] = input_scaling
                kwargs[f"bias_scaling{i+1}"] = bias_scaling
                if "units_sub" in locals():
                    kwargs[f"units_sub{i + 1}"] = units_sub
        self.range_list = list(kwargs.items())
        self.hp_ranges = [Categorical(v, name=k) for k, v in self.range_list]
        self.opt = Optimizer(self.hp_ranges)
        return self

    def tell(self, val):
        self.opt.tell(self.last_point, val)
        return