import random

import numpy as np
import torch


def save_random_states():
    rng_states = {
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
        'random': random.getstate(),
        'numpy': np.random.get_state(),
    }
    return rng_states


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_random_states(rng_states):
    if rng_states is None:
        set_global_seed(42)
    else:
        torch.set_rng_state(rng_states['torch_cpu'])
        torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
        random.setstate(rng_states['random'])
        np.random.set_state(rng_states['numpy'])
