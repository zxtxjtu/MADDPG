import random

import numpy as np
import torch

from common.arguments import get_args
from common.utils import make_charge_env
from runner import Runner

if __name__ == '__main__':
    # get the params
    # args = get_args()
    # env, args = make_env(args)
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(1000)

    my_args = get_args()
    my_env, my_args = make_charge_env(my_args)
    runner = Runner(my_args, my_env)
    if my_args.evaluate:
        returns = runner.charge_evaluate()
        print('Average returns is', returns)
    else:
        print("run")
        runner.charge_run()
