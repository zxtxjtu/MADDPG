from common.arguments import get_args
from common.utils import make_env, make_charge_env
from runner import Runner

# import numpy as np
# import random
# import torch


if __name__ == '__main__':
    # get the params
    # args = get_args()
    # env, args = make_env(args)

    my_args = get_args()
    my_env, my_args = make_charge_env(my_args)
    runner = Runner(my_args, my_env)
    if my_args.evaluate:
        returns = runner.charge_evaluate()
        print('Average returns is', returns)
    else:
        runner.charge_run()
