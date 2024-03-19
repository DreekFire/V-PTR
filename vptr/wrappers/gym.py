import gym

from vptr.wrappers.single_precision import SinglePrecision
from vptr.wrappers.universal_seed import UniversalSeed


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    return env