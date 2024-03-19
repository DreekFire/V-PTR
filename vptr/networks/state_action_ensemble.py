from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

import jax

from vptr.networks.mlp import MLP

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

default_kernel_init = nn.initializers.lecun_normal()



class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_action_sep: bool = False
    use_language_sep: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        inputs = {'states': observations, 'actions': actions}
        print ('Use action sep in StateActionvalue: ', self.use_action_sep)
        concat_argnames = []
        if self.use_action_sep:
            concat_argnames.append('actions')
        if self.use_pixel_sep:
            concat_argnames.append('states/pixels')
        if self.use_language_sep:
            concat_argnames.append('states/language')
        critic = MLP(
            (*self.hidden_dims, 1),
            activations=self.activations,
            concat_argnames=concat_argnames)(inputs, training=training)
        return jnp.squeeze(critic, -1)


class StateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    use_action_sep: bool = False
    use_language_sep: bool = False
    use_pixel_sep: bool = False

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        print(jax.tree_util.tree_map(jnp.shape, states), actions.shape, training)

        print ('Use action sep in state action ensemble: ', self.use_action_sep)
        VmapCritic = nn.vmap(StateActionValue,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        use_action_sep=self.use_action_sep,
                        use_language_sep=self.use_language_sep,
                        use_pixel_sep=self.use_pixel_sep,)(
                            states, actions, training)
        return qs
