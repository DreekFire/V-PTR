from typing import Callable, Optional, Sequence, Union, Dict
from flax.core import frozen_dict

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import jax

def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def xavier_init():
    return nn.initializers.xavier_normal()

def kaiming_init():
    return nn.initializers.kaiming_normal()

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    dropout_rate: Optional[float] = None
    concat_argnames: Sequence[str] = ()

    @nn.compact
    def __call__(self, x: Dict[str, jnp.ndarray], training: bool = False):
        z = jnp.concatenate([z_i for z_i in jax.tree_util.tree_leaves(x)], axis=-1)
        z_used = z
        for i, size in enumerate(self.hidden_dims):
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                z = nn.Dense(size, kernel_init=default_init())(z_used)
                z = self.activations(z)
                if self.dropout_rate:
                    z = nn.Dropout(rate=self.dropout_rate)(
                        z, deterministic=not training)
            else:
                z = nn.Dense(size, kernel_init=default_init(1e-2))(z_used)
                    
            z_used = [z]
            for argname in self.concat_argnames:
                val = x
                for name in argname.split('/'):
                    val = val[name]
                z_used.append(val)
            z_used = jnp.concatenate(z_used, axis=-1)
            print(f'MLP state post flatten at layer {i} shape: ', z_used.shape)

            print ('FF layers: ', z.shape, z_used.shape)
        return z