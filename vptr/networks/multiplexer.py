from typing import Dict, Optional, Union, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from vptr.networks.mlp import MLP

class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    stop_gradient: bool = False
    bottleneck_dim: Optional[int] = None
    use_film_conditioning: bool = False
    use_multiplicative_cond: bool = False

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        
        observations = FrozenDict(observations)

        x = PixelMultiplexerEncoder(self.encoder,
                                    self.stop_gradient,
                                    self.bottleneck_dim,
                                    self.use_film_conditioning,
                                    self.use_multiplicative_cond)(observations, training)

        print('fully connected keys', x.keys())
        print('x keys shape: ', jax.tree_map(lambda aa: aa.shape, x))

        return PixelMultiplexerDecoder(self.network)(x, actions, training)

'''
Split into Encoder and Decoder
'''
class PixelMultiplexerEncoder(nn.Module):
    encoder: Union[nn.Module, list]
    stop_gradient: bool = False
    bottleneck_dim: Optional[int] = None
    use_film_conditioning: bool = False
    use_multiplicative_cond: bool = False

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 training: bool = False):
        observations = FrozenDict(observations)

        assert not (self.use_film_conditioning and self.use_multiplicative_cond), "film overrides cond"

        if self.use_film_conditioning:
            x = self.encoder(observations['pixels'], training, film_conditioning=observations['language'])
        elif self.use_multiplicative_cond:
            x = self.encoder(observations['pixels'], training, cond_var=observations['task_id'])
        else:
            x = self.encoder(observations['pixels'], training)

        if self.stop_gradient:
            # We do not update conv layers with policy gradients.
            x = jax.lax.stop_gradient(x)

        if self.bottleneck_dim:
            x = nn.Dense(self.bottleneck_dim, kernel_init=nn.initializers.xavier_normal())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            # x = nn.relu(x)
            
        update_dict = {'pixels': x}

        return observations.copy(add_or_replace=update_dict)

class PixelMultiplexerDecoder(nn.Module):
    network: nn.Module
    
    @nn.compact
    def __call__(self, 
                 embedding: Union[FrozenDict, Dict], 
                 actions: Optional[jnp.ndarray] = None, 
                 training: bool = False):
        if actions is None:
            return self.network(embedding, training=training)
        else:
            return self.network(embedding, actions, training=training)

class FilterEncoder(nn.Module):
    encoder: nn.Module
    exclude_keys: list[str]

    @nn.compact
    def __call__(self, 
                 observations: Union[FrozenDict, Dict],
                 *args,
                 training: bool = False):
        
        observations = {k: v for k, v in observations.items() if k not in self.exclude_keys}
        
        return self.encoder(observations, *args, training=training)
        
class BottleneckEncoder(nn.Module):
    encoder: nn.Module
    bottleneck_sizes: Dict[str, Sequence[int]]

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 *args,
                 training: bool = False):
        
        update_dict = {}
        for k, v in self.bottleneck_sizes.items():
            if k not in observations:
                continue
            update_dict[k] = MLP(v)(observations[k], training=training)
        if isinstance(observations, FrozenDict):
            observations = observations.copy(add_or_replace=update_dict)
        else:
            observations = observations.copy().update(update_dict)
        
        return self.encoder(observations, *args, training=training)