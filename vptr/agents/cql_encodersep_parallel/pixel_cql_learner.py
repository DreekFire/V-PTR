"""Implementations of algorithms for continuous control."""
from audioop import cross
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flax.training import checkpoints
import pathlib

import copy
import functools
from typing import Dict, Optional, Sequence, Any

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze, unfreeze
from flax.training import train_state

from vptr.agents.agent import Agent

from vptr.agents.cql_encodersep_parallel.actor_updater import update_actor
from vptr.agents.cql_encodersep_parallel.critic_updater import update_critic
from vptr.agents.cql_encodersep_parallel.temperature_updater import update_temperature
from vptr.agents.cql_encodersep_parallel.temperature import Temperature

from vptr.utils.target_update import soft_target_update
from vptr.utils.types import Params, PRNGKey

from vptr.agents.agent import Agent
from vptr.networks.policy import LearnedStdTanhNormalPolicy
from vptr.networks.state_action_ensemble import StateActionEnsemble
from vptr.utils.augmentations import batched_random_crop, color_transform
from vptr.networks.multiplexer import *
from vptr.networks.encoders.resnet_encoderv1 import ResNet34
from jaxrl_m.vision import encoders as encoders
from jaxrl_m.vision.pretrained_encoder import ResizingEncoder

def load_encoder_params(pretrained_file, extra_variables, params, encoder_key):
    extra_variables, params = unfreeze(extra_variables), unfreeze(params)
    
    pretrained_dict = checkpoints.restore_checkpoint(pretrained_file, None)
    pretrained_extra_variables, pretrained_params = pretrained_dict.get('extra_variables', {}), pretrained_dict['params']
    encoder_params = params
    while 'encoder' in encoder_params and 'encoder' in encoder_params['encoder']:
        encoder_params = encoder_params['encoder']
    encoder_params['encoder'] = pretrained_params
    extra_variables.update(pretrained_extra_variables)

    new_encoder_params = params
    while 'encoder' in new_encoder_params:
        new_encoder_params = new_encoder_params['encoder']
    assert(all(jax.tree_leaves(jax.tree_map(lambda x, y: (x == y).all(), new_encoder_params, pretrained_params))))
    return freeze(extra_variables), freeze(params)

class TrainState(train_state.TrainState):
    batch_stats: Any = None


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=jax.lax.pmean(state.batch_stats, axis_name='pmap'))

@functools.partial(jax.pmap, static_broadcasted_argnums=list(range(8,29)), axis_name='pmap')
def _update_jit(
    rng: PRNGKey, actor: TrainState, critic_encoder: TrainState,
    critic_decoder: TrainState, target_critic_encoder_params: Params, 
    target_critic_decoder_params: Params, temp: TrainState, batch: TrainState,
    discount: float, tau: float, target_entropy: float, backup_entropy: bool,
    critic_reduction: str, cql_alpha: float, max_q_backup: bool,
    dr3_coefficient: float, random_crop: bool, color_jitter: bool, cross_norm:bool, aug_next:bool,
    basis_projection_coefficient: float, use_basis_projection: bool, use_gaussian_policy: bool,
    min_q_version: int, bound_q_with_mc:bool = False, use_bc_policy: bool = False,
    reward_regression: bool = False, mc_regression: bool = False, action_jitter_scale:float = 0,):

    # Comment out when using the naive replay buffer
    # batch = _unpack(batch)
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']

    if batch['observations']['pixels'].squeeze().ndim != 2:
        print(f"Data Augmentation, crop: {random_crop}, color: {color_jitter}, next: {aug_next}, action: {action_jitter_scale > 0}")
        if random_crop:
            rng, key = jax.random.split(rng)
            aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
        batch = batch.copy(add_or_replace={'observations': observations})

        if aug_next:
            if random_crop:
                rng, key = jax.random.split(rng)
                aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
            if color_jitter:
                rng, key = jax.random.split(rng)
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32) / 255.) * 255).astype(jnp.uint8)
            next_observations = batch['next_observations'].copy(
                add_or_replace={'pixels': aug_next_pixels})
            batch = batch.copy(add_or_replace={'next_observations': next_observations})

    key, rng = jax.random.split(rng)
    
    target_critic_encoder = critic_encoder.replace(params=target_critic_encoder_params)
    target_critic_decoder = critic_decoder.replace(params=target_critic_decoder_params)
    
    (new_critic_encoder, new_critic_decoder), critic_info = update_critic(
        key,
        actor,
        critic_encoder,
        critic_decoder,
        target_critic_encoder,
        target_critic_decoder,
        temp,
        batch,
        discount,
        backup_entropy=backup_entropy,
        critic_reduction=critic_reduction,
        cql_alpha=cql_alpha,
        max_q_backup=max_q_backup,
        dr3_coefficient=dr3_coefficient,
        cross_norm=cross_norm,
        use_basis_projection=use_basis_projection,
        basis_projection_coefficient=basis_projection_coefficient,
        use_gaussian_policy=use_gaussian_policy,
        min_q_version=min_q_version,
        bound_q_with_mc=bound_q_with_mc,
        reward_regression=reward_regression,
        mc_regression=mc_regression,
        action_jitter_scale=action_jitter_scale,
    )
    if hasattr(new_critic_encoder, 'batch_stats') and new_critic_encoder.batch_stats is not None:
        print ('Syncing batch stats for critic encoder')
        new_critic_encoder = sync_batch_stats(new_critic_encoder)
    if hasattr(new_critic_decoder, 'batch_stats') and new_critic_decoder.batch_stats is not None:
        print ('Syncing batch stats for critic decoder')
        new_critic_decoder = sync_batch_stats(new_critic_decoder)
    
    new_target_critic_encoder_params = soft_target_update(new_critic_encoder.params, target_critic_encoder_params, tau)
    new_target_critic_decoder_params = soft_target_update(new_critic_decoder.params, target_critic_decoder_params, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic_encoder, new_critic_decoder, temp, batch, cross_norm=cross_norm,
                                         use_gaussian_policy=use_gaussian_policy, use_bc_policy=use_bc_policy)
    
    if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
        actor = sync_batch_stats(actor)
    
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, (new_critic_encoder, new_critic_decoder), (new_target_critic_encoder_params, new_target_critic_decoder_params), new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class PixelCQLLearnerEncoderSepParallel(Agent):
    
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (1024, 1024, 1024),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 100,
                 discount: float = 0.99,
                 cql_alpha: float = 0.0,
                 tau: float = 0.0,
                 backup_entropy: bool = False,
                 target_entropy: Optional[float] = None,
                 critic_reduction: str = 'min',
                 dropout_rate: Optional[float] = None,
                 init_temperature: float = 1.0,
                 pretrained_encoder: str = None,
                 max_q_backup: bool = False,
                 policy_encoder_type: str = 'resnet_small',
                 encoder_type: str ='resnet_small',
                 encoder_resize_dim: int = 128,
                 encoder_norm: str = 'batch',
                 dr3_coefficient: float = 0.0,
                 method:bool = False,
                 method_const:float = 0.0,
                 method_type:int=0,
                 cross_norm:bool = False,
                 use_spatial_softmax=False,
                 softmax_temperature=-1,
                 use_spatial_learned_embeddings=False,
                 share_encoders=False,
                 freeze_encoders=False,
                 random_crop=True,
                 color_jitter=True,
                 action_jitter_scale=0,
                 use_bottleneck=True,
                 use_language_bottleneck=False,
                 aug_next=True,
                 use_action_sep=False,
                 use_basis_projection=False,
                 basis_projection_coefficient=0.0,
                 use_film_conditioning=False,
                 use_multiplicative_cond=False,
                 policy_use_multiplicative_cond=False,
                 target_entropy_factor=1,
                 use_gaussian_policy=False,
                 use_mixture_policy=False,
                 use_bc_policy=False,
                 reward_regression=False,
                 mc_regression=False,
                 use_language_sep_critic=False,
                 use_pixel_sep_critic=False,
                 include_state_critic=False,
                 include_state_actor=True,
                 min_q_version=3,
                 std_scale_for_gaussian_policy=0.05,
                 q_dropout_rate=0.0,
                debug:int=0,
                bound_q_with_mc=False,
                use_resizing_encoder=True,
                freeze_batch_stats=False,
                **kwargs,
        ):
        print('unknown arguments: ', kwargs)

        self.debug = debug
        self.random_crop=random_crop
        self.color_jitter=color_jitter
        self.action_jitter_scale=action_jitter_scale

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim * target_entropy_factor
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy
        self.critic_reduction = critic_reduction

        self.tau = tau
        self.discount = discount
        self.max_q_backup = max_q_backup
        self.dr3_coefficient = dr3_coefficient

        self.method = method
        self.method_const = method_const
        self.method_type = method_type
        self.cross_norm = cross_norm
        self.is_replicated=True
        self.use_action_sep = use_action_sep
        self.use_basis_projection = use_basis_projection
        self.basis_projection_coefficient = basis_projection_coefficient
        self.use_film_conditioning = use_film_conditioning
        self.use_multiplicative_cond = use_multiplicative_cond
        self.use_spatial_learned_embeddings = use_spatial_learned_embeddings
        self.use_gaussian_policy = use_gaussian_policy
        self.use_mixture_policy = use_mixture_policy
        self.use_bc_policy = use_bc_policy
        self.reward_regression = reward_regression
        self.mc_regression = mc_regression
        self.policy_use_multiplicative_cond = policy_use_multiplicative_cond
        self.use_pixel_sep_critic = use_pixel_sep_critic
        self.include_state_critic = include_state_critic
        self.include_state_actor = include_state_actor
        self.use_language_bottleneck = use_language_bottleneck
        self.min_q_version = min_q_version
        self.q_dropout_rate = q_dropout_rate
        self.bound_q_with_mc = bound_q_with_mc

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        encoder_is_v1 = False
        if encoder_type == 'resnet_34_v1':
            encoder_is_v1 = True
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature,
                                   use_multiplicative_cond=use_multiplicative_cond,
                                   use_spatial_learned_embeddings=use_spatial_learned_embeddings)
        elif encoder_type == 'resnetv2-50-1':
            encoder_def = encoders[encoder_type](use_film_conditioning=use_film_conditioning)
        elif encoder_type == 'pretrained_resnet':
            encoder_def = encoders['resnetv1-50']()
        elif encoder_type == 'pretrained_resnetv2':
            encoder_def = encoders['resnetv2-50-1'](use_film_conditioning=use_film_conditioning, raw_input=False)
        else:
            raise ValueError('encoder type not found!')

        policy_encoder_is_v1 = False
        if policy_encoder_type == 'resnet_34_v1':
            policy_encoder_is_v1 = True
            policy_encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax,
                                          softmax_temperature=softmax_temperature,
                                          use_spatial_learned_embeddings=use_spatial_learned_embeddings,
                                          use_multiplicative_cond=policy_use_multiplicative_cond)
        elif policy_encoder_type == 'resnetv2-50-1':
            policy_encoder_def = encoders[policy_encoder_type](use_film_conditioning=use_film_conditioning)
        elif policy_encoder_type == 'pretrained_resnet':
            policy_encoder_def = encoders['resnetv1-50']()
        elif policy_encoder_type == 'pretrained_resnetv2':
            policy_encoder_def = encoders['resnetv2-50-1'](use_film_conditioning=use_film_conditioning, raw_input=False)
        elif policy_encoder_type == 'same':
            policy_encoder_is_v1 = encoder_is_v1
            policy_encoder_def = encoder_def
            policy_use_multiplicative_cond = use_multiplicative_cond
        else:
            raise ValueError('encoder type not found!')

        if encoder_resize_dim != observations['pixels'].shape[-4] or use_resizing_encoder:
            train = None
            if freeze_encoders or freeze_batch_stats:
                train = False
            print('Using ResizingEncoder with params:')
            print(dict(resize=True,
                final_shape=(encoder_resize_dim, encoder_resize_dim),
                normalize_imagenet=('pretrained_resnet' in encoder_type),
                frame_stack=encoder_is_v1,
                freeze_encoder=freeze_encoders,
                train=train))
            
            encoder_def = ResizingEncoder(
                encoder_def,
                resize=True,
                final_shape=(encoder_resize_dim, encoder_resize_dim),
                normalize_imagenet=('pretrained_resnet' in encoder_type),
                frame_stack=encoder_is_v1,
                freeze_encoder=freeze_encoders,
                train=train
            )
            policy_encoder_def = ResizingEncoder(
                policy_encoder_def,
                resize=True,
                final_shape=(encoder_resize_dim, encoder_resize_dim),
                normalize_imagenet=('pretrained_resnet' in policy_encoder_type),
                frame_stack=policy_encoder_is_v1,
                freeze_encoder=freeze_encoders,
                train=train
            )
        else:
            print('Skipping ResizingEncoder')


        self.std_scale_for_gaussian_policy = std_scale_for_gaussian_policy
        policy_def = LearnedStdTanhNormalPolicy(hidden_dims, action_dim, dropout_rate=dropout_rate)

        actor_def = PixelMultiplexer(encoder=policy_encoder_def,
                                     network=policy_def,
                                     stop_gradient=share_encoders or freeze_encoders,
                                     bottleneck_dim=latent_dim if use_bottleneck else None,
                                     use_film_conditioning=use_film_conditioning,
                                     use_multiplicative_cond=policy_use_multiplicative_cond,
                                    )
        
        if use_language_bottleneck:
            policy_encoder_def = BottleneckEncoder(encoder=policy_encoder_def, bottleneck_sizes={'language': (latent_dim, latent_dim // 2)})

        if not self.include_state_actor:
            policy_encoder_def = FilterEncoder(encoder=policy_encoder_def, exclude_keys=['state'])

        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init.pop('params')
        if 'extra_variables' in actor_def_init:
            actor_extra_variables = actor_def_init.pop('extra_variables')
        else:
            actor_extra_variables = {}
        if pretrained_encoder:
            print(f'Loading actor encoder from {pretrained_encoder}')
            actor_extra_variables, actor_params = load_encoder_params(
                pretrained_encoder,
                actor_extra_variables,
                actor_params,
                'encoder/encoder'
            )
            if 'batch_stats' in actor_extra_variables:
                print('Loading actor batch stats')
                actor_extra_variables = unfreeze(actor_extra_variables)
                actor_extra_variables['batch_stats'] = {'encoder':  actor_extra_variables['batch_stats']}
                if use_resizing_encoder:
                    actor_extra_variables['batch_stats'] = {'encoder':  actor_extra_variables['batch_stats']}
        actor_batch_stats = actor_extra_variables['batch_stats'] if 'batch_stats' in actor_extra_variables else None
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  batch_stats=actor_batch_stats,
                                  tx=optax.adam(learning_rate=actor_lr))
        
        # print(jax.tree_map(lambda x: x.shape, actor.params))
        actor = flax.jax_utils.replicate(actor)
        # print(jax.tree_map(lambda x: x.shape, actor.params))

        network_def = StateActionEnsemble(hidden_dims, num_qs=2, use_action_sep=self.use_action_sep,
                                          use_language_sep=use_language_sep_critic,
                                          use_pixel_sep=self.use_pixel_sep_critic,
        )

        critic_def_encoder = PixelMultiplexerEncoder(
            encoder=encoder_def,
            bottleneck_dim=latent_dim if use_bottleneck else None,
            use_film_conditioning=use_film_conditioning,
            use_multiplicative_cond=use_multiplicative_cond,
            stop_gradient=freeze_encoders
        )
        
        if use_language_bottleneck:
            critic_def_encoder = BottleneckEncoder(encoder=critic_def_encoder, bottleneck_sizes={'language': (latent_dim, latent_dim // 2)})

        if not self.include_state_critic:
            critic_def_encoder = FilterEncoder(encoder=critic_def_encoder, exclude_keys=['state'])
        
        critic_def_decoder = PixelMultiplexerDecoder(network=network_def)
        
        critic_key_encoder, critic_key_decoder = jax.random.split(critic_key, 2)
        critic_def_encoder_init = critic_def_encoder.init(critic_key_encoder, observations)
        critic_encoder_params = critic_def_encoder_init.pop('params')
        if 'extra_variables' in critic_def_encoder_init:
            critic_encoder_extra_variables = critic_def_encoder_init.pop('extra_variables')
        else:
            critic_encoder_extra_variables = {}
        if pretrained_encoder:
            print(f'Loading critic encoder from {pretrained_encoder}')
            critic_encoder_extra_variables, critic_encoder_params = load_encoder_params(
                pretrained_encoder,
                critic_encoder_extra_variables,
                critic_encoder_params,
                'encoder/encoder'
            )
            if 'batch_stats' in critic_encoder_extra_variables:
                print('Loading critic batch_stats')
                critic_encoder_extra_variables = unfreeze(critic_encoder_extra_variables)
                critic_encoder_extra_variables['batch_stats'] = {'encoder': critic_encoder_extra_variables['batch_stats']}
                if use_resizing_encoder:
                    critic_encoder_extra_variables['batch_stats'] = {'encoder': critic_encoder_extra_variables['batch_stats']}
        critic_encoder_batch_stats = critic_encoder_extra_variables['batch_stats'] if 'batch_stats' in critic_encoder_extra_variables else None
        
        if 'batch_stats' in critic_def_encoder_init:
            embed_obs, _ = critic_def_encoder.apply(
                {'params': critic_encoder_params,
                'batch_stats': critic_def_encoder_init['batch_stats']}, observations, mutable=['batch_stats'])
        else:
            embed_obs = critic_def_encoder.apply({'params': critic_encoder_params}, observations)
        
        critic_def_decoder_init = critic_def_decoder.init(critic_key_decoder, embed_obs, actions)
        critic_decoder_params = critic_def_decoder_init['params']
        critic_decoder_batch_stats = critic_def_decoder_init['batch_stats'] if 'batch_stats' in critic_def_decoder_init else None

        critic_encoder = TrainState.create(apply_fn=critic_def_encoder.apply,
                                   params=critic_encoder_params,
                                   batch_stats=critic_encoder_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))
        
        critic_decoder = TrainState.create(apply_fn=critic_def_decoder.apply,
                                   params=critic_decoder_params,
                                   batch_stats=critic_decoder_batch_stats,
                                   tx=optax.adam(learning_rate=critic_lr))
        
        critic_encoder = flax.jax_utils.replicate(critic_encoder)
        critic_decoder = flax.jax_utils.replicate(critic_decoder)

        target_critic_encoder_params = copy.deepcopy(critic_encoder.params)
        target_critic_decoder_params = copy.deepcopy(critic_decoder.params)

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr))
        temp = flax.jax_utils.replicate(temp)

        self._rng = rng
        self._actor = actor
        self._critic_encoder = critic_encoder
        self._critic_decoder = critic_decoder
        self._critic = (critic_encoder, critic_decoder)
        self.is_replicated = True

        self.aug_next = aug_next
        
        self._temp = temp
        self._target_critic_encoder_params = target_critic_encoder_params
        self._target_critic_decoder_params = target_critic_decoder_params
        self._target_critic_params = (target_critic_encoder_params, target_critic_decoder_params)
        
        self._cql_alpha = cql_alpha
        print ('Discount: ', self.discount)
        print ('CQL Alpha: ', self._cql_alpha)
        print('Method: ', self.method, 'Const: ', self.method_const)

    def encoder_apply(self, obs, params=None, batch_stats=None, training=True):
        if params is None:
            params = self._critic_encoder.params
        if batch_stats is None and hasattr(self._critic_encoder, 'batch_stats'):
            batch_stats = self._critic_encoder.batch_stats
        if batch_stats is not None:
            embed_obs, new_model_state_encoder = self._critic_encoder.apply_fn(
                {'params': params, 'batch_stats': batch_stats},
                obs, mutable=['batch_stats'], training=training)
        else:
            embed_obs = self._critic_encoder.apply_fn({'params': params}, obs)
            new_model_state_encoder = None
        return embed_obs, new_model_state_encoder

    def decoder_apply(self, embed_obs, actions, dropout_key, params=None, batch_stats=None, training=True):
        if params is None:
            params = self._critic_decoder.params
        if batch_stats is None and hasattr(self._critic_decoder, 'batch_stats'):
            batch_stats = self._critic_decoder.batch_stats
            
        if batch_stats is not None:
            qs, new_model_state_decoder = self._critic_decoder.apply_fn(
                {'params': params, 'batch_stats': batch_stats},
                embed_obs, actions, mutable=['batch_stats'],
                rngs={'dropout': dropout_key}, training=training)
        else:
            qs = self._critic_decoder.apply_fn(
                {'params': params}, embed_obs, actions,
                rngs={'dropout': dropout_key})
            new_model_state_decoder = None
        return qs, new_model_state_decoder

    def critic_apply(self, obs, actions, dropout_key, params=None, batch_stats=None, training=True):
        if params is not None:
            encoder_params, decoder_params = params
        else:
            encoder_params, decoder_params = None, None
        if batch_stats is not None:
            encoder_batch_stats, decoder_batch_stats = batch_stats
        else:
            encoder_batch_stats, decoder_batch_stats = None, None
        embed_obs, new_encoder_state = self.encoder_apply(obs, encoder_params, encoder_batch_stats, training)
        qs, new_decoder_state = self.decoder_apply(embed_obs, actions, dropout_key, decoder_params, decoder_batch_stats, training)
        return qs, (new_encoder_state, new_decoder_state)

    def actor_apply(self, obs, dropout_key, params=None, batch_stats=None, training=True):
        actor = self._actor

        if params is None:
            params = actor.params
        if batch_stats is None and hasattr(actor, 'batch_stats'):
            batch_stats = actor.batch_stats

        new_actor_state = None
        if batch_stats is not None:
            dist, new_actor_state = actor.apply_fn({'params': params, 'batch_stats': batch_stats}, 
                                    obs, mutable=['batch_stats'], rngs={'dropout': dropout_key}, training=training)
        else:
            dist = actor.apply_fn({'params': params}, obs, rngs={'dropout': dropout_key})
        return dist, new_actor_state
        
    def unreplicate(self):
        if not self.is_replicated:
            raise RuntimeError('Not Replicated') 
        # else:
        #     print('UNREPLICATING '*5)
        self._actor = flax.jax_utils.unreplicate(self._actor)
        self._critic_encoder = flax.jax_utils.unreplicate(self._critic_encoder)
        self._critic_decoder = flax.jax_utils.unreplicate(self._critic_decoder)
        self._target_critic_encoder_params = flax.jax_utils.unreplicate(self._target_critic_encoder_params)
        self._target_critic_decoder_params = flax.jax_utils.unreplicate(self._target_critic_decoder_params)
        self._critic = (self._critic_encoder, self._critic_decoder)
        self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        self.is_replicated=False
    
    def replicate(self):
        if self.is_replicated:
            raise RuntimeError('Already Replicated') 
        # else:
        #     print('REPLICATING '*5)
        self._actor = flax.jax_utils.replicate(self._actor)
        self._critic_encoder = flax.jax_utils.replicate(self._critic_encoder)
        self._critic_decoder = flax.jax_utils.replicate(self._critic_decoder)
        self._target_critic_encoder_params = flax.jax_utils.replicate(self._target_critic_encoder_params)
        self._target_critic_decoder_params = flax.jax_utils.replicate(self._target_critic_decoder_params)
        self._critic = (self._critic_encoder, self._critic_decoder)
        self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
        self.is_replicated=True

    def use_bc_actor(self, switch=True):
        self.use_bc_policy = switch

    def update(self, batch: FrozenDict, no_update=False):
        if not self.debug:
            num_devices = len(jax.devices())
            copied_keys = jax.random.split(self._rng, num_devices)
        else:
            copied_keys = self._rng
            if self.is_replicated:
                self.unreplicate()
        
        new_rng, new_actor, new_critic, new_target_critic_params, new_temp, info = _update_jit(
            copied_keys, 
            self._actor, 
            self._critic_encoder, 
            self._critic_decoder, 
            self._target_critic_encoder_params, 
            self._target_critic_decoder_params,
            self._temp, 
            batch, 
            self.discount, 
            self.tau, 
            self.target_entropy,
            self.backup_entropy, 
            self.critic_reduction, 
            self._cql_alpha, 
            self.max_q_backup,
            self.dr3_coefficient,
            self.random_crop and not self.debug,  
            self.color_jitter and not self.debug, 
            self.cross_norm, 
            self.aug_next and not self.debug, 
            self.basis_projection_coefficient, 
            self.use_basis_projection, 
            self.use_gaussian_policy, 
            self.min_q_version,
            self.bound_q_with_mc,
            self.use_bc_policy,
            self.reward_regression,
            self.mc_regression,
            self.action_jitter_scale,
        )
        
        new_critic_encoder, new_critic_decoder = new_critic
        new_target_critic_encoder_params, new_target_critic_decoder_params = new_target_critic_params
        
        if not self.debug:
            info = {k:v[0] for k,v in info.items()}
            self._rng = new_rng[0]
        
        if not no_update:
            self._actor = new_actor
            self._critic_encoder = new_critic_encoder
            self._critic_decoder = new_critic_decoder
            self._critic = (new_critic_encoder, new_critic_decoder)
            
            self._target_critic_encoder_params = new_target_critic_encoder_params
            self._target_critic_decoder_params = new_target_critic_decoder_params
            self._target_critic_params = (new_target_critic_encoder_params, new_target_critic_decoder_params)
            
            self._temp = new_temp

        return info

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, prefix=None):
        # try:
        from examples.train_utils import make_multiple_value_reward_visualizations
        make_multiple_value_reward_visualizations(self, variant, i, eval_buffer, wandb_logger, prefix=prefix)
        # except Exception as e:
        #     print(e)
        #     print('Could not visualize')

    def show_value_reward_visualization(self, traj):
        q_vals = []
        for step in traj:
            q_vals.append(get_q_value(step['action'], step['observation'], self._critic_encoder, self._critic_decoder))
        rewards = [step['reward'] for step in traj]
        images = np.stack([step['observation']['pixels'][0] for step in traj], 0)
        make_visual_eval(q_vals, rewards, images, show_window=True)

    def make_value_reward_visualization(self, variant, trajs):
        traj_images = []
        # num_stack = variant.frame_stack

        for traj in trajs:
            observations = traj['observations']
            next_observations = traj['next_observations']
            actions = traj['actions']
            rewards = traj['rewards']
            masks = traj['masks']

            target_critic_encoder = self._critic_encoder.replace(params=self._target_critic_encoder_params)
            target_critic_decoder = self._critic_decoder.replace(params=self._target_critic_decoder_params)

            q_pred = []
            target_q_pred = []
            bellman_loss = []
            task_ids = []
            if variant.lang_embedding is not None:
                task_ids = None
                embedding = traj['observations']['language'][0]
            else:
                embedding= None
            
            # Do the frame stacking thing for observations
            # images = np.lib.stride_tricks.sliding_window_view(observations.pop('pixels'), num_stack + 1, axis=0)

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                    
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]
                
                q_value = get_q_value(action, obs_dict, self._critic_encoder, self._critic_decoder)
                next_action = get_action(next_obs_dict, self._actor)
                target_q_value = get_q_value(next_action, next_obs_dict, target_critic_encoder, target_critic_decoder)
                target_q_value = rewards[t] + target_q_value.min() * self.discount * masks[t]
                q_pred.append(q_value)
                target_q_pred.append(target_q_value.item())
                bellman_loss.append(((q_value-target_q_value)**2).mean().item())
                if variant.lang_embedding is None:
                    task_ids.append(np.argmax(observations['task_id']))
            
            # print ('lengths for verification: ', len(task_ids), len(q_pred), len(masks), len(bellman_loss))

            traj_images.append(make_visual(
                q_pred,
                traj['mc_returns'],
                rewards,
                observations['pixels'],
                masks,
                target_q_pred,
                bellman_loss,
                task_ids,
                embedding,
                traj['label'],
            ))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)
        # except Exception as e:
        #     print(e)
        #     return np.zeros((num_traj, 128, 128, 3))
    
    def make_language_counterfactual_visualization(self, trajs):
        counterfactual_trajs = [*trajs[1:], trajs[0]]
        traj_images = []

        for traj, counterfactual_traj in zip(trajs, counterfactual_trajs):
            observations = traj['observations']
            actions = traj['actions']

            q_pred = []
            counterfactual_q_pred = []

            embedding = traj['observations']['language'][0]
            counterfactual_embedding = counterfactual_traj['observations']['language'][0]
            
            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]

                q_value = get_q_value(action, obs_dict, self._critic_encoder, self._critic_decoder)
                q_pred.append(q_value)

                counterfactual_obs_dict = copy.deepcopy(obs_dict)
                counterfactual_obs_dict['language'] = counterfactual_embedding[None]
                counterfactual_q_value = get_q_value(action, counterfactual_obs_dict, self._critic_encoder, self._critic_decoder)
                counterfactual_q_pred.append(counterfactual_q_value)
            
            # print ('lengths for verification: ', len(task_ids), len(q_pred), len(masks), len(bellman_loss))

            traj_images.append(make_visual_counterfactual(
                q_pred,
                counterfactual_q_pred,
                observations['pixels'],
                embedding,
                traj['label'],
                counterfactual_traj['label'],
            ))
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return jax.device_get(save_dict)

    def restore_checkpoint(self, path):
        # assert pathlib.Path(dir).is_file(), 'path {} not found!'.format(dir)
        was_replicated = False
        if self.is_replicated:
            self.unreplicate()
            was_replicated = True
        try:
            output_dict = checkpoints.restore_checkpoint(path, self._save_dict)
            self._actor = output_dict['actor']
            self._critic_encoder, self._critic_decoder = output_dict['critic']
            self._critic = (self._critic_encoder, self._critic_decoder)
            self._target_critic_encoder_params, self._target_critic_decoder_params = output_dict['target_critic_params']
            self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
            self._temp = output_dict['temp']
            print('successfully restored')
        except Exception as e:
            output_dict = checkpoints.restore_checkpoint(path, None)
            breakpoint()
            # params_list = [self._actor.params,
            #                self._critic_encoder.params,
            #                self._target_critic_encoder_params,
            #                ]
            # ckpt_params_paths = [['actor', 'params'],
            #                     ['actor', 'opt_state'],
            #                     ['critic', '0', 'params'],
            #                     ['critic', '0', 'opt_state'],
            #                     ['target_critic_params', '0'],
            #                     ]

            def dict_replace(nested_dict, target, replacement):
                found = []
                for k, v in nested_dict.items():
                    if target in k:
                        found.append(k)
                    if isinstance(v, dict):
                        dict_replace(v, target, replacement)
                for k in found:
                    nested_dict[k.replace(target, replacement)] = nested_dict.pop(k)

            dict_replace(output_dict, 'MLPActionSep', 'MLP')

            if 'PixelMultiplexerEncoder_0' not in output_dict['actor']['params']:
                output_dict['actor']['params']['PixelMultiplexerEncoder_0'] = {
                    'Dense_0': output_dict['actor']['params'].pop('Dense_0'),
                    'LayerNorm_0': output_dict['actor']['params'].pop('LayerNorm_0'),
                }

            if not self.include_state_critic:
                output_dict['critic']['0']['params'] = {'encoder': output_dict['critic']['0']['params']}
            if not self.include_state_actor:
                output_dict['actor']['params'] = {'encoder': output_dict['actor']['params']}
            if self.use_language_bottleneck:
                output_dict['critic']['0']['params'] = {'encoder': output_dict['critic']['0']['params']}
                output_dict['actor']['params'] = {'encoder': output_dict['actor']['params']}

            # for param, ckpt_param_path in zip(params_list, ckpt_params_paths):
            #     ckpt_param = output_dict
            #     for p in ckpt_param_path:
            #         ckpt_param = ckpt_param[p]
            #     ckpt_enc = ckpt_param
            #     while 'encoder' in ckpt_enc:
            #         ckpt_enc = ckpt_enc['encoder']
            #     enc = param
            #     num_nest = -1
            #     while 'encoder' in enc:
            #         num_nest += 1
            #         enc = enc['encoder']
            #     print(num_nest)
            #     if num_nest == -1:
            #         continue
            #     for i in range(num_nest):
            #         ckpt_enc = {'encoder': ckpt_enc}
            #     ckpt_param['encoder'] = ckpt_enc
            # if 'PixelMultiplexerEncoder_0' not in output_dict['actor']['params']:
            #     output_dict['actor']['params']['PixelMultiplexerEncoder_0'] = {
            #         'Dense_0': output_dict['actor']['params'].pop('Dense_0'),
            #         'LayerNorm_0': output_dict['actor']['params'].pop('LayerNorm_0'),
            #     }
            self._actor = self._actor.replace(**output_dict['actor'])
            self._critic_encoder = self._critic_encoder.replace(**output_dict['critic']['0'])
            self._critic_decoder = self._critic_decoder.replace(**output_dict['critic']['1'])
            self._critic = (self._critic_encoder, self._critic_decoder)
            self._target_critic_encoder_params = output_dict['target_critic_params']['0']
            self._target_critic_decoder_params = output_dict['target_critic_params']['1']
            self._target_critic_params = (self._target_critic_encoder_params, self._target_critic_decoder_params)
            self._temp = self._temp.replace(**output_dict['temp'])
        
        self.is_replicated = False # stored in the checkpoint, so we need to reset this
        
        # stored as 8xtemp_size so unreplicate and then replicate
        if self._temp.params['log_temp'].size > 1:
            self._temp = flax.jax_utils.unreplicate(self._temp)

        if was_replicated:
            self.replicate()

        print('restored from ', path)
        

@functools.partial(jax.jit)
def get_action(obs_dict, actor):
    # print(f'{images.shape=}')
    # print(f'{images[..., None]=}')
    key_dropout, key_pi = jax.random.split(jax.random.PRNGKey(0))
    
    if actor.batch_stats is not None:
        dist = actor.apply_fn({'params': actor.params, 'batch_stats': actor.batch_stats}, obs_dict, rngs={'dropout': key_dropout})
    else:
        dist = actor.apply_fn({'params': actor.params}, obs_dict, rngs={'dropout': key_dropout})
        
    actions, policy_log_probs = dist.sample_and_log_prob(seed=key_pi)
    return actions

@jax.jit
def get_q_value(actions, obs_dict, critic_encoder, critic_decoder):
    print(obs_dict['pixels'].shape)
    if critic_encoder.batch_stats is not None:
        embed_obs, _ = critic_encoder.apply_fn({'params': critic_encoder.params, 'batch_stats': critic_encoder.batch_stats}, obs_dict, mutable=['batch_stats'])
    else:    
        embed_obs = critic_encoder.apply_fn({'params': critic_encoder.params}, obs_dict)
        
    if critic_decoder.batch_stats is not None:
        q_pred, _ = critic_decoder.apply_fn({'params': critic_decoder.params, 'batch_stats': critic_decoder.batch_stats}, embed_obs, actions, mutable=['batch_stats'])
    else:    
        q_pred = critic_decoder.apply_fn({'params': critic_decoder.params}, embed_obs, actions)
        
    return q_pred

def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, mc_returns, rewards, images, masks, target_q_pred, bellman_loss, task_ids, embedding=None, label=''):
    q_estimates_np = np.stack(q_estimates, 0)

    fig, axs = plt.subplots(8, 1, figsize=(8, 20))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, C, H, W shape
    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)

    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    
    axs[1].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('q values')

    axs[2].plot(mc_returns, linestyle='--', marker='o')
    axs[2].set_ylabel('mc returns')
    axs[2].set_xlim([0, len(mc_returns)])
    
    axs[3].plot(target_q_pred, linestyle='--', marker='o')
    axs[3].set_ylabel('target_q_pred')
    axs[3].set_xlim([0, len(target_q_pred)])
    
    axs[4].plot(bellman_loss, linestyle='--', marker='o')
    axs[4].set_ylabel('bellman_loss')
    axs[4].set_xlim([0, len(bellman_loss)])
    
    axs[5].plot(rewards, linestyle='--', marker='o')
    axs[5].set_ylabel('rewards')
    axs[5].set_xlim([0, len(rewards)])
    
    axs[6].plot(masks, linestyle='--', marker='o')
    axs[6].set_ylabel('masks')
    axs[6].set_xlim([0, len(masks)])
    
    if task_ids is not None:
        axs[7].plot(task_ids, linestyle='--', marker='o')
        axs[7].set_ylabel('task_ids')
        axs[7].set_xlim([0, len(masks)])
    elif embedding is not None:
        plt.xlim([0, 32])
        grid = np.pad(embedding.flatten(), (0, -embedding.size % 32)).reshape(-1, 32)
        axs[7].imshow(grid)
        axs[7].set_title('embedding:\n' + label)

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()
    return out_image

def make_visual_counterfactual(q_estimates, counterfactual_q_estimates, images, embedding, label='', counter_label=''):
    q_estimates_np = np.stack(q_estimates, 0)
    counterfactual_q_estimates_np = np.stack(counterfactual_q_estimates, 0)

    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, C, H, W shape
    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)

    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)

    axs[1].plot(counterfactual_q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(counterfactual_q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('counterfactual q values')
    
    axs[2].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[2].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[2].set_ylabel('q values')

    plt.xlim([0, 32])
    grid = np.pad(embedding.flatten(), (0, -embedding.size % 32)).reshape(-1, 32)
    axs[3].imshow(grid)
    axs[3].set_title('embedding:\n' + label + '\ncounterfactual:\n' + counter_label)

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return out_image

def make_visual_eval(q_estimates, rewards, images, masks=None, show_window=False, save_values=True):
    if show_window:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    import numpy as np

    q_estimates_np = np.stack(q_estimates, 0)

    fig, axs = plt.subplots(4, 1)
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates)])

    # assume image in T, H, W, C, 1 shape
    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = images.shape[0] // 4
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    axs[1].plot(q_estimates_np[:, 0], linestyle='--', marker='o')
    axs[1].plot(q_estimates_np[:, 1], linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    if masks is not None:
        axs[3].plot(masks, linestyle='--', marker='o')
        axs[3].set_ylabel('masks')
        axs[3].set_xlim([0, len(masks)])
    else:
        masks = 1-np.array(rewards)
        masks = masks.tolist()
        axs[3].plot(masks, linestyle='--', marker='o')
        axs[3].set_ylabel('masks')
        axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if show_window:
        plt.imshow(out_image)
        plt.show()

    if save_values:
        f_path='out.npz'
        np.savez(f_path, q_vals=q_estimates, all_images=images, sel_images=sel_images, rewards=rewards, masks=masks)

    plt.close(fig)
    return out_image
