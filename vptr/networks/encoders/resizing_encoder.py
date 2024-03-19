from flax import linen as nn
from jax import numpy as jnp
import jax

def preprocess_observations(obs,
                            normalize_imagenet=False,
                            resize=False,
                            final_shape=(224, 224),
                            center_crop=False,
                            pre_crop_shape=(256, 256)):
    if resize:
        if obs.shape[-3] != final_shape[0] or obs.shape[-2] != final_shape[1]: # Already resized
            resize_shape = pre_crop_shape if center_crop else final_shape
            if obs.shape[-3] != resize_shape[0] or obs.shape[-2] != resize_shape[1]:
                print('Resizing to %s' % str(resize_shape))
                obs = jax.image.resize(obs, (*obs.shape[:-3], *resize_shape, 3), method='bilinear')

            if center_crop:
                start_y, start_x = (pre_crop_shape[0] - final_shape[0]) // 2, (pre_crop_shape[1] - final_shape[1]) // 2
                obs = obs[..., start_y:start_y + final_shape[0], start_x:start_x + final_shape[1], :]
                print('Cropping to %s' % str(obs.shape))

    if normalize_imagenet:
        obs = obs / 255.0
        obs = obs - jnp.array([0.485, 0.456, 0.406])
        obs = obs / jnp.array([0.229, 0.224, 0.225])

    return obs

class ResizingEncoder(nn.Module):
    encoder: nn.Module
    normalize_imagenet: bool = False

    resize: bool = True
    final_shape: tuple = (224, 224)
    center_crop: bool = False
    pre_crop_shape: tuple = (256, 256)

    freeze_encoder: bool = False
    frame_stack: bool = False
    default_kwargs: dict = None
    train: bool = None

    @nn.compact
    def __call__(self, observations, train:bool = True, **kwargs):
        observations = observations.squeeze(-1)
        no_batch_dim = len(observations.shape) == 3
        if no_batch_dim:
            print('Adding batch dimension')
            observations = jnp.expand_dims(observations, 0)
        observations = preprocess_observations(observations, self.normalize_imagenet, self.resize, self.final_shape, self.center_crop, self.pre_crop_shape)
        if self.default_kwargs is not None:
            kwargs = {**self.default_kwargs, **kwargs}
        if self.frame_stack:
            observations = observations[..., None]
        if self.train is not None:
            train = self.train
            print('Overriding train to %s' % str(train))
        if self.freeze_encoder:
            train = False
        output = self.encoder(observations, train=train, **kwargs)
        if self.freeze_encoder:
            output = jax.lax.stop_gradient(output)        

        return output