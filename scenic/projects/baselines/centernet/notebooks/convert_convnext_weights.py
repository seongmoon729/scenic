import functools
from typing import Tuple, Callable, Any, Optional, Union, Dict

# from absl import logging
import flax
import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp

# import ml_collections
from jax import random
import numpy as np
from flax.training import checkpoints
import torch
import torch.utils.model_zoo
flax.config.update('flax_use_orbax_checkpointing', False)


# From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
MODEL_URLS = dict(
    convnext_tiny_1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
    convnext_small_1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth',
    convnext_base_1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth',
    convnext_large_1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth',
    convnext_tiny_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth',
    convnext_small_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth',
    convnext_base_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
    convnext_large_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth',
    convnext_xlarge_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth',

    convnext_tiny_384_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth',
    convnext_small_384_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
    convnext_base_384_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
    convnext_large_384_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
    convnext_xlarge_384_in22ft1k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',

    convnext_tiny_in22k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth',
    convnext_small_in22k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth',
    convnext_base_in22k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth',
    convnext_large_in22k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth',
    convnext_xlarge_in22k=
    'https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth',
)

SIZE_MAP = {
    'tiny': 'T',
    'small': 'S',
    'base': 'B',
    'large': 'L',
    'xlarge': 'XL'
}


#@title Jax model implementation

class ConvNeXtBlock(nn.Module):
  """Bottleneck ResNet block.
  """

  dim: int
  droplayer_p: float = 0
  layer_scale_init_value: float = 1e-6
  dtype: jnp.dtype = jnp.float32

  def get_drop_pattern(self,
                       x: jnp.ndarray,
                       deterministic: bool) -> jnp.ndarray:
    """Returns dropout mask for stochastic depth regularisation."""
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype(self.dtype)
    else:
      return 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    residual = x
    x = nn.Conv(
        self.dim, (7, 7),
        1,
        padding=3,
        feature_group_count=self.dim,
        use_bias=True,
        dtype=self.dtype,
        name='dwconv')(
            x)
    x = nn.LayerNorm(epsilon=1e-6, name='norm')(x)
    x = nn.Dense(4 * self.dim, name='pwconv1')(x)  # B x H x W x 4C
    x = nn.gelu(x)
    x = nn.Dense(self.dim, name='pwconv2')(x)  # B x H x W x C
    if self.layer_scale_init_value > 0:
      gamma = self.param(
          'gamma',
          initializers.constant(self.layer_scale_init_value),
          (self.dim))
      x = x * gamma[..., :]
    drop_pattern = self.get_drop_pattern(x, deterministic=not train)
    x = residual + (1.0 - drop_pattern) * x
    return x

SIZE_OPTIONS = {
    'T': ([3, 3, 9, 3], [96, 192, 384, 768], 0.1),
    'S': ([3, 3, 27, 3], [96, 192, 384, 768], 0.4),
    'B': ([3, 3, 27, 3], [128, 256, 512, 1024], 0.5),
    'L': ([3, 3, 27, 3], [192, 384, 768, 1536], 0.5),
    'XL': ([3, 3, 27, 3], [256, 512, 1024, 2048], 0.5),
}

class ConvNeXt(nn.Module):
  """ConvNeXt architecture.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    size: size as pre-defined in the paper. Options: T, S, B, L
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """
  num_outputs: Optional[int]
  size: str = 'T'
  layer_scale_init_value: float = 1e-6
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False,
      debug: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
       Un-normalized logits.
    """
    if self.size not in SIZE_OPTIONS:
      raise ValueError('Please provide a valid size')
    depths, dims, drop_path_rate = SIZE_OPTIONS[self.size]
    sum_depth = sum(depths)
    dp_rates = [drop_path_rate * i / (sum_depth - 1) for i in range(sum_depth)]
    layernorm = functools.partial(nn.LayerNorm, epsilon=1e-6)
    block = functools.partial(
        ConvNeXtBlock,
        layer_scale_init_value=self.layer_scale_init_value,
        dtype=self.dtype)
    x = nn.Conv(
        dims[0],
        kernel_size=(4, 4),
        strides=(4, 4),
        dtype=self.dtype,
        name='downsample_layers.0.0')(
            x)
    x = layernorm(name='downsample_layers.0.1')(x)
    representations = {'stem': x}
    cur = 0
    for i, (depth, dim) in enumerate(zip(depths, dims)):
      if i > 0:
        x = layernorm(name='downsample_layers.{}.0'.format(i))(x)
        x = nn.Conv(
            dims[i],
            kernel_size=(2, 2),
            strides=(2, 2),
            dtype=self.dtype,
            name='downsample_layers.{}.1'.format(i))(
                x)
      for j in range(depth):
        x = block(
            dim=dim, droplayer_p=dp_rates[cur + j],
            name='stages.{}.{}'.format(i, j))(x, train)
      cur += depth
      representations[f'stage_{i + 1}'] = x

    # Head.
    if self.num_outputs:
      x = jnp.mean(x, axis=(1, 2))
      x = layernorm(name='norm')(x)
      # x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='output_projection')(
              x)
      return x
    else:
      return representations


# Recursively matching layer names
def dfs(k, ori_k, v):
  if isinstance(v, jnp.ndarray):
    torch_data = torch_weights[k]
    if len(v.shape) == 2: # FC layers
      torch_data = np.transpose(torch_data, (1, 0))
    if len(v.shape) == 4: # Conv layers
      torch_data = np.transpose(torch_data, (2, 3, 1, 0))
    return [(k, torch_data.shape)], torch_data
  lst, tree = [], {}
  for kk, vv in v.items():
    if isinstance(vv, jnp.ndarray) and (kk == 'kernel' or kk == 'scale'):
      new_kk = 'weight'
    elif kk == 'output_projection':
      new_kk = 'head'
    else:
      new_kk = kk
    sub_lst, sub_tree = dfs(
        '{}.{}'.format(k, new_kk) if k != '' else new_kk, kk, vv)
    lst.extend(sub_lst)
    tree[kk] = sub_tree
  return lst, tree


# Load pytorch weights (converted state_dict to npy files) in list
#   and convert to Jax weights in tree
MODEL_NAMES = ['convnext_tiny_in22k']

for model_name in MODEL_NAMES:
  print('Processing', model_name)
  model_path = MODEL_URLS[model_name]
  torch_weights = torch.utils.model_zoo.load_url(model_path)['model']
  torch_weights = {k: v.cpu().numpy() for k, v in torch_weights.items()}
  num_params_torch = 0
  for k, v in torch_weights.items():
    num_params_torch += np.prod(v.shape)
  print('num_params_torch', num_params_torch)
  num_class = 21841 if model_name.endswith('_in22k') else 1000
  size = SIZE_MAP[model_name[9: model_name[10:].find('_') + 10]]
  res = 384 if '384' in model_name else 224
  model = ConvNeXt(num_outputs=num_class, size=size)
  x = jnp.zeros((1, res, res, 3))
  rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(0)}
  variables = model.init(rngs, x)
  ret, tree = dfs('', '', variables['params'])
  ret = [(k, v) for k, v in sorted(ret)]
  tot_params = 0
  for k, v in ret:
    tot_params += np.prod(v)
  print('tot_params      ', tot_params)
  tree.keys()
  new_variables = {'params': tree}
  save_path = '{}'.format(model_name)
  checkpoints.save_checkpoint(save_path, new_variables, 0)


