from typing import Dict, Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class GKPre:
    """precomputed terms container. separates dynamic arrays (leaves) from
    static metadata (auxiliary) so FFT sizes stay concrete under JIT."""

    def __init__(self, items: Dict[str, Any]):
        self._items = items

    def tree_flatten(self):
        leaves = []
        leaf_keys = []
        aux = {}
        for k, v in self._items.items():
            if k.startswith("nl_m") or k in ("ixzero", "iyzero", "nsp"):
                aux[k] = v
            elif isinstance(v, (jnp.ndarray, float, int, bool)):
                leaves.append(v)
                leaf_keys.append(k)
            else:
                aux[k] = v
        return tuple(leaves), {"leaf_keys": tuple(leaf_keys), "aux": aux}

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        items = dict(zip(metadata["leaf_keys"], leaves))
        items.update(metadata["aux"])
        return cls(items)

    def __getitem__(self, key):
        return self._items[key]

    def get(self, key, default=None):
        return self._items.get(key, default)

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKState:
    """Diagnostic state for large-step growth tracking and normalization."""

    time: jnp.ndarray
    step: jnp.ndarray
    accumulated_norm_factor: jnp.ndarray
    window_start_amp: jnp.ndarray
    last_growth_rate: jnp.ndarray

    def tree_flatten(self):
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(*leaves)