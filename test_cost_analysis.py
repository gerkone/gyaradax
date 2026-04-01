import jax
import jax.numpy as jnp
@jax.jit
def f(x): return jnp.dot(x, x) + 2.0
x = jnp.ones((100, 100))
c = f.lower(x).compile()
print(c.cost_analysis())
