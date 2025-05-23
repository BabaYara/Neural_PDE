import jax.numpy as jnp

from kfacpinn.optim.kfac import KFAC


def test_kfac_update():
    params = [(jnp.ones((2, 2)), jnp.zeros(2))]
    grads = [(jnp.ones((2, 2)), jnp.ones(2))]
    opt = KFAC(lr=0.1)
    state = opt.init(params)
    new_params, new_state = opt.update(params, state, grads)
    assert not jnp.allclose(new_params[0][0], params[0][0])
