from types import SimpleNamespace
import jax
import jax.numpy as jnp

from neural_pde.models import FBSDE
from neural_pde.hjbcore.loss_suite import LossSuite

params = SimpleNamespace(num_trees=2, d_noise_dim=1)


def test_forward_pass():
    model = FBSDE(jax.random.PRNGKey(0))
    batch = {
        "state": jnp.ones((4, params.num_trees)),
        "brownian": jnp.zeros((4, 2, params.d_noise_dim)),
    }
    cfg = {"dt": 1e-8, "params": params, "λσ2": 1e-3, "λμ": 1e-3}
    loss = LossSuite(cfg).total_loss(model, batch)
    assert jnp.isfinite(loss) and loss > 0
