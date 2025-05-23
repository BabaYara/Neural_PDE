import jax
from kfacpinn.nets.mlp import init_mlp, mlp
from kfacpinn.training_loop import Trainer


def test_training():
    key = jax.random.PRNGKey(0)
    params = init_mlp([1, 2, 1], key)
    state = None
    trainer = Trainer(model=None, regs=["mallavin"])
    trainer.fit(params, state, num_steps=1)
