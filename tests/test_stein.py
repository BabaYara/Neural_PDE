from kfacpinn.regularizers.stein import stein_loss


def test_stein():
    assert stein_loss(None, None) == 0.0
