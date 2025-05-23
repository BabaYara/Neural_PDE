from kfacpinn.regularizers.mallavin import malliavin_loss


def test_malliavin():
    assert malliavin_loss(None, None) == 0.0
