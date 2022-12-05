from src.transforms import Compose
from src.transforms.augment import RandomScaleAnisotropic, RandomTranslate


def test_random_translate():
    translate = RandomTranslate(delta=[0.5, 0.5, 0.5])
    assert translate.delta == [0.5, 0.5, 0.5]


def test_random_scale():
    scale = RandomScaleAnisotropic(scale=[1.2, 1.5])
    assert scale.scale_max == 1.5
    assert scale.scale_min == 1.2


def test_compose():
    translate = RandomTranslate(delta=[0.5, 0.5, 0.5])
    scale = RandomScaleAnisotropic(scale=[1.2, 1.5])

    compose = Compose([translate, scale])
    assert isinstance(compose.transforms[0], RandomTranslate)
    assert isinstance(compose.transforms[1], RandomScaleAnisotropic)
