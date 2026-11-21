
def build_my_acq_optimizer(func_str='local_random', config_space=None, rng=None, rand_prob=0.25):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        from .basic_maximizer import InterleavedLocalAndRandomSearchMaximizer
        return InterleavedLocalAndRandomSearchMaximizer(config_space=config_space, rng=rng, rand_prob=rand_prob)
    else:
        raise ValueError('Invalid string %s for acq_optimizer!' % func_str)

