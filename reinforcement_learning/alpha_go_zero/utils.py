# Code from https://github.com/suragnair/alpha-zero-general
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
