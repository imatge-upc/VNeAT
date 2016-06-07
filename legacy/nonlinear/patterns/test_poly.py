from functions import polynomial


class LIST(list):
    def __init__(self, *args, **kwargs):
        try:
            list.__init__(self, *args, **kwargs)
        except TypeError:
            list.__init__(self, args, **kwargs)

    def __mul__(self, other):
        return LIST(self + other)


d = polynomial(4, map(LIST, [0, 1, 2]))
