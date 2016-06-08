def combinatorial(func, features, n, start=0):
    num_features = len(features) - start
    if num_features > 0 and n > 0:
        for y in combinatorial(func, features, n, start + 1):
            yield y
        x = features[start]
        for d in range(1, n):
            for y in combinatorial(func, features, n - d, start + 1):
                yield func(x, y)
            x = func(x, features[start])
        yield x


def polynomial(degree, features, complete_polynomy=True, constant_term=False):
    if constant_term:
        assert len(features) > 0
        yield nparray([1] * len(features[0]))

    features = nparray(features)

    if complete_polynomy:
        init = 1
    else:
        init = degree

    for d in range(init, degree + 1):
        for term in combinatorial(lambda x, y: x * y, features, d):
            yield term
