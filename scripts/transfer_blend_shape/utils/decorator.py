from functools import wraps


def memoize(func):
    """
    The memoize decorator will cache the result of a function and store it
    in a cache dictionary using its arguments and keywords arguments as a key.
    The cache can be cleared by calling the cache_clear function on the
    decorated function.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    def clear():
        cache.clear()

    wrapper.clear = clear
    return wrapper
