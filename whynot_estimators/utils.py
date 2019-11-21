"""Utitility tools for working with whynot_estimators."""


def extract(listvector, argname):
    """Retrieve argument argname from R listvector."""
    if argname not in listvector.names:
        raise ValueError(f"{argname} not found.")
    index = list(listvector.names).index(argname)
    return listvector[index]
