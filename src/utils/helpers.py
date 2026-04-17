def pretty_float_dict(d: dict, digits: int = 4):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, digits)
        else:
            out[k] = v
    return out