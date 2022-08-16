import json


# -----------------------------
# Utilities to override methods
# -----------------------------
def overload(func):
    func.is_overloaded = True
    return func

def is_method_overloaded(func):
    if not hasattr(func, 'is_overloaded'):
        return False
    else:
        return func.is_overloaded

def init_overload_state(func):
    if not is_method_overloaded(func):
        func.is_overloaded = False
    return func


# ---------------
# Text formatting
# ---------------
def prettify_dict(d):
    return json.dumps(d, indent=2)
