import importlib

this_module = importlib.import_module(__name__)

foo = 1


def __getattr__(name):
    import_name = "." + name
    try:
        imported = importlib.import_module(
            import_name, this_module.__spec__.parent
        )
    except ModuleNotFoundError:
        raise AttributeError(f"'{__name__}' has no attribute '{name}'")
    return imported
