"""Presentation-only defaults for archived evaluation videos."""


def disable_debug_visualization(config):
    """Hide coordinate frames/contact markers before constructing the environment."""
    seen = set()
    changed = 0

    def visit(obj):
        nonlocal changed
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, type)) or callable(obj):
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'debug_vis' and isinstance(value, bool):
                    changed += int(value)
                    obj[key] = False
                else:
                    visit(value)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                visit(value)
        elif hasattr(obj, '__dict__'):
            for name, value in vars(obj).items():
                if name == 'debug_vis' and isinstance(value, bool):
                    changed += int(value)
                    setattr(obj, name, False)
                elif not name.startswith('_'):
                    visit(value)
    visit(config)
    return changed
