"""Cache PhysX metadata until its native setter invalidates the exact property."""


class ContextPropertyCache:
    properties = ("masses", "inertias", "coms", "material_properties")

    def __init__(self, assets, device="cpu"):
        self.values = {}
        self.device = device
        self.views = {label: asset.root_physx_view for label, asset in assets.items()}
        self.reads = self.hits = self.invalidations = 0
        for label, asset in assets.items():
            view = asset.root_physx_view
            for prop in self.properties:
                setter = getattr(view, "set_" + prop)
                key = (label, prop)

                def write(*args, _setter=setter, _key=key, **kwargs):
                    # Invalidate even if a partial setter fails after writing.
                    self.values.pop(_key, None)
                    self.invalidations += 1
                    return _setter(*args, **kwargs)

                setattr(view, "set_" + prop, write)

    def get(self, label, asset, prop):
        if asset.root_physx_view is not self.views[label]:
            raise RuntimeError("PhysX view changed; rebuild context cache")
        key = (label, prop)
        if key not in self.values:
            self.values[key] = (
                getattr(asset.root_physx_view, "get_" + prop)().to(self.device).clone()
            )
            self.reads += 1
        else:
            self.hits += 1
        return self.values[key]
