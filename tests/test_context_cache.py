from types import SimpleNamespace

import pytest
import torch
from reflex.world_model.context_cache import ContextPropertyCache


class View:
    def __init__(self):
        self.data = {p: torch.ones(4, 3) for p in ContextPropertyCache.properties}
        for p in ContextPropertyCache.properties:
            setattr(self, "get_" + p, lambda p=p: self.data[p])

            def write(value, indices, p=p):
                self.data[p][indices] = value

            setattr(self, "set_" + p, write)


def test_physical_cache_invalidates_partial_native_write_without_cross_asset_changes():
    assets = {p: SimpleNamespace(root_physx_view=View()) for p in ("robot", "box")}
    cache = ContextPropertyCache(assets)
    for label, asset in assets.items():
        for prop in cache.properties:
            torch.testing.assert_close(
                cache.get(label, asset, prop), asset.root_physx_view.data[prop]
            )
    assert cache.reads == 8
    cache.get("robot", assets["robot"], "masses")
    assert cache.reads == 8 and cache.hits == 1
    assets["box"].root_physx_view.set_masses(torch.full((1, 3), 7.0), torch.tensor([2]))
    torch.testing.assert_close(
        cache.get("box", assets["box"], "masses"),
        assets["box"].root_physx_view.data["masses"],
    )
    assert cache.reads == 9 and cache.invalidations == 1
    torch.testing.assert_close(
        cache.get("robot", assets["robot"], "masses"), torch.ones(4, 3)
    )
    assets["robot"].root_physx_view = View()
    with pytest.raises(RuntimeError, match="view changed"):
        cache.get("robot", assets["robot"], "masses")
