"""Shared event decisions and write provenance for native and imagined execution."""


def rethrow_mask(common_step, count, grace):
    return (count >= grace) & (common_step % 5 == 0)


def event_write(asset, method, *args, write_mask=None, **kwargs):
    """Annotate intervention provenance without changing the native API call.

    False fields explicitly pass through the current state. Constant writes,
    including equal-valued zero writes, remain True. The native call receives
    precisely its original arguments; only the WM recorder reads this metadata.
    """
    previous = getattr(asset, "_reflex_event_write_mask", None)
    asset._reflex_event_write_mask = write_mask
    try:
        return getattr(asset, method)(*args, **kwargs)
    finally:
        asset._reflex_event_write_mask = previous
