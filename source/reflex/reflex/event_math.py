"""Pure decisions shared by native and imagined event execution."""


def rethrow_mask(common_step, count, grace):
    return (count >= grace) & (common_step % 5 == 0)
