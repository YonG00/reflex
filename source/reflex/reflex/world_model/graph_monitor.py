"""Count actual public CUDAGraph.replay calls without depending on CUPTI."""

from functools import wraps

import torch


class GraphReplayMonitor:
    def __init__(self, graph_type=None):
        self.owner = graph_type or torch.cuda.CUDAGraph
        self.original = self.owner.replay
        self.count = 0
        if getattr(self.original, "_reflex_monitor", False):
            raise RuntimeError("CUDA Graph replay monitor already installed")

        @wraps(self.original)
        def replay(graph, *args, **kwargs):
            result = self.original(graph, *args, **kwargs)
            self.count += 1
            return result

        replay._reflex_monitor = True
        self.wrapper = replay
        self.owner.replay = replay

    def close(self):
        if self.owner.replay is self.wrapper:
            self.owner.replay = self.original
