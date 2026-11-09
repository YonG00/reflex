from reflex.world_model.graph_monitor import GraphReplayMonitor


def test_replay_result_and_restoration():
    class Graph:
        def replay(self, value):
            return value

    original = Graph.replay
    monitor = GraphReplayMonitor(Graph)
    value = object()
    assert Graph().replay(value) is value
    assert monitor.count == 1
    monitor.close()
    assert Graph.replay is original
    assert Graph().replay(value) is value
    assert monitor.count == 1


def test_failed_replay_not_counted():
    import pytest

    class Graph:
        def replay(self):
            raise RuntimeError("test")

    monitor = GraphReplayMonitor(Graph)
    try:
        with pytest.raises(RuntimeError):
            Graph().replay()
        assert monitor.count == 0
    finally:
        monitor.close()
