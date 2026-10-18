import pytest
from adaptive_defense.replay_state import advance

def test_render_does_not_advance_before_interval():
    assert advance(1,9,True,10,10.2,.9)==(1,True,10)

def test_tick_advances_one_window():
    assert advance(1,9,True,10,11,.9)==(2,True,11)

def test_pause_preserves_displayed_window():
    assert advance(4,9,False,10,30,.9)==(4,False,10)

def test_completion_stops_at_last_window():
    assert advance(8,9,True,10,11,.9)==(9,False,11)

def test_slow_browser_does_not_skip_windows():
    assert advance(2,9,True,10,100,.9)==(3,True,100)

def test_single_window_stops():
    assert advance(1,1,True,10,10.2,.9)==(1,False,10)

@pytest.mark.parametrize("total,interval",[(0,.9),(9,0)])
def test_invalid_clock_rejected(total,interval):
    with pytest.raises(ValueError):
        advance(1,total,True,0,1,interval)
