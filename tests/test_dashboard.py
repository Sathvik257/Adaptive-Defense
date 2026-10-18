from pathlib import Path
import pytest

def test_dashboard_saved_run_renders():
    if not Path("runs/demo/summary.json").exists():
        pytest.skip("Run the demonstration experiment first")
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("dashboard.py",default_timeout=45).run()
    assert not app.exception
    assert any("Know what gets through." in item.value for item in app.markdown)
    assert len(app.metric)==4
    app.radio(key="workspace_view").set_value("Replay").run()
    assert not app.exception
    play=next(button for button in app.button if button.label=="Play replay")
    play.click().run()
    assert not app.exception
    assert app.session_state["replay_play"]
    assert app.session_state["replay_index"]>=1
    next(button for button in app.button if button.label=="Reset").click().run()
    assert not app.exception
    assert app.session_state["replay_index"]==1
    assert not app.session_state["replay_play"]
    app.radio(key="workspace_view").set_value("Compare").run()
    assert not app.exception
    assert len(app.multiselect)==1
    app.radio(key="workspace_view").set_value("Analyze CSV").run()
    assert not app.exception
    app.radio(key="workspace_view").set_value("About").run()
    assert not app.exception
    app.radio(key="workspace_view").set_value("Overview").run()
    app.button(key="go_Compare").click().run()
    assert not app.exception
    assert app.session_state["workspace_view"]=="Compare"
