"""Deterministic, user-controlled playback of saved detection windows."""
import time
import altair as alt
import pandas as pd
import streamlit as st
from adaptive_defense.replay_state import advance


def render(run, summary, metrics, pred):
    trace = pd.read_csv(run / 'threshold_trace.csv')
    if trace.empty:
        st.info('This experiment has no replay windows.')
        return
    if st.session_state.get('replay_run') != str(run):
        st.session_state.update(replay_run=str(run), replay_play=False, replay_index=1,
                                replay_tick=time.monotonic())
    st.session_state.setdefault('replay_tick', time.monotonic())

    def toggle_play():
        if st.session_state.replay_index >= len(trace):
            st.session_state.replay_index = 1
        st.session_state.replay_play = not st.session_state.replay_play
        st.session_state.replay_tick = time.monotonic()

    def reset_replay():
        st.session_state.update(replay_play=False, replay_index=1, replay_tick=time.monotonic())

    def seek():
        st.session_state.update(replay_play=False, replay_index=st.session_state.replay_seek,
                                replay_tick=time.monotonic())

    st.subheader('Traffic playback')
    st.caption('Recorded decisions, not a live network feed. Playback changes only what you see.')
    controls = st.columns([1.2, 1, 2])
    controls[0].button('Pause' if st.session_state.replay_play else 'Play replay',
                       on_click=toggle_play, type='primary', width='stretch')
    controls[1].button('Reset', on_click=reset_replay, width='stretch')
    speed = controls[2].selectbox('Playback speed', ['0.5×', '1×', '2×'], index=1)
    interval = {'0.5×': 1.8, '1×': .9, '2×': .45}[speed]
    if len(trace) > 1:
        st.session_state.replay_seek = st.session_state.replay_index
        st.slider('Jump to window', 1, len(trace), key='replay_seek', on_change=seek,
                  disabled=st.session_state.replay_play)

    @st.fragment(run_every=.2 if st.session_state.replay_play else None)
    def replay_view():
        was_playing = st.session_state.replay_play
        index, playing, tick = advance(st.session_state.replay_index, len(trace),
                                      was_playing, st.session_state.replay_tick,
                                      time.monotonic(), interval)
        st.session_state.update(replay_index=index, replay_play=playing, replay_tick=tick)
        st.progress(index / len(trace), text=f'Window {index} of {len(trace)}')
        visible = trace.iloc[:index].copy()
        visible['window'] = range(1, index + 1)
        end = int(visible.iloc[-1]['end'])
        a, b, c = st.columns(3)
        a.metric('Flows reviewed', f'{end:,}')
        b.metric('Alerts raised', f'{int(visible.alerts.sum()):,}')
        c.metric('Current threshold', f'{visible.iloc[-1].threshold:.4f}')
        chart = alt.Chart(visible).mark_line(point=True, color='#347459').encode(
            x=alt.X('window:Q', title='Window', scale=alt.Scale(domain=[1, max(2, len(trace))])),
            y=alt.Y('threshold:Q', title='Threshold', scale=alt.Scale(domain=[0, max(float(trace.threshold.max())*1.1, .001)])),
            tooltip=['window', 'threshold', 'alerts'],
        ).properties(height=240)
        st.altair_chart(chart, width='stretch')
        st.caption('Fixed axes keep the chart steady while playback progresses.')
        st.dataframe(pred.head(end).tail(20)[['attack_cat', 'anomaly_score', 'DQN frozen']],
                     width='stretch', hide_index=True)
        if was_playing and not playing:
            st.rerun()
    replay_view()
