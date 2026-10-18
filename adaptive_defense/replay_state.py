"""Pure replay clock: rendering never silently advances past a paused frame."""
def advance(index, total, playing, last_tick, now, interval):
    if total < 1:
        raise ValueError("Replay requires at least one window.")
    if interval <= 0:
        raise ValueError("Replay interval must be positive.")
    index = min(max(index, 1), total)
    if not playing:
        return index, False, last_tick
    if now - last_tick >= interval:
        index = min(index + 1, total)
        last_tick = now
    return index, index < total, last_tick
