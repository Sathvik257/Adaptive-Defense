"""Experimental calibrated ADT gate with budgeted, delayed feedback.

The gate uses an independent benign calibration sample. Its marginal false-alarm
interpretation requires exchangeability; this is not a guarantee under drift.
"""
from collections import deque
import math
import numpy as np
from .adt import History


def calibrated_threshold(benign_scores, alpha=.05):
    values = np.asarray(benign_scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need nonempty finite benign calibration scores.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else float('inf')


def gated_replay(agent, scores, threshold, *, feedback=None, budget=.05, delay=32, k=2, review_strategy="periodic"):
    """Frozen ADT policy, raw-score gate, and queried labels delivered later.

    feedback(index) is called only for a queried row once its delay has elapsed.
    Periodic querying is score-independent. Boundary querying selects the closest
    score to the threshold from flows seen since the previous request; it never
    inspects future flows or labels. Delay starts at request time. Budget is a strict
    prefix cap. Delay counts replay rows, not wall-clock time. Feedback counts
    describe the last k delivered reviews of final gated predictions; score
    statistics describe the last k flows. This differs from the paper state. No gradient updates occur here.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Need nonempty finite nonnegative raw scores.')
    if np.isnan(threshold) or threshold < 0 or not 0 <= budget <= 1 or not isinstance(delay, int) or delay < 1 or k < 1:
        raise ValueError('Invalid threshold, budget, delay or history size.')
    if review_strategy not in ("periodic", "boundary"):
        raise ValueError("Unknown review strategy.")
    candidates = []
    recent, pending, reviewed = deque(maxlen=k), deque(), deque(maxlen=k)
    predictions, trace = [], []
    queried = delivered = 0
    # Label feedback is sparse; score statistics always use the last k flows.
    for t, score in enumerate(scores):
        while pending and pending[0][0] <= t:
            _, index, record = pending.popleft()
            label = feedback(index)
            if label not in (0, 1):
                raise ValueError('Feedback must be binary.')
            record[2] = int(label)
            reviewed.append(tuple(record))
            delivered += 1
        history = History(k)
        history.items.extend(tuple(row) for row in recent)
        state = history.state()
        feedback_history = History(k)
        feedback_history.items.extend(reviewed)
        state[2:] = feedback_history.counts() / k
        action = agent.action(state)
        prediction = int(action == 0 and score > threshold)
        # Keep the original policy score representation, but gate on raw scores.
        record = [float(min(score, 1)), prediction, None]
        recent.append(record)
        if feedback is not None and budget > 0:
            candidates.append((t, record))
        request = feedback is not None and queried < math.floor((t + 1) * budget + 1e-12)
        if request:
            queried += 1
            selected_index, selected_record = (t, record) if review_strategy == "periodic" else min(
                candidates, key=lambda item: (abs(scores[item[0]] - threshold), item[0]))
            pending.append((t + delay, selected_index, selected_record))
            candidates.clear()
        predictions.append(prediction)
        trace.append({'index': t, 'action': int(action), 'prediction': prediction,
                      'queried': bool(request), 'review_index': selected_index if request else None,
                      'delivered': delivered})
    return np.asarray(predictions), trace
