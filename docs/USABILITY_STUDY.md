# Five-participant usability protocol

Status: not conducted. Recruit five classmates voluntarily; explain that this
evaluates the interface, not their skill. Use anonymous IDs P01–P05, no names.
Do not coach participants while timing tasks; record assistance and failures.

Tasks: start and pause replay; inspect one alert; explain score versus threshold;
request a review; save an assessment; find the false-positive rate of the frozen
DQN. Ask whether feedback has retrained the model (correct answer: no).
Record completion, seconds, mistakes, help requested and short optional comments.

Administer the unchanged ten-item System Usability Scale after the tasks, with
responses 1 (strongly disagree) to 5 (strongly agree). Use the original instrument:
https://digital.ahrq.gov/sites/default/files/docs/survey/systemusabilityscale%2528sus%2529_comp%255B1%255D.pdf

Save CSV headers participant_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10. Run
`python tools/score_sus.py responses.csv`. Odd items contribute response−1;
even items contribute 5−response; multiply the sum by 2.5. Do not impute missing
answers. Report individual anonymous scores, mean, sample SD and task success.
Five convenience participants do not establish population-wide usability.
Never present SUS as percent satisfied, and do not invent scores for the report.
