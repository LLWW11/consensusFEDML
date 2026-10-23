SELECT method, round, mrr, passes FROM training_rounds ORDER BY method, round;

SELECT method, round, mrr, passes FROM training_rounds WHERE passes <= 122.83016064950769 ORDER BY method, round;

SELECT method, self_fraction FROM self_diagnostics ORDER BY method;
