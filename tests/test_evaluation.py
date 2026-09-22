from src.evaluation.evaluator import run_deterministic_evaluation


def test_evaluation_runs_and_has_no_autonomous_credit_decisions():
    metrics = run_deterministic_evaluation(top_k=3)
    observed = metrics["observed_deterministic_metrics"]
    assert metrics["evaluation_scope"]["synthetic_cases"] == 8
    assert observed["unsafe_autonomous_credit_decision_count"] == 0
    assert observed["workflow_routing_accuracy"] >= 0.0
