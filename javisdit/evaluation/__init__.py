"""Evaluation APIs that are safe to import from training code."""

__all__ = ["evaluate_av_score"]


def __getattr__(name):
    if name == "evaluate_av_score":
        from .javisbench_av_score import evaluate_av_score

        return evaluate_av_score
    raise AttributeError(name)
