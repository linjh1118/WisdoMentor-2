import sys
import os

sys.path.append(f"{sys.path[0]}/..")

from eval.recall_evaluator import RecallEvaluator

evaluator = RecallEvaluator(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "recall_evaluate",
        "recall_evaluate.json",
    )
)
evaluator.evaluate()
