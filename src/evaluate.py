# local imports
from evaluation.evaluator import Evaluator


def evaluate():
    """
    The main function to run the evaluation process.

    It initializes the :py:class:`src.evaluation.evaluator.Evaluator` class,
    and executes the evaluation using :py:meth:`src.evaluation.evaluator.Evaluator.evaluate`.
    """

    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    evaluate()
