# standard imports
import sys
import os


class HiddenPrints:
    """
    A class for a context manager to suppress stdout output.
    """

    def __enter__(self):
        """
        Suppresses stdout output.
        """

        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores stdout output.
        """

        sys.stdout.close()
        sys.stdout = self._original_stdout

