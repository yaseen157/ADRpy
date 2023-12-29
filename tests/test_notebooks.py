"""Unit test that automatically runs the example notebooks."""
import os
import unittest

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class RunNotebooks(unittest.TestCase):

    def test_notebooks(self):
        """
        Tries to run all the code in the notebooks as texts. If it can't, the
        notebook is copied over anyway.

        Notes:
            If you're getting "NoSuchKernel" errors, you probably need to run:

                $ pip install --upgrade pip ipython ipykernel
                $ ipython kernel install --name "python3" --user

        """
        # Locate notebooks
        try:
            notebooks_src = "../docs/ADRpy/notebooks/"
            notebooks_paths = {
                filename: os.path.join(notebooks_src, filename)
                for filename in os.listdir(notebooks_src)
                if filename.endswith(".ipynb")
            }
        except FileNotFoundError as _:
            self.skipTest("Could not locate notebook source directory")
            return
        else:
            if not notebooks_paths:
                self.skipTest(f"Did not find any .ipynb files in source folder")
                return

        # Run every notebook
        failed2open = []
        failed2run = []
        for filename, filepath in notebooks_paths.items():

            # Load the notebook
            try:
                nb = nbformat.read(filepath, as_version=4)
            except UnicodeDecodeError:
                failed2open.append(filename)
                continue

            # Configure a notebook executor
            try:
                client = NotebookClient(
                    nb, timeout=60, kernel_name='python3',
                    resources={'metadata': {'path': notebooks_src}}
                )
            except Exception as e:
                raise Exception("Check advice in this test's notes?") from e

            try:
                # Run the notebook
                client.execute()
            except CellExecutionError:
                # If we have allowed errors, prepare report to the developer
                out = None
                failed2run.append(filename)
            finally:
                # Save the notebook (and any tracebacks)
                nbformat.write(nb, filepath)

        # Pretty printing for unittest runner
        if failed2open or failed2run:
            print("")
        if failed2open:
            print(f"[!] These notebook(s) couldn't be tested:"
                  f"\n\t> " + "\n\t> ".join(failed2open))
        if failed2run:
            print(f"[!] These notebook(s) threw errors:"
                  f"\n\t> " + "\n\t> ".join(failed2run))
            print(f"Saved tracebacks in '{notebooks_src}'")
            self.fail(f"One or more notebooks failed to run")
        print("\n!!! Make sure to check figures have rendered properly !!!\n")

        return


if __name__ == '__main__':
    unittest.main()
