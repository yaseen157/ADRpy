"""


"""
import os
import shutil
import unittest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

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
        for filename, src_path in notebooks_paths.items():

            # Load the notebook
            try:
                with open(src_path) as f:
                    nb = nbformat.read(f, as_version=4)
            except UnicodeDecodeError:
                print(f"Error opening {filename}, skipping test for this file")
                continue

            # Configure a notebook executor
            try:
                ep = ExecutePreprocessor(
                    timeout=60,
                    kernel_name='python3',
                    allow_errors=False  # <-- What makes this unit test useful!
                )
            except Exception as e:
                raise Exception("Check advice in this test's notes?") from e

            try:
                # Run the notebook
                out = ep.preprocess(nb, {'metadata': {'path': notebooks_src}})
            except CellExecutionError:
                out = None
                msg = f'Error executing the notebook "{filename}".\n\n'
                msg += f'See notebook "{src_path}" for the traceback.'
                print(msg)
                raise
            # finally:
            #     with open(out_path, mode='w', encoding='utf-8') as f:
            #         nbformat.write(nb, f)

        return


if __name__ == '__main__':
    unittest.main()


