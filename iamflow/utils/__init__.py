"""IAMFlow utility package.

This file makes `utils` a regular package so local imports remain stable
even when the Python environment also provides an unrelated external
`utils` package. This matters for vLLM spawn workers, which re-import the
entry script in a fresh interpreter.
"""

