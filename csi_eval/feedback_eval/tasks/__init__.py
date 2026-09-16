"""Task adapters: pluggable task definitions.

Tasks are NOT auto-imported here. To use the built-in eigenvector_feedback
task, the user explicitly imports it:

        from ..tasks import eigenvector_feedback

After this import, ``TaskRegistry.create("eigenvector_feedback")`` works.
This keeps the package import graph shallow and lets users opt into
only the tasks they need.

To add a new task, create ``tasks/<task_name>/__init__.py`` and decorate
your class with ``@TaskRegistry.register("<task_name>")``.
"""

__all__: list = []
