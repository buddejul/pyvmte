"""Tasks for compiling the paper and presentation(s)."""
import shutil

import pytask
from pytask_latex import compilation_steps as cs
from pyvmte.config import BLD, PAPER_DIR

documents = ["pyvmte"]

# Note: Explicitly define dependencies to ensure pytask parallel works without error
# on the first run.
_DEPENDENCIES = {
    1: BLD / "python" / "figures" / "simulation_results_by_target.png",
    2: BLD
    / "python"
    / "figures"
    / "simulation_results_by_sample_size_lower_bound_figure5.png",
    3: BLD
    / "python"
    / "figures"
    / "simulation_results_by_sample_size_upper_bound_figure5.png",
}

for document in documents:

    @pytask.mark.latex(
        script=PAPER_DIR / f"{document}.tex",
        document=BLD / "latex" / f"{document}.pdf",
        compilation_steps=cs.latexmk(
            options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
        ),
    )
    @pytask.task(id=document)
    def task_compile_document(dependencies_to_ensure_parallel_works=_DEPENDENCIES):
        """Compile the document specified in the latex decorator."""

    kwargs = {
        "depends_on": BLD / "latex" / f"{document}.pdf",
        "produces": BLD.parent.resolve() / f"{document}.pdf",
    }

    @pytask.task(id=document, kwargs=kwargs)
    def task_copy_to_root(depends_on, produces):
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)
