from pathlib import Path

# Where the simulations results will be saved.
DATA_PATH = Path(__file__).parent.joinpath('resources/data')
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Where the generated figures will be saved.
FIGURES_PATH = Path(__file__).parent.joinpath('resources/figures')
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
