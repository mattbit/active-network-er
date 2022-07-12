import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Where the simulations results will be saved.
DATA_PATH = Path(os.getenv('ER_DATA_PATH'))
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Where the generated figures will be saved.
FIGURES_PATH = Path(os.getenv('ER_FIGURES_PATH'))
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
