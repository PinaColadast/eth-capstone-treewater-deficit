# Example: importing the package inside a notebook or script

# Option 1: if you've installed the package editable (pip install -e .) just import
# from treewater.utils import FeatureConfig

# Option 2: during development, add src folder to sys.path
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
src_path = str(repo_root / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from treewater.utils import FeatureConfig, compute_recursive_predictions_fast

print('Imported:', FeatureConfig, compute_recursive_predictions_fast)
