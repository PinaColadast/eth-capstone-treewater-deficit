def test_import_treewater_utils():
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    from treewater import utils
    assert hasattr(utils, 'FeatureConfig')
    assert hasattr(utils, 'get_feature_windows')
