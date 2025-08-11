def test_imports():
    import importlib
    assert importlib.import_module("app.app")
    assert importlib.import_module("src.inference")