import parallax as par


def test_import_succeeds():
    assert par is not None


def test_version():
    assert isinstance(par.__version__, str)
    assert len(par.__version__) > 0


def test_modules_present():
    assert hasattr(par, "survey")
    assert hasattr(par, "view")
    assert hasattr(par, "catalog")
    assert hasattr(par, "chart")
    assert hasattr(par, "monitor")
    assert hasattr(par, "archive")


def test_types_importable():
    assert par.Candidate is not None
    assert par.Report is not None
    assert par.Criteria is not None
    assert par.CutoutView is not None
    assert par.ViewSession is not None


def test_exception_importable():
    assert par.ParallaxError is not None
    assert issubclass(par.ParallaxError, Exception)


def test_config_accessible():
    assert par.config is not None
    assert hasattr(par.config, "get")
    assert hasattr(par.config, "set")


def test_version_format():
    parts = par.__version__.split(".")
    assert len(parts) == 3
