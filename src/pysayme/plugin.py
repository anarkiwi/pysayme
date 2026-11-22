"""pytest plugin: one ``pysayme`` test item on full runs of a project whose
pyproject.toml has a ``[tool.pysayme]`` table, or whenever ``--pysayme`` is given."""

from __future__ import annotations

import pytest

from . import config as _config
from . import gate


def pytest_addoption(parser):
    group = parser.getgroup("pysayme")
    group.addoption(
        "--pysayme",
        action="store_true",
        default=None,
        help="check for code duplicating dependencies even when tests are selected",
    )
    group.addoption(
        "--no-pysayme",
        action="store_true",
        default=False,
        help="do not check for code duplicating dependencies",
    )


def _pyproject(config):
    inipath = config.inipath
    if inipath is not None and inipath.name == "pyproject.toml":
        return inipath
    return config.rootpath / "pyproject.toml"


def pytest_configure(config):
    config.addinivalue_line("markers", "pysayme: dependency-duplication gate")
    if config.option.no_pysayme:
        config.option.pysayme = False
    elif config.option.pysayme is None:
        source = getattr(config, "args_source", None)
        full = source is not pytest.Config.ArgsSource.ARGS
        config.option.pysayme = "auto" if full else False


class GateFile(pytest.File):
    """pyproject.toml, holding the gate item."""

    def collect(self):
        yield GateItem.from_parent(self, name="pysayme")


class GateItem(pytest.Item):
    """Fails listing duplicated dependency code and stale allowlist entries."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.add_marker("pysayme")

    def runtest(self):
        result = gate.check(_config.load_config(self.path, progress=False))
        if not result.ok:
            raise GateFailure(str(result))

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, GateFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        return self.path, None, "pysayme"


class GateFailure(Exception):
    """The gate found duplicates or allowlist problems."""


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    pyproject = _pyproject(config)
    want = config.option.pysayme
    if want is True or (want == "auto" and _config.has_section(pyproject)):
        items += GateFile.from_parent(session, path=pyproject).collect()
