import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).absolute().parent
VERSION_FILE = HERE / 'safe_rlhf_v' / 'version.py'

def get_version():
    version_globals = {}
    if VERSION_FILE.exists():
        with open(VERSION_FILE, encoding="utf-8") as f:
            exec(f.read(), version_globals)
    return version_globals.get("__version__", "0.0.1")

__version__ = get_version()

setup(
    name="safe_rlhf_v",
    version=__version__,
)
