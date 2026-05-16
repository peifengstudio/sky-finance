from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sky-finance")
except PackageNotFoundError:
    __version__ = "unknown"
