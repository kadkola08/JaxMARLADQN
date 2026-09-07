"""Jumanji environment wrappers for JaxMARL compatibility."""

try:
    from .robot_warehouse_wrapper import RobotWarehouseWrapper
    __all__ = ["RobotWarehouseWrapper"]
except ImportError as e:
    # jumanji not installed
    __all__ = []
    RobotWarehouseWrapper = None
