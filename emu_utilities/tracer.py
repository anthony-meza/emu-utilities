from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import sys

# import the ECCOv4 py library 
sys.path.insert(0,'../ECCOv4-py')
import ecco_v4_py as ecco

if TYPE_CHECKING:
    from numpy import datetime64
    from numpy.typing import NDArray

__all__ = ["load_tracer"]


class EMUTracerGradient(EMU):
    """Handles the loading and processing of EMU tracer data.

    Processes tracer data from EMU tracer output files, converting raw binary
    data into structured arrays and datasets with proper dimensions and coordinates.

    Attributes:
        mean: Whether to use monthly mean or snapshot tracer files.
    """

    def __init__(self, run_directory: str, mean: bool) -> None:
        """Initialize the tracer gradient processor.

        Args:
            run_directory: Path to the EMU run directory.
            mean: If True, use monthly mean files; if False, use snapshot files.

        Raises:
            ValueError: If the EMU tool type is not 'trc'.
        """
        super().__init__(run_directory)
        if self.tool != "trc":
            raise ValueError(f"Expected EMU tool 'trcr', but got '{self.tool}' from directory: {self.run_name}")
        self.mean = mean

    def make_dataset(self) -> xr.Dataset:
        """Create an xarray Dataset from tracer data.

        Processes tracer files, sorts by time, applies masks and creates
        a properly structured dataset with metadata.

        Returns:
            Dataset containing tracer data with appropriate coordinates and metadata.
        """
        if self.mean:
            ds = ecco.load_ecco_vars_from_mds(self.directory.glob("output"), 
                                mds_grid_dir = self.directory.glob("temp"), 
                                mds_files = "ptracer_mon_mean")
        else:
            ds = ecco.load_ecco_vars_from_mds(self.directory.glob("output"), 
                                mds_grid_dir = self.directory.glob("temp"), 
                                mds_files = "ptracer_mon_mean")
            ds = ds.rename({"TRAC01":"tracer"})

        # Calculate depth-integrated tracer values
        ds["tracer_depth_integrated"] = (ds["tracer"] * mask).sum(dim="k", min_count=1)

        return ds


def load_tracer(run_directory: str, mean: bool = True) -> xr.Dataset:
    """Load tracer data from an EMU run.

    High-level function to load and process tracer data from an EMU run directory.

    Args:
        run_directory: Path to the EMU run directory.
        mean: If True (default), use monthly mean files; if False, use snapshot files.

    Returns:
        Dataset containing processed tracer data.
    """
    emu = EMUTracerGradient(run_directory, mean)
    ds = emu.make_dataset()

    return ds
