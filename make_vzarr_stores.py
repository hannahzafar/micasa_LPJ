#!/usr/bin/env python

from pathlib import Path

from virtualizarr_nc4_toolkit.virtualize import create_vzarr_store

# Turn this into an argparse???
# Have to use absolute paths
micasa_path = "/css/gmao/geos_carb/pub/MiCASA/v1/netcdf/"
micasa_files = sorted(list(Path(f"{micasa_path}daily/").glob("*/*/*.nc4")))
create_vzarr_store("micasa_virtualized", micasa_files)
