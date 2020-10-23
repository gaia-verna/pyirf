import logging

from astropy.table import QTable, unique
import astropy.units as u
import pandas as pd

from ..simulations import SimulatedEventsInfo

log = logging.getLogger(__name__)

def read_protopipe_hdf5(infile, run_header):
    """
    Read a DL2 HDF5 file as produced by the protopipe pipeline:
    https://github.com/cta-observatory/protopipe

    Parameters
    ----------
    infile: str or pathlib.Path
        Path to the input fits file
        
    run_header: dict
        Dictionary with info about simulated particle informations

    Returns
    -------
    events: astropy.QTable
        Astropy Table object containing the reconstructed events information.
    simulated_events: ``~pyirf.simulations.SimulatedEventsInfo``

    """
    log.debug(f"Reading {infile}")
    df = pd.read_hdf(infile, "/reco_events")

    # These values are hard-coded at the moment
    true_alt = [70] * len(df)
    true_az = [180] * len(df)
    pointing_alt = [70] * len(df)
    pointing_az = [180] * len(df)

    events = QTable([list(df['obs_id']),
                     list(df['event_id']),
                     list(df['xi']) * u.deg, 
                     list(df['mc_energy']) * u.TeV, 
                     list(df['reco_energy']) * u.TeV,
                     list(df['gammaness']),
                     list(df['NTels_reco']),
                     list(df['reco_alt']) * u.deg,
                     list(df['reco_az']) * u.deg,
                     true_alt * u.deg,
                     true_az * u.deg,
                     pointing_alt * u.deg,
                     pointing_az * u.deg,
                    ],
                    names=('obs_id',
                           'event_id',
                           'theta',
                           'true_energy', 
                           'reco_energy', 
                           'gh_score',
                           'multiplicity',
                           'reco_alt',
                           'reco_az',
                           'true_alt',
                           'true_az',
                           'pointing_alt',
                           'pointing_az',   
                          ),
                   )
    
    n_runs = len(set(events['obs_id']))
    log.info(f"Estimated number of runs from obs ids: {n_runs}")

    n_showers = n_runs * run_header["num_use"] * run_header["num_showers"]
    log.debug(f"Number of events from n_runs and run header: {n_showers}")

    sim_info = SimulatedEventsInfo(
        n_showers=n_showers,
        energy_min=u.Quantity(run_header["e_min"], u.TeV),
        energy_max=u.Quantity(run_header["e_max"], u.TeV),
        max_impact=u.Quantity(run_header["gen_radius"], u.m),
        spectral_index=run_header["gen_gamma"],
        viewcone=u.Quantity(run_header["diff_cone"], u.deg),
    )

    return events, sim_info
