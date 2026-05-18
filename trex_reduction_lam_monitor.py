import numpy as np
import mcstasscript as ms
import mcstastox as mx
import scipp as sc
from scipp.typing import VariableLike
from scippneutron.conversion.beamline import _canonical_length
from scipy.ndimage import label

# Utility Functions

def v_bin_norm(binner, event_object, vanad_object, mag_Q_dim = 'mag_Q'):

    if type(binner) != list:
        binner = [binner]

    hist_kwargs = {bin_dim.dims[0]: bin_dim for bin_dim in binner}
    hist_event = event_object.bin(**hist_kwargs).hist(**hist_kwargs)
    hist_vanad = vanad_object.bin(**hist_kwargs).hist(**hist_kwargs)

    if mag_Q_dim in hist_kwargs:
        Q_dict = {mag_Q_dim:hist_kwargs[mag_Q_dim]}
        vanad_q_mag = vanad_object.bin(**Q_dict).hist(**Q_dict)
        hist_norm = hist_event / vanad_q_mag
    else:
        print(f'dimension: {mag_Q_dim} not found, not normalising over q')
        vanad_q_mag = 1
        hist_norm  = np.nan

    return hist_event, hist_vanad, hist_norm


def _calc_pulse_centroid(tof_monitor,threshold=0, to_s=1e-6):
    # Assumes all pulses are evenly spaced in xaxis - but isn't bin data??? confirm this
    # Confirm correct way to axis tof data - is xaxis correct????? 
    mask = tof_monitor.Intensity > threshold
    labels, num_features = label(mask)
    weighted_sum = np.bincount(labels, weights=tof_monitor.xaxis * tof_monitor.Intensity)[1:]
    weight_total = np.bincount(labels, weights=tof_monitor.Intensity)[1:]

    coms = weighted_sum / weight_total * to_s
    coms = sc.array(dims=['tof'], values=coms, unit='s')
    return coms

def create_time_lamda_mappings(lambda_tom_monitor, n_bins, threshold):
    """
    lambda tom monitor is the 2D monitor with lambda and time axes i.e.: 
    lambda bins = 401 limits [3.8 : 6.2], t bins= 501 limits [0.16 : 0.26]
    """

    lambda_axis = np.linspace(*lambda_tom_monitor.metadata.limits[0:2], lambda_tom_monitor.metadata.dimension[0])
    time_axis = np.linspace(*lambda_tom_monitor.metadata.limits[2:], lambda_tom_monitor.metadata.dimension[1])
    time_intensity = lambda_tom_monitor.Intensity.sum(axis=1)


    mask = time_intensity > threshold
    labels, num_features = label(mask)
    weighted_sum = np.bincount(labels, weights=time_axis * time_intensity)[1:]
    weight_total = np.bincount(labels, weights=time_intensity)[1:]

    centre_times = weighted_sum / weight_total
    n_bins = n_bins

    centre_lambdas = np.zeros(num_features)

    for i in range(1,num_features+1):

        idx = np.where(labels == i)[0]
        expanded_idx = np.arange(max(0, idx[0] - n_bins), min(len(time_axis), idx[-1] + n_bins + 1))
        lambda_selected_intensity = lambda_tom_monitor.Intensity[expanded_idx,:].sum(axis=0)
        centre_lambda = np.sum(lambda_axis * lambda_selected_intensity) / np.sum(lambda_selected_intensity)
        centre_lambdas[i-1] = centre_lambda


    if len(centre_times) != len(centre_lambdas):
        raise ValueError("Number of time centroids and lambda centroids do not match")
    

    tom_map = sc.DataArray(
        data=sc.array(dims=['tof'], values=centre_times, unit='s'),
        coords={'tof': sc.array(dims=['tof'], values=centre_times, unit='s')}
    )

    lambda_map = sc.DataArray(
        data=sc.array(dims=['time_on_monitor'], values=centre_lambdas, unit='angstrom'),
        coords={'time_on_monitor': sc.array(dims=['time_on_monitor'], values=centre_times, unit='s')}
    )

    
    return tom_map, lambda_map


def produce_trex_event_object(event_object, time_lam_monitor, threshold, n_bins):
    """ 
    time_lam_monitor should be a 2d time lambda monitor at the sample

    """

    # with mx.Read(data_path) as loaded_data:
    #     monitor_position = loaded_data.get_global_component_coordinates(time_lam_monitor)

    tom_map, lambda_map = create_time_lamda_mappings(time_lam_monitor, threshold=threshold, n_bins=n_bins)

    tof_to_centroid = sc.lookup(tom_map, mode='previous')
    tom_to_lambda = sc.lookup(lambda_map, mode='nearest')

    event_object = event_object.transform_coords(time_on_monitor=tof_to_centroid)
    event_object = event_object.transform_coords(lambda_i=tom_to_lambda)

    return event_object


# Graph Functions

def straight_monitor_beam(
    source_position: VariableLike, monitor_position: VariableLike
):
    """ """
    return _canonical_length(monitor_position) - _canonical_length(source_position)


def Lm(monitor_beam):
    """ """
    return _canonical_length(sc.norm(monitor_beam))


# -------- ki ---------


def unit_ki_from_source_and_sample(
    source_position: VariableLike, sample_position: VariableLike
):
    """ """
    d = sample_position - source_position
    unit_ki = d / sc.norm(d)
    return unit_ki


def vi_from_monitor(Lm, time_on_monitor):
    """ """
    vi = Lm / time_on_monitor
    return vi


def mag_ki_from_vi(vi):
    """ """
    mag_ki = (sc.constants.neutron_mass * vi) / sc.constants.hbar
    return sc.to_unit(mag_ki, "1/angstrom")


def mag_ki_from_lambda(lambda_i):
    """ """
    mag_ki = (2 * sc.constants.pi) / lambda_i
    return sc.to_unit(mag_ki, "1/angstrom")


def ki(mag_ki, unit_ki):
    """ """
    return mag_ki * unit_ki


# -------- kf ---------


def unit_kf_from_detector_and_sample(
    position: VariableLike, sample_position: VariableLike
):
    """ """
    d = position - sample_position
    unit_kf = d / sc.norm(d)

    return unit_kf


def time_on_sample_from_velocity(L1, vi):
    """ """
    time_on_sample = L1 / vi
    return time_on_sample


def time_on_sample_from_monitor(time_on_monitor):
    """ """
    return time_on_monitor


def vf_from_tof(L2, tof, time_on_sample):
    """ """
    vf = L2 / (tof - time_on_sample)
    return vf


def mag_kf_from_vf(vf):
    """ """
    mag_kf = (sc.constants.neutron_mass * vf) / sc.constants.hbar
    return sc.to_unit(mag_kf, "1/angstrom")


def kf(mag_kf, unit_kf):
    """ """
    return mag_kf * unit_kf


# --------- Q and dE --------


def Q_from_k(ki, kf):
    """ """
    return ki - kf


def mag_Q(Q):
    """ """
    return sc.to_unit(sc.norm(Q), "1/angstrom")


def energy_transfer_from_k(mag_kf, mag_ki):
    """ """
    dE = (sc.constants.hbar**2 / (2 * sc.constants.neutron_mass)) * (
        mag_ki**2 - mag_kf**2
    )
    return sc.to_unit(dE, sc.units.meV)


inelastic = {}

inelastic["Lm"] = Lm
inelastic["monitor_beam"] = straight_monitor_beam

inelastic["unit_ki"] = unit_ki_from_source_and_sample
inelastic["vi"] = vi_from_monitor
#inelastic["mag_ki"] = mag_ki_from_vi
inelastic["mag_ki"] = mag_ki_from_lambda
inelastic["ki"] = ki

inelastic["unit_kf"] = unit_kf_from_detector_and_sample
inelastic["vf"] = vf_from_tof
inelastic["mag_kf"] = mag_kf_from_vf
inelastic["kf"] = kf

inelastic["Q"] = Q_from_k
inelastic["mag_Q"] = mag_Q
inelastic["dE"] = energy_transfer_from_k
inelastic["time_on_sample"] = time_on_sample_from_monitor
