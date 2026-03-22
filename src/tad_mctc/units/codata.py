# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Units: CODATA
=============

CODATA values for various physical constants from qcelemental.
"""

from __future__ import annotations

__all__ = ["CODATA", "get_constant"]


class _CodataContext:
    """Explicitly typed CODATA constants."""

    # Created with the following script:
    #
    # import qcelemental as qcel
    #
    # if __name__ == "__main__":
    #     print("class _CodataContext:")
    #     print('    """Explicitly typed CODATA constants."""')
    #
    #     _transtable = str.maketrans(" -/{", "__p_", ".,()")
    #
    #     # 1. Generate explicit class attributes
    #     for name, datum in qcel.constants.pc.items():
    #         attr_name = name.translate(_transtable)
    #         print(f"    {attr_name}: float = {float(datum.data)}")

    alpha_particle_electron_mass_ratio: float = 7294.29954136
    alpha_particle_mass: float = 6.64465723e-27
    alpha_particle_mass_energy_equivalent: float = 5.971920097e-10
    alpha_particle_mass_energy_equivalent_in_mev: float = 3727.379378
    alpha_particle_mass_in_u: float = 4.001506179127
    alpha_particle_molar_mass: float = 0.004001506179127
    alpha_particle_proton_mass_ratio: float = 3.97259968907
    angstrom_star: float = 1.00001495e-10
    atomic_mass_constant: float = 1.66053904e-27
    atomic_mass_constant_energy_equivalent: float = 1.492418062e-10
    atomic_mass_constant_energy_equivalent_in_mev: float = 931.4940954
    atomic_mass_unit_electron_volt_relationship: float = 931494095.4
    atomic_mass_unit_hartree_relationship: float = 34231776.902
    atomic_mass_unit_hertz_relationship: float = 2.2523427206e23
    atomic_mass_unit_inverse_meter_relationship: float = 751300661660000.0
    atomic_mass_unit_joule_relationship: float = 1.492418062e-10
    atomic_mass_unit_kelvin_relationship: float = 10809543800000.0
    atomic_mass_unit_kilogram_relationship: float = 1.66053904e-27
    atomic_unit_of_1st_hyperpolarizability: float = 3.206361329e-53
    atomic_unit_of_2nd_hyperpolarizability: float = 6.235380085e-65
    atomic_unit_of_action: float = 1.0545718e-34
    atomic_unit_of_charge: float = 1.6021766208e-19
    atomic_unit_of_charge_density: float = 1081202377000.0
    atomic_unit_of_current: float = 0.006623618183
    atomic_unit_of_electric_dipole_mom: float = 8.478353552e-30
    atomic_unit_of_electric_field: float = 514220670700.0
    atomic_unit_of_electric_field_gradient: float = 9.717362356e21
    atomic_unit_of_electric_polarizability: float = 1.6487772731e-41
    atomic_unit_of_electric_potential: float = 27.21138602
    atomic_unit_of_electric_quadrupole_mom: float = 4.486551484e-40
    atomic_unit_of_energy: float = 4.35974465e-18
    atomic_unit_of_force: float = 8.23872336e-08
    atomic_unit_of_length: float = 5.2917721067e-11
    atomic_unit_of_mag_dipole_mom: float = 1.854801999e-23
    atomic_unit_of_mag_flux_density: float = 235051.755
    atomic_unit_of_magnetizability: float = 7.8910365886e-29
    atomic_unit_of_mass: float = 9.10938356e-31
    atomic_unit_of_momum: float = 1.992851882e-24
    atomic_unit_of_permittivity: float = 1.112650056e-10
    atomic_unit_of_time: float = 2.418884326509e-17
    atomic_unit_of_velocity: float = 2187691.26277
    avogadro_constant: float = 6.022140857e23
    bohr_magneton: float = 9.274009994e-24
    bohr_magneton_in_evpt: float = 5.7883818012e-05
    bohr_magneton_in_hzpt: float = 13996245042.0
    bohr_magneton_in_inverse_meters_per_tesla: float = 46.68644814
    bohr_magneton_in_kpt: float = 0.67171405
    bohr_radius: float = 5.2917721067e-11
    boltzmann_constant: float = 1.38064852e-23
    boltzmann_constant_in_evpk: float = 8.6173303e-05
    boltzmann_constant_in_hzpk: float = 20836612000.0
    boltzmann_constant_in_inverse_meters_per_kelvin: float = 69.503457
    characteristic_impedance_of_vacuum: float = 376.730313461
    classical_electron_radius: float = 2.8179403227e-15
    compton_wavelength: float = 2.4263102367e-12
    compton_wavelength_over_2_pi: float = 3.8615926764e-13
    conductance_quantum: float = 7.748091731e-05
    conventional_value_of_josephson_constant: float = 483597900000000.0
    conventional_value_of_von_klitzing_constant: float = 25812.807
    cu_x_unit: float = 1.00207697e-13
    deuteron_electron_mag_mom_ratio: float = -0.0004664345535
    deuteron_electron_mass_ratio: float = 3670.48296785
    deuteron_g_factor: float = 0.8574382311
    deuteron_mag_mom: float = 4.33073504e-27
    deuteron_mag_mom_to_bohr_magneton_ratio: float = 0.0004669754554
    deuteron_mag_mom_to_nuclear_magneton_ratio: float = 0.8574382311
    deuteron_mass: float = 3.343583719e-27
    deuteron_mass_energy_equivalent: float = 3.005063183e-10
    deuteron_mass_energy_equivalent_in_mev: float = 1875.612928
    deuteron_mass_in_u: float = 2.013553212745
    deuteron_molar_mass: float = 0.002013553212745
    deuteron_neutron_mag_mom_ratio: float = -0.44820652
    deuteron_proton_mag_mom_ratio: float = 0.3070122077
    deuteron_proton_mass_ratio: float = 1.99900750087
    deuteron_rms_charge_radius: float = 2.1413e-15
    electric_constant: float = 8.854187817e-12
    electron_charge_to_mass_quotient: float = -175882002400.0
    electron_deuteron_mag_mom_ratio: float = -2143.923499
    electron_deuteron_mass_ratio: float = 0.0002724437107484
    electron_g_factor: float = -2.00231930436182
    electron_gyromag_ratio: float = 176085964400.0
    electron_gyromag_ratio_over_2_pi: float = 28024.95164
    electron_helion_mass_ratio: float = 0.0001819543074854
    electron_mag_mom: float = -9.28476462e-24
    electron_mag_mom_anomaly: float = 0.00115965218091
    electron_mag_mom_to_bohr_magneton_ratio: float = -1.00115965218091
    electron_mag_mom_to_nuclear_magneton_ratio: float = -1838.28197234
    electron_mass: float = 9.10938356e-31
    electron_mass_energy_equivalent: float = 8.18710565e-14
    electron_mass_energy_equivalent_in_mev: float = 0.5109989461
    electron_mass_in_u: float = 0.00054857990907
    electron_molar_mass: float = 5.4857990907e-07
    electron_muon_mag_mom_ratio: float = 206.766988
    electron_muon_mass_ratio: float = 0.0048363317
    electron_neutron_mag_mom_ratio: float = 960.9205
    electron_neutron_mass_ratio: float = 0.00054386734428
    electron_proton_mag_mom_ratio: float = -658.2106866
    electron_proton_mass_ratio: float = 0.000544617021352
    electron_tau_mass_ratio: float = 0.000287592
    electron_to_alpha_particle_mass_ratio: float = 0.0001370933554798
    electron_to_shielded_helion_mag_mom_ratio: float = 864.058257
    electron_to_shielded_proton_mag_mom_ratio: float = -658.2275971
    electron_triton_mass_ratio: float = 0.0001819200062203
    electron_volt: float = 1.6021766208e-19
    electron_volt_atomic_mass_unit_relationship: float = 1.0735441105e-09
    electron_volt_hartree_relationship: float = 0.03674932248
    electron_volt_hertz_relationship: float = 241798926200000.0
    electron_volt_inverse_meter_relationship: float = 806554.4005
    electron_volt_joule_relationship: float = 1.6021766208e-19
    electron_volt_kelvin_relationship: float = 11604.5221
    electron_volt_kilogram_relationship: float = 1.782661907e-36
    elementary_charge: float = 1.6021766208e-19
    elementary_charge_over_h: float = 241798926200000.0
    faraday_constant: float = 96485.33289
    faraday_constant_for_conventional_electric_current: float = 96485.3251
    fermi_coupling_constant: float = 1.1663787e-05
    fine_structure_constant: float = 0.0072973525664
    first_radiation_constant: float = 3.74177179e-16
    first_radiation_constant_for_spectral_radiance: float = 1.191042953e-16
    hartree_atomic_mass_unit_relationship: float = 2.9212623197e-08
    hartree_electron_volt_relationship: float = 27.21138602
    hartree_energy: float = 4.35974465e-18
    hartree_energy_in_ev: float = 27.21138602
    hartree_hertz_relationship: float = 6579683920711000.0
    hartree_inverse_meter_relationship: float = 21947463.13702
    hartree_joule_relationship: float = 4.35974465e-18
    hartree_kelvin_relationship: float = 315775.13
    hartree_kilogram_relationship: float = 4.850870129e-35
    helion_electron_mass_ratio: float = 5495.88527922
    helion_g_factor: float = -4.255250616
    helion_mag_mom: float = -1.074617522e-26
    helion_mag_mom_to_bohr_magneton_ratio: float = -0.001158740958
    helion_mag_mom_to_nuclear_magneton_ratio: float = -2.127625308
    helion_mass: float = 5.0064127e-27
    helion_mass_energy_equivalent: float = 4.499539341e-10
    helion_mass_energy_equivalent_in_mev: float = 2808.391586
    helion_mass_in_u: float = 3.01493224673
    helion_molar_mass: float = 0.00301493224673
    helion_proton_mass_ratio: float = 2.99315267046
    hertz_atomic_mass_unit_relationship: float = 4.4398216616e-24
    hertz_electron_volt_relationship: float = 4.135667662e-15
    hertz_hartree_relationship: float = 1.5198298460088e-16
    hertz_inverse_meter_relationship: float = 3.335640951e-09
    hertz_joule_relationship: float = 6.62607004e-34
    hertz_kelvin_relationship: float = 4.7992447e-11
    hertz_kilogram_relationship: float = 7.372497201e-51
    inverse_fine_structure_constant: float = 137.035999139
    inverse_meter_atomic_mass_unit_relationship: float = 1.331025049e-15
    inverse_meter_electron_volt_relationship: float = 1.2398419739e-06
    inverse_meter_hartree_relationship: float = 4.556335252767e-08
    inverse_meter_hertz_relationship: float = 299792458.0
    inverse_meter_joule_relationship: float = 1.986445824e-25
    inverse_meter_kelvin_relationship: float = 0.0143877736
    inverse_meter_kilogram_relationship: float = 2.210219057e-42
    inverse_of_conductance_quantum: float = 12906.4037278
    josephson_constant: float = 483597852500000.0
    joule_atomic_mass_unit_relationship: float = 6700535363.0
    joule_electron_volt_relationship: float = 6.241509126e18
    joule_hartree_relationship: float = 2.293712317e17
    joule_hertz_relationship: float = 1.509190205e33
    joule_inverse_meter_relationship: float = 5.034116651e24
    joule_kelvin_relationship: float = 7.2429731e22
    joule_kilogram_relationship: float = 1.112650056e-17
    kelvin_atomic_mass_unit_relationship: float = 9.2510842e-14
    kelvin_electron_volt_relationship: float = 8.6173303e-05
    kelvin_hartree_relationship: float = 3.1668105e-06
    kelvin_hertz_relationship: float = 20836612000.0
    kelvin_inverse_meter_relationship: float = 69.503457
    kelvin_joule_relationship: float = 1.38064852e-23
    kelvin_kilogram_relationship: float = 1.53617865e-40
    kilogram_atomic_mass_unit_relationship: float = 6.022140857e26
    kilogram_electron_volt_relationship: float = 5.60958865e35
    kilogram_hartree_relationship: float = 2.061485823e34
    kilogram_hertz_relationship: float = 1.356392512e50
    kilogram_inverse_meter_relationship: float = 4.524438411e41
    kilogram_joule_relationship: float = 8.987551787e16
    kilogram_kelvin_relationship: float = 6.5096595e39
    lattice_parameter_of_silicon: float = 5.431020504e-10
    loschmidt_constant_27315_k_100_kpa: float = 2.6516467e25
    loschmidt_constant_27315_k_101325_kpa: float = 2.6867811e25
    mag_constant: float = 1.2566370614e-06
    mag_flux_quantum: float = 2.067833831e-15
    molar_gas_constant: float = 8.3144598
    molar_mass_constant: float = 0.001
    molar_mass_of_carbon_12: float = 0.012
    molar_planck_constant: float = 3.990312711e-10
    molar_planck_constant_times_c: float = 0.119626565582
    molar_volume_of_ideal_gas_27315_k_100_kpa: float = 0.022710947
    molar_volume_of_ideal_gas_27315_k_101325_kpa: float = 0.022413962
    molar_volume_of_silicon: float = 1.205883214e-05
    mo_x_unit: float = 1.00209952e-13
    muon_compton_wavelength: float = 1.173444111e-14
    muon_compton_wavelength_over_2_pi: float = 1.867594308e-15
    muon_electron_mass_ratio: float = 206.7682826
    muon_g_factor: float = -2.0023318418
    muon_mag_mom: float = -4.49044826e-26
    muon_mag_mom_anomaly: float = 0.00116592089
    muon_mag_mom_to_bohr_magneton_ratio: float = -0.00484197048
    muon_mag_mom_to_nuclear_magneton_ratio: float = -8.89059705
    muon_mass: float = 1.883531594e-28
    muon_mass_energy_equivalent: float = 1.692833774e-11
    muon_mass_energy_equivalent_in_mev: float = 105.6583745
    muon_mass_in_u: float = 0.1134289257
    muon_molar_mass: float = 0.0001134289257
    muon_neutron_mass_ratio: float = 0.1124545167
    muon_proton_mag_mom_ratio: float = -3.183345142
    muon_proton_mass_ratio: float = 0.1126095262
    muon_tau_mass_ratio: float = 0.0594649
    natural_unit_of_action: float = 1.0545718e-34
    natural_unit_of_action_in_ev_s: float = 6.582119514e-16
    natural_unit_of_energy: float = 8.18710565e-14
    natural_unit_of_energy_in_mev: float = 0.5109989461
    natural_unit_of_length: float = 3.8615926764e-13
    natural_unit_of_mass: float = 9.10938356e-31
    natural_unit_of_momum: float = 2.730924488e-22
    natural_unit_of_momum_in_mevpc: float = 0.5109989461
    natural_unit_of_time: float = 1.28808866712e-21
    natural_unit_of_velocity: float = 299792458.0
    neutron_compton_wavelength: float = 1.31959090481e-15
    neutron_compton_wavelength_over_2_pi: float = 2.1001941536e-16
    neutron_electron_mag_mom_ratio: float = 0.00104066882
    neutron_electron_mass_ratio: float = 1838.68366158
    neutron_g_factor: float = -3.82608545
    neutron_gyromag_ratio: float = 183247172.0
    neutron_gyromag_ratio_over_2_pi: float = 29.1646933
    neutron_mag_mom: float = -9.662365e-27
    neutron_mag_mom_to_bohr_magneton_ratio: float = -0.00104187563
    neutron_mag_mom_to_nuclear_magneton_ratio: float = -1.91304273
    neutron_mass: float = 1.674927471e-27
    neutron_mass_energy_equivalent: float = 1.505349739e-10
    neutron_mass_energy_equivalent_in_mev: float = 939.5654133
    neutron_mass_in_u: float = 1.00866491588
    neutron_molar_mass: float = 0.00100866491588
    neutron_muon_mass_ratio: float = 8.89248408
    neutron_proton_mag_mom_ratio: float = -0.68497934
    neutron_proton_mass_difference: float = 2.30557377e-30
    neutron_proton_mass_difference_energy_equivalent: float = 2.07214637e-13
    neutron_proton_mass_difference_energy_equivalent_in_mev: float = 1.29333205
    neutron_proton_mass_difference_in_u: float = 0.001388449
    neutron_proton_mass_ratio: float = 1.00137841898
    neutron_tau_mass_ratio: float = 0.52879
    neutron_to_shielded_proton_mag_mom_ratio: float = -0.68499694
    newtonian_constant_of_gravitation: float = 6.67408e-11
    newtonian_constant_of_gravitation_over_h_bar_c: float = 6.70861e-39
    nuclear_magneton: float = 5.050783699e-27
    nuclear_magneton_in_evpt: float = 3.152451255e-08
    nuclear_magneton_in_inverse_meters_per_tesla: float = 0.02542623432
    nuclear_magneton_in_kpt: float = 0.0003658269
    nuclear_magneton_in_mhzpt: float = 7.622593285
    planck_constant: float = 6.62607004e-34
    planck_constant_in_ev_s: float = 4.135667662e-15
    planck_constant_over_2_pi: float = 1.0545718e-34
    planck_constant_over_2_pi_in_ev_s: float = 6.582119514e-16
    planck_constant_over_2_pi_times_c_in_mev_fm: float = 197.3269788
    planck_length: float = 1.616229e-35
    planck_mass: float = 2.17647e-08
    planck_mass_energy_equivalent_in_gev: float = 1.22091e19
    planck_temperature: float = 1.416808e32
    planck_time: float = 5.39116e-44
    proton_charge_to_mass_quotient: float = 95788332.26
    proton_compton_wavelength: float = 1.32140985396e-15
    proton_compton_wavelength_over_2_pi: float = 2.10308910109e-16
    proton_electron_mass_ratio: float = 1836.15267389
    proton_g_factor: float = 5.585694702
    proton_gyromag_ratio: float = 267522190.0
    proton_gyromag_ratio_over_2_pi: float = 42.57747892
    proton_mag_mom: float = 1.4106067873e-26
    proton_mag_mom_to_bohr_magneton_ratio: float = 0.0015210322053
    proton_mag_mom_to_nuclear_magneton_ratio: float = 2.7928473508
    proton_mag_shielding_correction: float = 2.5691e-05
    proton_mass: float = 1.672621898e-27
    proton_mass_energy_equivalent: float = 1.503277593e-10
    proton_mass_energy_equivalent_in_mev: float = 938.2720813
    proton_mass_in_u: float = 1.007276466879
    proton_molar_mass: float = 0.001007276466879
    proton_muon_mass_ratio: float = 8.88024338
    proton_neutron_mag_mom_ratio: float = -1.45989805
    proton_neutron_mass_ratio: float = 0.99862347844
    proton_rms_charge_radius: float = 8.751e-16
    proton_tau_mass_ratio: float = 0.528063
    quantum_of_circulation: float = 0.00036369475486
    quantum_of_circulation_times_2: float = 0.00072738950972
    rydberg_constant: float = 10973731.568508
    rydberg_constant_times_c_in_hz: float = 3289841960355000.0
    rydberg_constant_times_hc_in_ev: float = 13.605693009
    rydberg_constant_times_hc_in_j: float = 2.179872325e-18
    sackur_tetrode_constant_1_k_100_kpa: float = -1.1517084
    sackur_tetrode_constant_1_k_101325_kpa: float = -1.1648714
    second_radiation_constant: float = 0.0143877736
    shielded_helion_gyromag_ratio: float = 203789458.5
    shielded_helion_gyromag_ratio_over_2_pi: float = 32.43409966
    shielded_helion_mag_mom: float = -1.07455308e-26
    shielded_helion_mag_mom_to_bohr_magneton_ratio: float = -0.001158671471
    shielded_helion_mag_mom_to_nuclear_magneton_ratio: float = -2.12749772
    shielded_helion_to_proton_mag_mom_ratio: float = -0.7617665603
    shielded_helion_to_shielded_proton_mag_mom_ratio: float = -0.7617861313
    shielded_proton_gyromag_ratio: float = 267515317.1
    shielded_proton_gyromag_ratio_over_2_pi: float = 42.57638507
    shielded_proton_mag_mom: float = 1.410570547e-26
    shielded_proton_mag_mom_to_bohr_magneton_ratio: float = 0.001520993128
    shielded_proton_mag_mom_to_nuclear_magneton_ratio: float = 2.7927756
    speed_of_light_in_vacuum: float = 299792458.0
    standard_acceleration_of_gravity: float = 9.80665
    standard_atmosphere: float = 101325.0
    standard_state_pressure: float = 100000.0
    stefan_boltzmann_constant: float = 5.670367e-08
    tau_compton_wavelength: float = 6.97787e-16
    tau_compton_wavelength_over_2_pi: float = 1.11056e-16
    tau_electron_mass_ratio: float = 3477.15
    tau_mass: float = 3.16747e-27
    tau_mass_energy_equivalent: float = 2.84678e-10
    tau_mass_energy_equivalent_in_mev: float = 1776.82
    tau_mass_in_u: float = 1.90749
    tau_molar_mass: float = 0.00190749
    tau_muon_mass_ratio: float = 16.8167
    tau_neutron_mass_ratio: float = 1.89111
    tau_proton_mass_ratio: float = 1.89372
    thomson_cross_section: float = 6.6524587158e-29
    triton_electron_mass_ratio: float = 5496.92153588
    triton_g_factor: float = 5.95792492
    triton_mag_mom: float = 1.504609503e-26
    triton_mag_mom_to_bohr_magneton_ratio: float = 0.0016223936616
    triton_mag_mom_to_nuclear_magneton_ratio: float = 2.97896246
    triton_mass: float = 5.007356665e-27
    triton_mass_energy_equivalent: float = 4.500387735e-10
    triton_mass_energy_equivalent_in_mev: float = 2808.921112
    triton_mass_in_u: float = 3.01550071632
    triton_molar_mass: float = 0.00301550071632
    triton_proton_mass_ratio: float = 2.99371703348
    unified_atomic_mass_unit: float = 1.66053904e-27
    von_klitzing_constant: float = 25812.8074555
    weak_mixing_angle: float = 0.2223
    wien_frequency_displacement_law_constant: float = 58789238000.0
    wien_wavelength_displacement_law_constant: float = 0.0028977729
    calorie_joule_relationship: float = 4.184
    h: float = 6.62607004e-34
    hbar: float = 1.0545718e-34
    c: float = 299792458.0
    kb: float = 1.38064852e-23
    r: float = 8.3144598
    bohr2angstroms: float = 0.52917721067
    bohr2m: float = 5.2917721067e-11
    bohr2cm: float = 5.2917721067e-09
    amu2g: float = 1.66053904e-24
    amu2kg: float = 1.66053904e-27
    au2amu: float = 0.00054857990907
    hartree2j: float = 4.35974465e-18
    hartree2aj: float = 4.35974465
    cal2j: float = 4.184
    dipmom_au2si: float = 8.478353552e-30
    dipmom_au2debye: float = 2.541746451895026
    dipmom_debye2si: float = 3.335640951e-30
    c_au: float = 137.035999139
    hartree2ev: float = 27.21138602
    hartree2wavenumbers: float = 219474.6313702
    hartree2kcalmol: float = 627.5094737775374
    hartree2kjmol: float = 2625.4996382852164
    hartree2mhz: float = 6579683920.711
    na: float = 6.022140857e23
    me: float = 9.10938356e-31
    kcalmol2wavenumbers: float = 349.7550882318032
    e0: float = 8.854187817e-12


CODATA = _CodataContext()
"""CODATA 2018 values from qcelemental 0.29.1."""


def get_constant(name: str) -> float:
    """
    Get a constant from the CODATA 2018 context.

    Parameters
    ----------
    name : str
        Name of the constant.

    Returns
    -------
    float
        Value of the constant.

    Raises
    ------
    KeyError
        If the constant is not found.
    """
    _transtable = str.maketrans(" -/{", "__p_", ".,()")
    attr_name = name.translate(_transtable).casefold()

    if not hasattr(CODATA, attr_name):
        raise KeyError(f"Constant '{name}' not found.")

    return getattr(CODATA, attr_name)
