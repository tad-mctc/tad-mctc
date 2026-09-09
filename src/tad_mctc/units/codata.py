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

    alpha_particle_electron_mass_ratio: float = 7294.29954142
    alpha_particle_mass: float = 6.6446573357e-27
    alpha_particle_mass_energy_equivalent: float = 5.9719201914e-10
    alpha_particle_mass_energy_equivalent_in_mev: float = 3727.3794066
    alpha_particle_mass_in_u: float = 4.001506179127
    alpha_particle_molar_mass: float = 0.0040015061777
    alpha_particle_proton_mass_ratio: float = 3.97259969009
    alpha_particle_relative_atomic_mass: float = 4.001506179127
    angstrom_star: float = 1.00001495e-10
    atomic_mass_constant: float = 1.6605390666e-27
    atomic_mass_constant_energy_equivalent: float = 1.4924180856e-10
    atomic_mass_constant_energy_equivalent_in_mev: float = 931.49410242
    atomic_mass_unit_electron_volt_relationship: float = 931494102.42
    atomic_mass_unit_hartree_relationship: float = 34231776.874
    atomic_mass_unit_hertz_relationship: float = 2.25234271871e23
    atomic_mass_unit_inverse_meter_relationship: float = 751300661040000.0
    atomic_mass_unit_joule_relationship: float = 1.4924180856e-10
    atomic_mass_unit_kelvin_relationship: float = 10809540191600.0
    atomic_mass_unit_kilogram_relationship: float = 1.6605390666e-27
    atomic_unit_of_1st_hyperpolarizability: float = 3.2063613061e-53
    atomic_unit_of_2nd_hyperpolarizability: float = 6.2353799905e-65
    atomic_unit_of_action: float = 1.054571817e-34
    atomic_unit_of_charge: float = 1.602176634e-19
    atomic_unit_of_charge_density: float = 1081202384570.0
    atomic_unit_of_current: float = 0.00662361823751
    atomic_unit_of_electric_dipole_mom: float = 8.4783536255e-30
    atomic_unit_of_electric_field: float = 514220674763.0
    atomic_unit_of_electric_field_gradient: float = 9.7173624292e21
    atomic_unit_of_electric_polarizability: float = 1.64877727436e-41
    atomic_unit_of_electric_potential: float = 27.211386245988
    atomic_unit_of_electric_quadrupole_mom: float = 4.4865515246e-40
    atomic_unit_of_energy: float = 4.3597447222071e-18
    atomic_unit_of_force: float = 8.2387234983e-08
    atomic_unit_of_length: float = 5.29177210903e-11
    atomic_unit_of_mag_dipole_mom: float = 1.85480201566e-23
    atomic_unit_of_mag_flux_density: float = 235051.756758
    atomic_unit_of_magnetizability: float = 7.8910366008e-29
    atomic_unit_of_mass: float = 9.1093837015e-31
    atomic_unit_of_momentum: float = 1.9928519141e-24
    atomic_unit_of_permittivity: float = 1.11265005545e-10
    atomic_unit_of_time: float = 2.4188843265857e-17
    atomic_unit_of_velocity: float = 2187691.26364
    avogadro_constant: float = 6.02214076e23
    bohr_magneton: float = 9.2740100783e-24
    bohr_magneton_in_evpt: float = 5.788381806e-05
    bohr_magneton_in_hzpt: float = 13996244936.1
    bohr_magneton_in_inverse_meter_per_tesla: float = 46.686447783
    bohr_magneton_in_kpt: float = 0.67171381563
    bohr_radius: float = 5.29177210903e-11
    boltzmann_constant: float = 1.380649e-23
    boltzmann_constant_in_evpk: float = 8.617333262e-05
    boltzmann_constant_in_hzpk: float = 20836619120.0
    boltzmann_constant_in_inverse_meter_per_kelvin: float = 69.50348004
    characteristic_impedance_of_vacuum: float = 376.730313668
    classical_electron_radius: float = 2.8179403262e-15
    compton_wavelength: float = 2.42631023867e-12
    conductance_quantum: float = 7.748091729e-05
    conventional_value_of_ampere_90: float = 1.00000008887
    conventional_value_of_coulomb_90: float = 1.00000008887
    conventional_value_of_farad_90: float = 0.9999999822
    conventional_value_of_henry_90: float = 1.00000001779
    conventional_value_of_josephson_constant: float = 483597900000000.0
    conventional_value_of_ohm_90: float = 1.00000001779
    conventional_value_of_volt_90: float = 1.00000010666
    conventional_value_of_von_klitzing_constant: float = 25812.807
    conventional_value_of_watt_90: float = 1.00000019553
    copper_x_unit: float = 1.00207697e-13
    deuteron_electron_mag_mom_ratio: float = -0.0004664345551
    deuteron_electron_mass_ratio: float = 3670.48296788
    deuteron_g_factor: float = 0.8574382338
    deuteron_mag_mom: float = 4.330735094e-27
    deuteron_mag_mom_to_bohr_magneton_ratio: float = 0.000466975457
    deuteron_mag_mom_to_nuclear_magneton_ratio: float = 0.8574382338
    deuteron_mass: float = 3.3435837724e-27
    deuteron_mass_energy_equivalent: float = 3.00506323102e-10
    deuteron_mass_energy_equivalent_in_mev: float = 1875.61294257
    deuteron_mass_in_u: float = 2.013553212745
    deuteron_molar_mass: float = 0.00201355321205
    deuteron_neutron_mag_mom_ratio: float = -0.44820653
    deuteron_proton_mag_mom_ratio: float = 0.30701220939
    deuteron_proton_mass_ratio: float = 1.99900750139
    deuteron_relative_atomic_mass: float = 2.013553212745
    deuteron_rms_charge_radius: float = 2.12799e-15
    electron_charge_to_mass_quotient: float = -175882001076.0
    electron_deuteron_mag_mom_ratio: float = -2143.9234915
    electron_deuteron_mass_ratio: float = 0.0002724437107462
    electron_g_factor: float = -2.00231930436256
    electron_gyromag_ratio: float = 176085963023.0
    electron_gyromag_ratio_in_mhzpt: float = 28024.9514242
    electron_helion_mass_ratio: float = 0.0001819543074573
    electron_mag_mom: float = -9.2847647043e-24
    electron_mag_mom_anomaly: float = 0.00115965218128
    electron_mag_mom_to_bohr_magneton_ratio: float = -1.00115965218128
    electron_mag_mom_to_nuclear_magneton_ratio: float = -1838.28197188
    electron_mass: float = 9.1093837015e-31
    electron_mass_energy_equivalent: float = 8.1871057769e-14
    electron_mass_energy_equivalent_in_mev: float = 0.51099895
    electron_mass_in_u: float = 0.000548579909065
    electron_molar_mass: float = 5.4857990888e-07
    electron_muon_mag_mom_ratio: float = 206.7669883
    electron_muon_mass_ratio: float = 0.00483633169
    electron_neutron_mag_mom_ratio: float = 960.9205
    electron_neutron_mass_ratio: float = 0.00054386734424
    electron_proton_mag_mom_ratio: float = -658.21068789
    electron_proton_mass_ratio: float = 0.000544617021487
    electron_relative_atomic_mass: float = 0.000548579909065
    electron_tau_mass_ratio: float = 0.000287585
    electron_to_alpha_particle_mass_ratio: float = 0.0001370933554787
    electron_to_shielded_helion_mag_mom_ratio: float = 864.058257
    electron_to_shielded_proton_mag_mom_ratio: float = -658.2275971
    electron_triton_mass_ratio: float = 0.0001819200062251
    electron_volt: float = 1.602176634e-19
    electron_volt_atomic_mass_unit_relationship: float = 1.07354410233e-09
    electron_volt_hartree_relationship: float = 0.036749322175655
    electron_volt_hertz_relationship: float = 241798924200000.0
    electron_volt_inverse_meter_relationship: float = 806554.3937
    electron_volt_joule_relationship: float = 1.602176634e-19
    electron_volt_kelvin_relationship: float = 11604.51812
    electron_volt_kilogram_relationship: float = 1.782661921e-36
    elementary_charge: float = 1.602176634e-19
    elementary_charge_over_h_bar: float = 1519267447000000.0
    faraday_constant: float = 96485.33212
    fermi_coupling_constant: float = 1.1663787e-05
    fine_structure_constant: float = 0.0072973525693
    first_radiation_constant: float = 3.741771852e-16
    first_radiation_constant_for_spectral_radiance: float = 1.191042972e-16
    hartree_atomic_mass_unit_relationship: float = 2.92126232205e-08
    hartree_electron_volt_relationship: float = 27.211386245988
    hartree_energy: float = 4.3597447222071e-18
    hartree_energy_in_ev: float = 27.211386245988
    hartree_hertz_relationship: float = 6579683920502000.0
    hartree_inverse_meter_relationship: float = 21947463.13632
    hartree_joule_relationship: float = 4.3597447222071e-18
    hartree_kelvin_relationship: float = 315775.02480407
    hartree_kilogram_relationship: float = 4.8508702095432e-35
    helion_electron_mass_ratio: float = 5495.88528007
    helion_g_factor: float = -4.255250615
    helion_mag_mom: float = -1.074617532e-26
    helion_mag_mom_to_bohr_magneton_ratio: float = -0.001158740958
    helion_mag_mom_to_nuclear_magneton_ratio: float = -2.127625307
    helion_mass: float = 5.0064127796e-27
    helion_mass_energy_equivalent: float = 4.4995394125e-10
    helion_mass_energy_equivalent_in_mev: float = 2808.39160743
    helion_mass_in_u: float = 3.014932247175
    helion_molar_mass: float = 0.00301493224613
    helion_proton_mass_ratio: float = 2.99315267167
    helion_relative_atomic_mass: float = 3.014932247175
    helion_shielding_shift: float = 5.996743e-05
    hertz_atomic_mass_unit_relationship: float = 4.4398216652e-24
    hertz_electron_volt_relationship: float = 4.135667696e-15
    hertz_hartree_relationship: float = 1.519829846057e-16
    hertz_inverse_meter_relationship: float = 3.335640951e-09
    hertz_joule_relationship: float = 6.62607015e-34
    hertz_kelvin_relationship: float = 4.799243073e-11
    hertz_kilogram_relationship: float = 7.372497323e-51
    hyperfine_transition_frequency_of_cs_133: float = 9192631770.0
    inverse_fine_structure_constant: float = 137.035999084
    inverse_meter_atomic_mass_unit_relationship: float = 1.3310250501e-15
    inverse_meter_electron_volt_relationship: float = 1.239841984e-06
    inverse_meter_hartree_relationship: float = 4.556335252912e-08
    inverse_meter_hertz_relationship: float = 299792458.0
    inverse_meter_joule_relationship: float = 1.986445857e-25
    inverse_meter_kelvin_relationship: float = 0.01438776877
    inverse_meter_kilogram_relationship: float = 2.210219094e-42
    inverse_of_conductance_quantum: float = 12906.40372
    josephson_constant: float = 483597848400000.0
    joule_atomic_mass_unit_relationship: float = 6700535256.5
    joule_electron_volt_relationship: float = 6.241509074e18
    joule_hartree_relationship: float = 2.2937122783963e17
    joule_hertz_relationship: float = 1.509190179e33
    joule_inverse_meter_relationship: float = 5.034116567e24
    joule_kelvin_relationship: float = 7.242970516e22
    joule_kilogram_relationship: float = 1.112650056e-17
    kelvin_atomic_mass_unit_relationship: float = 9.2510873014e-14
    kelvin_electron_volt_relationship: float = 8.617333262e-05
    kelvin_hartree_relationship: float = 3.1668115634556e-06
    kelvin_hertz_relationship: float = 20836619120.0
    kelvin_inverse_meter_relationship: float = 69.50348004
    kelvin_joule_relationship: float = 1.380649e-23
    kelvin_kilogram_relationship: float = 1.536179187e-40
    kilogram_atomic_mass_unit_relationship: float = 6.0221407621e26
    kilogram_electron_volt_relationship: float = 5.609588603e35
    kilogram_hartree_relationship: float = 2.0614857887409e34
    kilogram_hertz_relationship: float = 1.356392489e50
    kilogram_inverse_meter_relationship: float = 4.524438335e41
    kilogram_joule_relationship: float = 8.987551787e16
    kilogram_kelvin_relationship: float = 6.50965726e39
    lattice_parameter_of_silicon: float = 5.431020511e-10
    lattice_spacing_of_ideal_si_220: float = 1.920155716e-10
    loschmidt_constant_27315_k_100_kpa: float = 2.651645804e25
    loschmidt_constant_27315_k_101325_kpa: float = 2.686780111e25
    luminous_efficacy: float = 683.0
    mag_flux_quantum: float = 2.067833848e-15
    molar_gas_constant: float = 8.314462618
    molar_mass_constant: float = 0.00099999999965
    molar_mass_of_carbon_12: float = 0.0119999999958
    molar_planck_constant: float = 3.990312712e-10
    molar_volume_of_ideal_gas_27315_k_100_kpa: float = 0.02271095464
    molar_volume_of_ideal_gas_27315_k_101325_kpa: float = 0.02241396954
    molar_volume_of_silicon: float = 1.205883199e-05
    molybdenum_x_unit: float = 1.00209952e-13
    muon_compton_wavelength: float = 1.17344411e-14
    muon_electron_mass_ratio: float = 206.768283
    muon_g_factor: float = -2.0023318418
    muon_mag_mom: float = -4.4904483e-26
    muon_mag_mom_anomaly: float = 0.00116592089
    muon_mag_mom_to_bohr_magneton_ratio: float = -0.00484197047
    muon_mag_mom_to_nuclear_magneton_ratio: float = -8.89059703
    muon_mass: float = 1.883531627e-28
    muon_mass_energy_equivalent: float = 1.692833804e-11
    muon_mass_energy_equivalent_in_mev: float = 105.6583755
    muon_mass_in_u: float = 0.1134289259
    muon_molar_mass: float = 0.0001134289259
    muon_neutron_mass_ratio: float = 0.112454517
    muon_proton_mag_mom_ratio: float = -3.183345142
    muon_proton_mass_ratio: float = 0.1126095264
    muon_tau_mass_ratio: float = 0.0594635
    natural_unit_of_action: float = 1.054571817e-34
    natural_unit_of_action_in_ev_s: float = 6.582119569e-16
    natural_unit_of_energy: float = 8.1871057769e-14
    natural_unit_of_energy_in_mev: float = 0.51099895
    natural_unit_of_length: float = 3.8615926796e-13
    natural_unit_of_mass: float = 9.1093837015e-31
    natural_unit_of_momentum: float = 2.73092453075e-22
    natural_unit_of_momentum_in_mevpc: float = 0.51099895
    natural_unit_of_time: float = 1.28808866819e-21
    natural_unit_of_velocity: float = 299792458.0
    neutron_compton_wavelength: float = 1.31959090581e-15
    neutron_electron_mag_mom_ratio: float = 0.00104066882
    neutron_electron_mass_ratio: float = 1838.68366173
    neutron_g_factor: float = -3.82608545
    neutron_gyromag_ratio: float = 183247171.0
    neutron_gyromag_ratio_in_mhzpt: float = 29.1646931
    neutron_mag_mom: float = -9.6623651e-27
    neutron_mag_mom_to_bohr_magneton_ratio: float = -0.00104187563
    neutron_mag_mom_to_nuclear_magneton_ratio: float = -1.91304273
    neutron_mass: float = 1.67492749804e-27
    neutron_mass_energy_equivalent: float = 1.50534976287e-10
    neutron_mass_energy_equivalent_in_mev: float = 939.56542052
    neutron_mass_in_u: float = 1.00866491595
    neutron_molar_mass: float = 0.0010086649156
    neutron_muon_mass_ratio: float = 8.89248406
    neutron_proton_mag_mom_ratio: float = -0.68497934
    neutron_proton_mass_difference: float = 2.30557435e-30
    neutron_proton_mass_difference_energy_equivalent: float = 2.07214689e-13
    neutron_proton_mass_difference_energy_equivalent_in_mev: float = 1.29333236
    neutron_proton_mass_difference_in_u: float = 0.00138844933
    neutron_proton_mass_ratio: float = 1.00137841931
    neutron_relative_atomic_mass: float = 1.00866491595
    neutron_tau_mass_ratio: float = 0.528779
    neutron_to_shielded_proton_mag_mom_ratio: float = -0.68499694
    newtonian_constant_of_gravitation: float = 6.6743e-11
    newtonian_constant_of_gravitation_over_h_bar_c: float = 6.70883e-39
    nuclear_magneton: float = 5.0507837461e-27
    nuclear_magneton_in_evpt: float = 3.15245125844e-08
    nuclear_magneton_in_inverse_meter_per_tesla: float = 0.0254262341353
    nuclear_magneton_in_kpt: float = 0.00036582677756
    nuclear_magneton_in_mhzpt: float = 7.6225932291
    planck_constant: float = 6.62607015e-34
    planck_constant_in_evphz: float = 4.135667696e-15
    planck_length: float = 1.616255e-35
    planck_mass: float = 2.176434e-08
    planck_mass_energy_equivalent_in_gev: float = 1.22089e19
    planck_temperature: float = 1.416784e32
    planck_time: float = 5.391247e-44
    proton_charge_to_mass_quotient: float = 95788331.56
    proton_compton_wavelength: float = 1.32140985539e-15
    proton_electron_mass_ratio: float = 1836.15267343
    proton_g_factor: float = 5.5856946893
    proton_gyromag_ratio: float = 267522187.44
    proton_gyromag_ratio_in_mhzpt: float = 42.577478518
    proton_mag_mom: float = 1.41060679736e-26
    proton_mag_mom_to_bohr_magneton_ratio: float = 0.0015210322023
    proton_mag_mom_to_nuclear_magneton_ratio: float = 2.79284734463
    proton_mag_shielding_correction: float = 2.5689e-05
    proton_mass: float = 1.67262192369e-27
    proton_mass_energy_equivalent: float = 1.50327761598e-10
    proton_mass_energy_equivalent_in_mev: float = 938.27208816
    proton_mass_in_u: float = 1.007276466621
    proton_molar_mass: float = 0.00100727646627
    proton_muon_mass_ratio: float = 8.88024337
    proton_neutron_mag_mom_ratio: float = -1.45989805
    proton_neutron_mass_ratio: float = 0.99862347812
    proton_relative_atomic_mass: float = 1.007276466621
    proton_rms_charge_radius: float = 8.414e-16
    proton_tau_mass_ratio: float = 0.528051
    quantum_of_circulation: float = 0.00036369475516
    quantum_of_circulation_times_2: float = 0.00072738951032
    reduced_compton_wavelength: float = 3.8615926796e-13
    reduced_muon_compton_wavelength: float = 1.867594306e-15
    reduced_neutron_compton_wavelength: float = 2.1001941552e-16
    reduced_planck_constant: float = 1.054571817e-34
    reduced_planck_constant_in_ev_s: float = 6.582119569e-16
    reduced_planck_constant_times_c_in_mev_fm: float = 197.3269804
    reduced_proton_compton_wavelength: float = 2.10308910336e-16
    reduced_tau_compton_wavelength: float = 1.110538e-16
    rydberg_constant: float = 10973731.56816
    rydberg_constant_times_c_in_hz: float = 3289841960250800.0
    rydberg_constant_times_hc_in_ev: float = 13.605693122994
    rydberg_constant_times_hc_in_j: float = 2.1798723611035e-18
    sackur_tetrode_constant_1_k_100_kpa: float = -1.15170753706
    sackur_tetrode_constant_1_k_101325_kpa: float = -1.16487052358
    second_radiation_constant: float = 0.01438776877
    shielded_helion_gyromag_ratio: float = 203789456.9
    shielded_helion_gyromag_ratio_in_mhzpt: float = 32.43409942
    shielded_helion_mag_mom: float = -1.07455309e-26
    shielded_helion_mag_mom_to_bohr_magneton_ratio: float = -0.001158671471
    shielded_helion_mag_mom_to_nuclear_magneton_ratio: float = -2.127497719
    shielded_helion_to_proton_mag_mom_ratio: float = -0.7617665618
    shielded_helion_to_shielded_proton_mag_mom_ratio: float = -0.7617861313
    shielded_proton_gyromag_ratio: float = 267515315.1
    shielded_proton_gyromag_ratio_in_mhzpt: float = 42.57638474
    shielded_proton_mag_mom: float = 1.41057056e-26
    shielded_proton_mag_mom_to_bohr_magneton_ratio: float = 0.001520993128
    shielded_proton_mag_mom_to_nuclear_magneton_ratio: float = 2.792775599
    shielding_difference_of_d_and_p_in_hd: float = 2.02e-08
    shielding_difference_of_t_and_p_in_ht: float = 2.414e-08
    speed_of_light_in_vacuum: float = 299792458.0
    standard_acceleration_of_gravity: float = 9.80665
    standard_atmosphere: float = 101325.0
    standard_state_pressure: float = 100000.0
    stefan_boltzmann_constant: float = 5.670374419e-08
    tau_compton_wavelength: float = 6.97771e-16
    tau_electron_mass_ratio: float = 3477.23
    tau_energy_equivalent: float = 1776.86
    tau_mass: float = 3.16754e-27
    tau_mass_energy_equivalent: float = 2.84684e-10
    tau_mass_in_u: float = 1.90754
    tau_molar_mass: float = 0.00190754
    tau_muon_mass_ratio: float = 16.817
    tau_neutron_mass_ratio: float = 1.89115
    tau_proton_mass_ratio: float = 1.89376
    thomson_cross_section: float = 6.6524587321e-29
    triton_electron_mass_ratio: float = 5496.92153573
    triton_g_factor: float = 5.957924931
    triton_mag_mom: float = 1.5046095202e-26
    triton_mag_mom_to_bohr_magneton_ratio: float = 0.0016223936651
    triton_mag_mom_to_nuclear_magneton_ratio: float = 2.9789624656
    triton_mass: float = 5.0073567446e-27
    triton_mass_energy_equivalent: float = 4.500387806e-10
    triton_mass_energy_equivalent_in_mev: float = 2808.92113298
    triton_mass_in_u: float = 3.01550071621
    triton_molar_mass: float = 0.00301550071517
    triton_proton_mass_ratio: float = 2.99371703414
    triton_relative_atomic_mass: float = 3.01550071621
    triton_to_proton_mag_mom_ratio: float = 1.0666399191
    unified_atomic_mass_unit: float = 1.6605390666e-27
    vacuum_electric_permittivity: float = 8.8541878128e-12
    vacuum_mag_permeability: float = 1.25663706212e-06
    von_klitzing_constant: float = 25812.80745
    weak_mixing_angle: float = 0.2229
    wien_frequency_displacement_law_constant: float = 58789257570.0
    wien_wavelength_displacement_law_constant: float = 0.002897771955
    w_to_z_mass_ratio: float = 0.88153
    calorie_joule_relationship: float = 4.184
    atomic_unit_of_momum: float = 1.9928519141e-24
    planck_constant_over_2_pi: float = 1.054571817e-34
    planck_constant_over_2_pi_in_ev_s: float = 6.582119569e-16
    planck_constant_over_2_pi_times_c_in_mev_fm: float = 197.3269804
    natural_unit_of_momum: float = 2.73092453075e-22
    natural_unit_of_momum_in_mevpc: float = 0.51099895
    electron_gyromag_ratio_over_2_pi: float = 28024.9514242
    mag_constant: float = 1.25663706212e-06
    planck_constant_in_ev_s: float = 4.135667696e-15
    bohr_magneton_in_inverse_meters_per_tesla: float = 46.686447783
    boltzmann_constant_in_inverse_meters_per_kelvin: float = 69.50348004
    cu_x_unit: float = 1.00207697e-13
    mo_x_unit: float = 1.00209952e-13
    proton_gyromag_ratio_over_2_pi: float = 42.577478518
    shielded_proton_gyromag_ratio_over_2_pi: float = 42.57638474
    proton_compton_wavelength_over_2_pi: float = 2.10308910336e-16
    tau_compton_wavelength_over_2_pi: float = 1.110538e-16
    tau_mass_energy_equivalent_in_mev: float = 1776.86
    neutron_compton_wavelength_over_2_pi: float = 2.1001941552e-16
    neutron_gyromag_ratio_over_2_pi: float = 29.1646931
    nuclear_magneton_in_inverse_meters_per_tesla: float = 0.0254262341353
    shielded_helion_gyromag_ratio_over_2_pi: float = 32.43409942
    compton_wavelength_over_2_pi: float = 3.8615926796e-13
    electric_constant: float = 8.8541878128e-12
    muon_compton_wavelength_over_2_pi: float = 1.867594306e-15
    molar_planck_constant_times_c: float = 0.11962656561191261
    faraday_constant_for_conventional_electric_current: float = 96485.3235453493
    elementary_charge_over_h: float = 241798924068654.12
    h: float = 6.62607015e-34
    hbar: float = 1.054571817e-34
    c: float = 299792458.0
    kb: float = 1.380649e-23
    r: float = 8.314462618
    bohr2angstroms: float = 0.529177210903
    bohr2m: float = 5.29177210903e-11
    bohr2cm: float = 5.29177210903e-09
    amu2g: float = 1.6605390666e-24
    amu2kg: float = 1.6605390666e-27
    au2amu: float = 0.000548579909065
    hartree2j: float = 4.3597447222071e-18
    hartree2aj: float = 4.3597447222071
    cal2j: float = 4.184
    dipmom_au2si: float = 8.4783536255e-30
    dipmom_au2debye: float = 2.5417464739297717
    dipmom_debye2si: float = 3.335640951e-30
    c_au: float = 137.035999084
    hartree2ev: float = 27.211386245988
    hartree2wavenumbers: float = 219474.6313632
    hartree2kcalmol: float = 627.5094740630558
    hartree2kjmol: float = 2625.4996394798254
    hartree2mhz: float = 6579683920.502
    na: float = 6.02214076e23
    me: float = 9.1093837015e-31
    kcalmol2wavenumbers: float = 349.755088144347
    e0: float = 8.8541878128e-12


CODATA = _CodataContext()
"""CODATA 2018 values from qcelemental 0.29.0."""


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
