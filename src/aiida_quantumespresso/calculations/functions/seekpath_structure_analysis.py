# -*- coding: utf-8 -*-
"""Calcfunction to primitivize a structure and return high symmetry k-point path through its Brillouin zone."""
from aiida.engine import calcfunction
from aiida.orm import Data
import numpy as np
from aiida.plugins import DataFactory, WorkflowFactory
from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData


@calcfunction
def seekpath_structure_analysis(structure, **kwargs):
    """Primitivize the structure with SeeKpath and generate the high symmetry k-point path through its Brillouin zone.

    This calcfunction will take a structure and pass it through SeeKpath to get the normalized primitive cell and the
    path of high symmetry k-points through its Brillouin zone. Note that the returned primitive cell may differ from the
    original structure in which case the k-points are only congruent with the primitive cell.

    The keyword arguments can be used to specify various Seekpath parameters, such as:

        with_time_reversal: True
        reference_distance: 0.025
        recipe: 'hpkot'
        threshold: 1e-07
        symprec: 1e-05
        angle_tolerance: -1.0

    Note that exact parameters that are available and their defaults will depend on your Seekpath version.
    """
    from aiida.tools import get_explicit_kpoints_path

    # All keyword arugments should be `Data` node instances of base type and so should have the `.value` attribute
    unwrapped_kwargs = {key: node.value for key, node in kwargs.items() if isinstance(node, Data)}
    try:
        result['explicit_kpoints'] = get_explicit_kpoints_path(structure, **unwrapped_kwargs)
    except:
        result = {}
        result['primitive_structure'] = structure
        result['explicit_kpoints'] = generate_kpath_2d(structure=structure, kpoints_distance=unwrapped_kwargs.reference_distance, kpath_2d=4)
    if isinstance(structure, HubbardStructureData):
        result['primitive_structure'] = update_structure_with_hubbard(result['primitive_structure'], structure)
        result['conv_structure'] = update_structure_with_hubbard(result['conv_structure'], structure)

    return result


def update_structure_with_hubbard(structure, orig_structure):
    """Update the structure based on Hubbard parameters if the input structure is a HubbardStructureData."""
    from aiida_quantumespresso.utils.hubbard import is_intersite_hubbard

    hubbard_structure = HubbardStructureData.from_structure(structure)

    if is_intersite_hubbard(orig_structure.hubbard):
        raise NotImplementedError('Intersite Hubbard parameters are not yet supported.')

    for parameter in orig_structure.hubbard.parameters:
        hubbard_structure.initialize_onsites_hubbard(
            atom_name=orig_structure.sites[parameter.atom_index].kind_name,
            atom_manifold=parameter.atom_manifold,
            value=parameter.value,
            hubbard_type=parameter.hubbard_type,
            use_kinds=True,
        )

    return hubbard_structure


def calculate_bands_kpoints_distance(kpoints_distance):
    """function to calculate the bands_kpoints_distance depending on the kpoints_distance"""
    if kpoints_distance >= 0.5:
        return 0.1
    elif 0.15 < kpoints_distance < 0.5:
        return 0.025
    else:
        return 0.015

def points_per_branch(vector_a, vector_b, reciprocal_cell, bands_kpoints_distance):
    """function to calculate the number of points per branch depending on the kpoints_distance and the reciprocal cell"""
    scaled_vector_a = np.array(vector_a)
    scaled_vector_b = np.array(vector_b)
    reciprocal_vector_a = scaled_vector_a.dot(reciprocal_cell)
    reciprocal_vector_b = scaled_vector_b.dot(reciprocal_cell)
    distance = np.linalg.norm(reciprocal_vector_a - reciprocal_vector_b)
    return round(distance / bands_kpoints_distance)



def generate_kpath_2d(structure, kpoints_distance, kpath_2d):
    """ Implementation by Xing Wang,  Jusong Yu , Andres Ortega Guerrero
    Return a kpoints object for two dimensional systems based on the selected 2D symmetry path
    The number of kpoints is calculated based on the kpoints_distance (as in the PwBandsWorkChain protocol)
    """
    GAMMA = "\u0393"

    KpointsData = DataFactory("core.array.kpoints")    
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    reciprocal_cell = kpoints.reciprocal_cell
    bands_kpoints_distance = calculate_bands_kpoints_distance(kpoints_distance)

    # dictionary with the 2D symmetry paths
    selected_paths = {
        4: {
            "path": [
                [0.0, 0.0, 0.0],
                [0.33333, 0.33333, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "labels": [GAMMA, "K", "M", GAMMA],
        },
        "square": {
            "path": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "labels": [GAMMA, "X", "M", GAMMA],
        },
        "rectangular": {
            "path": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "labels": [GAMMA, "X", "S", "Y", GAMMA],
        },
    }
    # if the selected path is centered_rectangular or oblique, the path is calculated based on the reciprocal cell
    if kpath_2d in ["centered_rectangular", "oblique"]:
        a1 = reciprocal_cell[0]
        a2 = reciprocal_cell[1]
        norm_a1 = np.linalg.norm(a1)
        norm_a2 = np.linalg.norm(a2)
        cos_gamma = (
            a1.dot(a2) / (norm_a1 * norm_a2)
        )  # Angle between a1 and a2 # like in https://pubs.acs.org/doi/10.1021/acs.jpclett.2c02972
        gamma = np.arccos(cos_gamma)
        eta = (1 - (norm_a1 / norm_a2) * cos_gamma) / (2 * np.power(np.sin(gamma), 2))
        nu = 0.5 - (eta * norm_a2 * cos_gamma) / norm_a1
        selected_paths["centered_rectangular"] = {
            "path": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1 - eta, nu, 0],
                [0.5, 0.5, 0.0],
                [eta, 1 - nu, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "labels": [GAMMA, "X", "H_1", "C", "H", GAMMA],
        }
        selected_paths["oblique"] = {
            "path": [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1 - eta, nu, 0],
                [0.5, 0.5, 0.0],
                [eta, 1 - nu, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ],
            "labels": [GAMMA, "X", "H_1", "C", "H", "Y", GAMMA],
        }
    points_branch = []
    num_per_branch = []
    path = selected_paths[kpath_2d]["path"]
    labels = selected_paths[kpath_2d]["labels"]
    branches = zip(
        path[:-1], path[1:]
    )  # zip the path with the next point in the path to define the branches

    # Calculate the number of points per branch and generate the kpoints
    for branch in branches:
        num_points_per_branch = points_per_branch(
            branch[0], branch[1], reciprocal_cell, bands_kpoints_distance
        )
        if branch[1] == [1.0, 0.0, 0.0]:
            points = np.linspace(
                start=branch[0],
                stop=branch[1],
                endpoint=True,
                num=num_points_per_branch,
            )
        else:
            points = np.linspace(
                start=branch[0], stop=branch[1], num=num_points_per_branch
            )
        points_branch.append(points.tolist())
        num_per_branch.append(num_points_per_branch)

    # Generate the kpoints as single list and add the labels
    list_kpoints = [item for sublist in points_branch for item in sublist]
    kpoints.set_kpoints(list_kpoints)
    kpoints.labels = [
        [index, labels[index]]
        if index == 0
        else [list_kpoints.index(value, 1), labels[index]]
        for index, value in enumerate(path)
    ]
    return kpoints
