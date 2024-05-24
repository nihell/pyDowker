"""
Topological fingerprints for macromolecular structures.
"""
import numpy as np
import logging
import itertools
from deepchem.utils.hash_utils import hash_ecfp
from deepchem.feat import ComplexFeaturizer
from deepchem.utils.rdkit_utils import load_complex
from deepchem.utils.hash_utils import vectorize
from deepchem.utils.voxel_utils import voxelize
from deepchem.utils.voxel_utils import convert_atom_to_voxel
from deepchem.utils.rdkit_utils import compute_all_ecfp
from deepchem.utils.rdkit_utils import compute_contact_centroid
from deepchem.utils.rdkit_utils import MoleculeLoadException
from deepchem.utils.geometry_utils import compute_pairwise_distances
from deepchem.utils.geometry_utils import subtract_centroid
import gudhi as gd
from gudhi.representations import BettiCurve, Landscape
from pyrivet import rivet


import os
import sys
sys.path.append(os.getcwd() + '/..')
from pyDowker.DowkerComplex import DowkerComplex
from pyDowker.TwoParameterUtils import discretize_graded_rank, grid_ECP
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

def featurize_contacts_dowker(
    frag1: Tuple,
    frag2: Tuple,
    pairwise_distances: np.ndarray = None,
    cutoff: float = 4.5,
    m: int = 1) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Computes ECFP dicts for pairwise interaction between two molecular fragments.

    Parameters
    ----------
    frag1: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
    frag2: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
    pairwise_distances: np.ndarray
    Array of pairwise fragment-fragment distances (Angstroms)
    cutoff: float
    Cutoff distance for contact consideration
    m: int
    number of witnesses

    Returns
    -------
    Pair of lists of persistence diagrams ind dimensions 0 and 1
    """
    if pairwise_distances is None:
        pairwise_distances = compute_pairwise_distances(frag1[0], frag2[0])
        
    ligand_atoms = np.array([a.GetSymbol() for a in frag1[1].GetAtoms()])
    protein_atoms = np.array([a.GetSymbol() for a in frag2[1].GetAtoms()])
    
    local_dgm0 = []
    local_dgm1 = []
    for la in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
        for pa in ['C', 'N', 'O', 'S']:
            local_dist = pairwise_distances[ligand_atoms==la,:][:,protein_atoms==pa]
            if local_dist.size>0:
                mnc = DowkerComplex(local_dist, max_filtration=4.5)
                st = mnc.create_simplex_tree(m=m, filtration="Sublevel", max_dimension=2)
                st.compute_persistence()
                local_dgm0.append(st.persistence_intervals_in_dimension(0))
                local_dgm1.append(st.persistence_intervals_in_dimension(1))
            else:
                local_dgm0.append(np.empty((0,2)))
                local_dgm1.append(np.empty((0,2)))

    return (local_dgm0,local_dgm1)

def featurize_contacts_dowker_bifi(
    frag1: Tuple,
    frag2: Tuple,
    pairwise_distances: np.ndarray = None,
    cutoff: float = 4.5,
    m_max: int = 10) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Computes ECFP dicts for pairwise interaction between two molecular fragments.

    Parameters
    ----------
    frag1: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
    frag2: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
    pairwise_distances: np.ndarray
    Array of pairwise fragment-fragment distances (Angstroms)
    cutoff: float
    Cutoff distance for contact consideration
    m_max: int
    maximal number of witnesses to consider.

    Returns
    -------
    pair of lists of persistence diagrams for dimensions 0 and 1 with number of witnesses from 1 to m_max
    """
    if pairwise_distances is None:
        pairwise_distances = compute_pairwise_distances(frag1[0], frag2[0])
        
    ligand_atoms = np.array([a.GetSymbol() for a in frag1[1].GetAtoms()])
    protein_atoms = np.array([a.GetSymbol() for a in frag2[1].GetAtoms()])
    
    local_dgm0 = []
    local_dgm1 = []
    for la in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
        for pa in ['C', 'N', 'O', 'S']:
            local_dist = pairwise_distances[ligand_atoms==la,:][:,protein_atoms==pa]
            for m in range(1,m_max+1):
                if local_dist.shape[0]>=m and local_dist.shape[1]>=m:
                    mnc = DowkerComplex(local_dist, max_filtration=cutoff)
                    st = mnc.create_simplex_tree(max_dimension=2, m=m, filtration="Sublevel")
                    st.compute_persistence()
                    local_dgm0.append(st.persistence_intervals_in_dimension(0))
                    local_dgm1.append(st.persistence_intervals_in_dimension(1))
                #bifi = mnc.create_rivet_bifiltration(max_dimension=2,m_max=m_max)
                #betti0 = rivet.betti(bifi,homology=0,x=m_max,y=m_max)
                #betti1 = rivet.betti(bifi,homology=1,x=m_max,y=m_max)
                #lsc0 = multiparameter_landscape(computed_data=rivet._compute_bytes(bifi,homology=0,x=m_max,y=m_max, verify=False), grid_step_size=1).compute_multiparameter_landscape()
                #lsc0 = multiparameter_landscape(computed_data=rivet._compute_bytes(bifi,homology=1,x=m_max,y=m_max, verify=False), grid_step_size=1).compute_multiparameter_landscape()
                
                else:
                    local_dgm0.append(np.empty((0,2)))
                    local_dgm1.append(np.empty((0,2)))
                #lsc0=np.zeros((m_max,m_max))
                #lsc1=np.zeros((m_max,m_max))
#                 local_betti0.append(rivet.MultiBetti(dimensions = rivet.Dimensions([0],[0]), 
#                                                      graded_rank=np.array([[0.]]),
#                                                      xi_0=[],xi_1=[],xi_2=[]))
#                 local_betti1.append(rivet.MultiBetti(dimensions = rivet.Dimensions([0],[0]), 
#                                                      graded_rank=np.array([[0.]]),
#                                                      xi_0=[],xi_1=[],xi_2=[]))
#             local_betti0.append(lsc0)
#             local_betti1.append(lsc1)     
    return (local_dgm0,local_dgm1)


def featurize_contacts_dowker_ecp(
    frag1: Tuple,
    frag2: Tuple,
    pairwise_distances: np.ndarray = None,
    cutoff: float = 4.5,
    m_max: int = 10) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Computes ECFP dicts for pairwise interaction between two molecular fragments.

    Parameters
    ----------
    frag1: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
    frag2: Tuple
    A tuple of (coords, mol) returned by `load_molecule`.
    pairwise_distances: np.ndarray
    Array of pairwise fragment-fragment distances (Angstroms)
    cutoff: float
    Cutoff distance for contact consideration
    m_max: int
    maximal number of witnesses

    Returns
    -------
    list of ECPs encoded via contributions
    """
    if pairwise_distances is None:
        pairwise_distances = compute_pairwise_distances(frag1[0], frag2[0])
        
    ligand_atoms = np.array([a.GetSymbol() for a in frag1[1].GetAtoms()])
    protein_atoms = np.array([a.GetSymbol() for a in frag2[1].GetAtoms()])
    
    contribs = []
    for la in ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']:
        for pa in ['C', 'N', 'O', 'S']:
            local_dist = pairwise_distances[ligand_atoms==la,:][:,protein_atoms==pa]
            if local_dist.size>0:
                mnc = DowkerComplex(pairwise_distances, max_filtration=cutoff)
                contribs.append( mnc.euler_profile_contributions(m_max=m_max))
                
            else:
                contribs.append([])
    return contribs


class DowkerFeaturizer(ComplexFeaturizer):

    def __init__(self, cutoff: float = 4.5, size = 128, m=1):
        self.cutoff = cutoff
        self.size = size
        self.m=1
        
    def _featurize(self, datapoint, **kwargs):
        """
        Compute featurization for a molecular complex

        Parameters
        ----------
        datapoint: Tuple[str, str]
          Filenames for molecule and protein.
        """
        if 'complex' in kwargs:
            datapoint = kwargs.get("complex")
            raise DeprecationWarning(
              'Complex is being phased out as a parameter, please pass "datapoint" instead.'
          )

        try:
            fragments = load_complex(datapoint, add_hydrogens=False)

        except MoleculeLoadException:
            logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
            return None
        pairwise_features = []
        # We compute pairwise contact fingerprints
        dgm0 = []
        dgm1 = []
        for (frag1, frag2) in itertools.combinations(fragments, 2):
          # Get coordinates
            distances = compute_pairwise_distances(frag1[0], frag2[0])
            
            d0,d1 = featurize_contacts_dowker(frag1, frag2, distances, self.cutoff, self.m)
            dgm0.append(d0)
            dgm1.append(d1)
        #bc = Landscape(num_landscapes = 4, resolution=self.size, sample_range = (0,100))
        bc=BettiCurve(predefined_grid=np.linspace(0,100,self.size))
        bc0 = np.concatenate(bc.fit_transform(dgm0[0]))
        bc1 = np.concatenate(bc.fit_transform(dgm1[0]))
        pairwise_features = np.concatenate([bc0,bc1])
        return pairwise_features
    
class DowkerBifiFeaturizer(ComplexFeaturizer):

    def __init__(self, cutoff: float = 4.5, size = 128, m_max=10):
        self.cutoff = cutoff
        self.size = size
        self.m_max=m_max
        
    def _featurize(self, datapoint, **kwargs):
        """
        Compute featurization for a molecular complex

        Parameters
        ----------
        datapoint: Tuple[str, str]
          Filenames for molecule and protein.
        """
        if 'complex' in kwargs:
            datapoint = kwargs.get("complex")
            raise DeprecationWarning(
              'Complex is being phased out as a parameter, please pass "datapoint" instead.'
          )

        try:
            fragments = load_complex(datapoint, add_hydrogens=False)

        except MoleculeLoadException:
            logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
            return None
        pairwise_features = []
        # We compute pairwise contact fingerprints
        dgm0 = []
        dgm1 = []
        for (frag1, frag2) in itertools.combinations(fragments, 2):
          # Get coordinates
            distances = compute_pairwise_distances(frag1[0], frag2[0])
            
            d0,d1 = featurize_contacts_dowker_bifi(frag1, frag2, distances, self.cutoff, self.m_max)
            dgm0.append(d0)
            dgm1.append(d1)
        #db0 = [discretize_graded_rank(b, x_grid = np.linspace(0,100,self.size), y_grid = np.arange(1,self.m_max,self.m_max)) for b in b0]
        #db1 = [discretize_graded_rank(b, x_grid = np.linspace(0,100,self.size), y_grid = np.arange(1,self.m_max,self.m_max)) for b in b1]
        #db0 = np.concatenate(b0)
        #db1 = np.concatenate(b1)
        #pairwise_features = np.concatenate([db0.flatten(),db1.flatten()])
        #bc = Landscape(num_landscapes = 4, resolution=self.size, sample_range = (0,100))
        bc=BettiCurve(predefined_grid=np.linspace(0,100,self.size))       
        bc0 = np.concatenate(bc.fit_transform(dgm0[0]))
        bc1 = np.concatenate(bc.fit_transform(dgm1[0]))
        pairwise_features =  np.concatenate([bc0,bc1])
        return pairwise_features
    
    
class DowkerECPFeaturizer(ComplexFeaturizer):

    def __init__(self, cutoff: float = 4.5, size = 128, m_max=10):
        self.cutoff = cutoff
        self.size = size
        self.m_max=m_max
        
    def _featurize(self, datapoint, **kwargs):
        """
        Compute featurization for a molecular complex

        Parameters
        ----------
        datapoint: Tuple[str, str]
          Filenames for molecule and protein.
        """
        if 'complex' in kwargs:
            datapoint = kwargs.get("complex")
            raise DeprecationWarning(
              'Complex is being phased out as a parameter, please pass "datapoint" instead.'
          )

        try:
            fragments = load_complex(datapoint, add_hydrogens=False)

        except MoleculeLoadException:
            logger.warning("This molecule cannot be loaded by Rdkit. Returning None")
            return None
        pairwise_features = []
        # We compute pairwise contact fingerprints
        ecps = []
        for (frag1, frag2) in itertools.combinations(fragments, 2):
          # Get coordinates
            distances = compute_pairwise_distances(frag1[0], frag2[0])
            
            contribs = featurize_contacts_dowker_ecp(frag1, frag2, distances, self.cutoff, self.m_max)
            
        ecp = [grid_ECP(contrib, x_grid = np.linspace(0,100,self.size), y_grid = np.linspace(0,self.m_max,self.size)) for contrib in contribs]
       
        pairwise_features = [e.flatten() for e in ecp]
        return np.concatenate(pairwise_features)