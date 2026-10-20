from __future__ import annotations

from . import _dowker_complex
from gudhi import SimplexTree
import numpy as np
from typing import Literal

try:
    import multipers as _mp
    _HAS_MULTIPERS = True
except ImportError:
    _HAS_MULTIPERS = False

class DowkerComplex:
    """
    Class MNeighborComplex. Constructs Dowker's simplicial complex for a relation.
    Filtrations can be added using filtered relations, or total weight, or combining both into a bifiltration.
    """

    def __init__(self, rel_matrix, max_filtration=float('inf')) -> None:
        """_summary_
        Constructor for the filtered Dowker complex from the relations given by sublevels of the matrix rel_matrix.
        The vertices in the complex will correspond to the rows of the matrix.
        
        Parameters
        ----------
        rel_matrix (Sequence[Sequence[float]]): distance matrix (full square or lower triangular).
        max_filtration (float): specifies the maximal filtration value to be considered.      
        """
        self.rel_matrix = rel_matrix
        self.st = None
        
    def create_simplex_tree_cpp(self, 
                            max_dimension, 
                            filtration = 'None', 
                            m=1, 
                            level = 0, 
                            max_filtration = np.inf):
        cpp_result = _dowker_complex.create_simplex_tree(self.rel_matrix,max_dimension, filtration, m, level, max_filtration)
        st = SimplexTree()
        for (simplex, value) in cpp_result:
            st.insert(simplex, value)
        return st
     

    def create_simplex_tree(self, 
                            max_dimension: int, 
                            filtration : Literal['None', 'Sublevel', 'TotalWeight']= 'None' , 
                            m: int =1, 
                            level: float = 0, 
                            max_filtration: float = np.inf):
        """
            Creates a gudhi simplex tree storing a skeleton of the (filtered) simpicial complex.
            Uses recursive algorithm with nummpy arrays, fast for small datasets but worse runtime and memory complexity.


        Parameters
        ----------
        max_dimension : int
            Dimension of the skeleton to compute.
        filtration : str, optional
            valid values: "Sublevel", "TotalWeight", "None".
            "Sublevel" takes the filtration of relations by sublevels of the matrix.
            "Total weight" takes the filtration by sublevels of the negative total weight function.
            By default 'None'
        m : int, optional
            restriction to this superlevel of total weight (this is only used if filtration!="TotalWeight"); m=1 corresponds to the whole Dowker complex, by default 1
        level : int, optional
            restriction to this sublevel of the matrix as relation (this is only used if filtration!="Sublevel"), by default 0
        max_filtration : float, optional
            cutoff for the filtration (only used if filtration="Sublevel"), by default np.inf

        Returns
        -------
        gudhi.SimplexTree
            The simplex tree storing the (filtered) simplicial complex
        """
        
        self.st=SimplexTree()


        LAMBDA = self.rel_matrix
        num_points=len(LAMBDA)
        
        if filtration == "Sublevel":
            if LAMBDA.dtype != np.float64:
                raise TypeError("Only float arrays are allowed with sublevel filtration") 
            def append_upper_cofaces(sigma, r, witness_values):
                if r > max_filtration:
                    return
                self.st.insert(sigma,r)
                if len(sigma)<=max_dimension:
                    for j in range(np.max(sigma)+1,num_points):
                        tau = sigma+[j]

                        j_witness_values=LAMBDA[j,:]
                        common_witness_values = np.maximum(j_witness_values,witness_values)
                        new_r = np.partition(common_witness_values, m-1)[m-1]
                        append_upper_cofaces(tau, new_r, common_witness_values)

            for k in range(num_points-1,-1,-1):
                witness_values = LAMBDA[k,:]
                r_new = np.partition(witness_values, m-1)[m-1]
                append_upper_cofaces([k],r_new,witness_values)
            return self.st
        
        elif filtration == "TotalWeight":
            if LAMBDA.dtype != np.bool_:
                LAMBDA = LAMBDA <= level

            def append_upper_cofaces(sigma, witnesses):
                
                self.st.insert(sigma,-np.sum(witnesses))
                if len(sigma)<=max_dimension:
                    for j in range(np.max(sigma)+1,num_points):
                        tau = sigma+[j]
                        j_witnesses=LAMBDA[j,:]
                        common_witnesses = np.logical_and(j_witnesses,witnesses)
                        if np.sum(common_witnesses>0):
                            append_upper_cofaces(tau, common_witnesses)

            for k in range(num_points-1,-1,-1):
                witnesses = LAMBDA[k,:]
                append_upper_cofaces([k], witnesses)
            return self.st

        elif filtration == "None":
            if LAMBDA.dtype != np.bool_:
                LAMBDA = LAMBDA <= level

            def append_upper_cofaces(sigma, witnesses):
                if len(witnesses)<m:
                    return
                self.st.insert(sigma)
                if len(sigma)<=max_dimension:
                    for j in range(np.max(sigma)+1,num_points):
                        tau = sigma+[j]
                        j_witnesses=LAMBDA[j,:]
                        common_witnesses = np.logical_and(j_witnesses,witnesses)
                        if len(common_witnesses>0):
                            append_upper_cofaces(tau, common_witnesses)

            for k in range(num_points-1,-1,-1):
                witnesses = LAMBDA[k,:]
                append_upper_cofaces([k], witnesses)
            return self.st
        
        else:
            raise Exception("filtration parameter must be one of 'Sublevel', 'TotalWeight', 'None'")


    def create_rivet_bifiltration(self, max_dimension, m_max=5, normalize = False):
        """
        
        creates a rivet bifiltration storing the filtered simpicial complex.
            Uses recursive algorithm with nummpy arrays, fast for small datasets but worse runtime and memory complexity.
        

        Parameters
        ----------
        max_dimension : int
            Dimension of the skeleton to compute.
        m_max : int, optional
            Maximal number of witnesses to include, by default 5
        normalize : bool, optional
            whether to normalize the number of neighbors to the interval [0,1], by default False

        Returns
        -------
        rivet.Bifiltration
            A RIVET-compatible bifiltration listing simplices and their bidegrees of appearance.
        """
        

        LAMBDA = self.rel_matrix
        num_points=len(LAMBDA)

        simplices = []
        appearances = []

        if m_max > self.rel_matrix.shape[1]:
            m_max = self.rel_matrix.shape[1]
            

        def append_upper_cofaces(sigma, witness_values):

            simplices.append(sigma)
            sorted_witness_values = np.sort(witness_values)
            this_appearances = []
            for i in range(1,m_max+1):
                
                if np.isinf(sorted_witness_values[i-1]):
                    break

                if normalize:
                    this_appearances.append((sorted_witness_values[i-1],i/(LAMBDA.shape[1])))
                else:            
                    this_appearances.append((sorted_witness_values[i-1],i))
            appearances.append(this_appearances)
            if len(sigma)<=max_dimension:
                for j in range(np.max(sigma)+1,num_points):
                    tau = sigma+[j]

                    j_witness_values=LAMBDA[j,:]
                    common_witness_values = np.maximum(j_witness_values,witness_values)
                    append_upper_cofaces(tau, common_witness_values)

        for k in range(num_points-1,-1,-1):
            witness_values = LAMBDA[k,:]
            append_upper_cofaces([k],witness_values)
        

        return {tuple(simplices[i]): appearances[i] for i in range(len(simplices))}
    

    def create_rivet_bifiltration_cpp(self, max_dimension, m_max=5):
        """
        Calls the experimental C++ implementation, which is faster but currently only handles float matrices.

        Parameters
        ----------
        max_dimension : int
            Dimension of the skeleton to compute.
        m_max : int, optional
            Maximal number of witnesses to include, by default 5

        Returns
        -------
        rivet.Bifiltration
            A RIVET-compatible bifiltration listing simplices and their bidegrees of appearance.
        """
        simplices,appearances = _dowker_complex.create_bifiltration(self.rel_matrix, max_dimension, m_max)
        return {tuple(simplices[i]): appearances[i] for i in range(len(simplices))}
    
    
    def create_multipers_bifiltration(self, max_dimension: int | None, m_max: int | None = 5):
        """
        Calls the experimental C++ implementation, which is faster but currently only handles float matrices.

        Parameters
        ----------
        max_dimension : int
            Dimension of the skeleton to compute.
        m_max : int, optional
            Maximal number of witnesses to include, by default 5

        Returns
        -------
        multipers.SimplexTreeMulti
            A multipers bifiltration listing simplices and their bidegrees of appearance.

        Raises
        ------
        ImportError
            If the ``multipers`` package is not installed.
            Install it with ``pip install pyDowker[multipers]``.
        """
        if not _HAS_MULTIPERS:
            raise ImportError(
                "The 'multipers' package is required for this method. "
                "Install it with: pip install pyDowker[multipers]"
            )
        st = _mp.SimplexTreeMulti(num_parameters=2, kcritical=True)
        simplices, appearances = _dowker_complex.create_bifiltration(self.rel_matrix, max_dimension, m_max)
        for i in range(len(simplices)):
            for a in appearances[i]:
                st.insert(simplices[i], [a[0], -a[1]])
        return st

    def create_subdivision_simplex_tree(self, max_dimension: int, level: float = 0):
        """
        Creates a simplex tree for the subdivision filtration S(D(Y, X, R^T)) of
        the dual Dowker complex (Theorem 3.3 of the preprint):

            |D(X, Y, R)_•| ≃ |S(D(Y, X, R^T))_•|

        Vertices of the returned simplex tree are integer indices labelling the
        simplices of D(Y, X, R^T).  A flag σ_0 ⊊ ... ⊊ σ_k in the barycentric
        subdivision Sd(D(Y, X, R^T)) is inserted with filtration value -|σ_0|,
        matching the negative convention of the TotalWeight filtration.

        Parameters
        ----------
        max_dimension : int
            Maximum flag dimension to include (= max simplex dimension in output).
        level : float, optional
            For non-boolean matrices, entries ≤ level are treated as witnesses.
            By default 0.

        Returns
        -------
        gudhi.SimplexTree
        """
        LAMBDA = self.rel_matrix
        if LAMBDA.dtype != np.bool_:
            LAMBDA = LAMBDA <= level

        num_cols = LAMBDA.shape[1]

        # Enumerate all simplices of D(Y, X, R^T).
        # A set of column indices is a simplex iff some row witnesses all of them.
        dual_simplices = []

        def build_dual(sigma, witnesses):
            dual_simplices.append(tuple(sigma))
            if len(sigma) <= max_dimension:
                for j in range(sigma[-1] + 1, num_cols):
                    new_witnesses = witnesses & LAMBDA[:, j]
                    if np.any(new_witnesses):
                        build_dual(sigma + [j], new_witnesses)

        for j in range(num_cols):
            col_witnesses = LAMBDA[:, j]
            if np.any(col_witnesses):
                build_dual([j], col_witnesses)

        # Sort by (size, lex): any chain σ_0 ⊊ ... ⊊ σ_k has strictly increasing
        # sizes, so its indices in this list are automatically in increasing order.
        dual_simplices.sort(key=lambda s: (len(s), s))
        dual_simplex_sets = [frozenset(s) for s in dual_simplices]

        st = SimplexTree()

        def extend_flag(chain_indices, top_idx, bottom_size):
            st.insert(chain_indices, float(-bottom_size))
            if len(chain_indices) <= max_dimension:
                top_set = dual_simplex_sets[top_idx]
                for next_idx in range(top_idx + 1, len(dual_simplices)):
                    if top_set < dual_simplex_sets[next_idx]:
                        extend_flag(chain_indices + [next_idx], next_idx, bottom_size)

        for i in range(len(dual_simplices)):
            extend_flag([i], i, len(dual_simplices[i]))

        return st

    def euler_profile_contributions(self, m_max=10):
        """
            creates a list of contributions of bifiltration values to the Euler characteristic profile.


        Parameters
        ----------
        m_max : int, optional
            Maximal number of witnesses to include, by default 10

        Returns
        -------
        list
            list of pairs (bidegree, contribution), where contribution is either +1 or -1, giving the contributions to the Euler characteristic profile.
        """
        
        LAMBDA = self.distance_matrix
        num_points=len(LAMBDA)
        max_dimension = num_points
        contributions = []

        def append_upper_cofaces(sigma, witness_values):

            sorted_witness_values = np.sort(witness_values)
            for i in range(1,m_max+1):
                contributions.append(((sorted_witness_values[i-1],i),(-1)**(len(sigma)-1)))
            for i in range(1,m_max):
                contributions.append(((sorted_witness_values[i],i),(-1)**(len(sigma))))
            

            if len(sigma)<=max_dimension:
                for j in range(np.max(sigma)+1,num_points):
                    tau = sigma+[j]

                    j_witness_values=LAMBDA[j,:]
                    common_witness_values = np.maximum(j_witness_values,witness_values)
                    append_upper_cofaces(tau, common_witness_values)

        for k in range(num_points-1,-1,-1):
            witness_values = LAMBDA[k,:]
            append_upper_cofaces([k],witness_values)
            
        return contributions
  
    