from gudhi import SimplexTree
import numpy as np
from pyrivet import rivet
import subprocess
import tempfile

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
        
        
     

    def create_simplex_tree(self, 
                            max_dimension, 
                            filtration = 'None', 
                            m=1, 
                            level = 0, 
                            max_filtration = np.inf):
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
            if LAMBDA.dtype != np.float_:
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

        def append_upper_cofaces(sigma, witness_values):

            simplices.append(sigma)
            sorted_witness_values = np.sort(witness_values)
            if normalize:
                appearances.append([(sorted_witness_values[i-1],i/(LAMBDA.shape[1])) for i in range(1,m_max+1)])
            else:            
                appearances.append([(sorted_witness_values[i-1],i) for i in range(1,m_max+1)])

            if len(sigma)<=max_dimension:
                for j in range(np.max(sigma)+1,num_points):
                    tau = sigma+[j]

                    j_witness_values=LAMBDA[j,:]
                    common_witness_values = np.maximum(j_witness_values,witness_values)
                    append_upper_cofaces(tau, common_witness_values)

        for k in range(num_points-1,-1,-1):
            witness_values = LAMBDA[k,:]
            append_upper_cofaces([k],witness_values)
        

        bf = rivet.Bifiltration(
            x_label = "relation_value",
            y_label = "num_witnesses",
            simplices = simplices,
            appearances = appearances,
            yreverse = True
        )

        return bf
    

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

        np.savetxt('tmp_dist_',self.distance_matrix,delimiter=',')
        run_args = ["./a.out", 'tmp_dist_', '{}'.format(max_dimension), '{}'.format(m_max)]
        if self.point_is_its_own_neighbor:
            run_args.append("closed")
        subprocess.run(run_args)
        f = open("tmp_dist_mneighbor.bifi")
        return f
    
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
  
    