# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import cupy as cp
from cffi import FFI
import pickle
import os
import gc
# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class sample:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=os.getcwd()):
        self.directory = directory
        self.dimensions = None #Todo: make property
        self.offset = None #Todo: make property
        self.chunk_volume = None #Todo: make property
        self._chunk_total = None 
        self._matrix = None
        self._corners = None
        self._default_filenames = np.array(["atomic_positions.npy","atomic_species.npy","sample_metadata.npy"]) #sample_metadata will be a struct
        # self._chunk_filenames = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
    def create_sample(self,dimensions,offset=[0,0,0],chunk_volume=(600*600*600)):
        self.dimensions = np.array(dimensions,dtype=np.float32)
        self.offset = np.array(offset,dtype=np.float32)
        self.chunk_volume = np.array(chunk_volume,dtype=np.float32)
        
    def read_sample(self): #incomplete
        ## Once write format is complete, finish this.
        ## Reads sample metadata
        self.dimensions = None
        self.offset = None
        self.chunk_volume = None
        self._chunk_total = None
    
    ## Data Handling Functions
    def write_chunk_positions(self,data,chunk_num):
        base, ext = os.path.splitext(self._default_filenames[0])
        chunk_filename = f"{base}_{chunk_num}{ext}"
        np.save(os.path.join(self.directory,chunk_filename), data)
    
    def write_chunk_species(self,data,chunk_num):
        base, ext = os.path.splitext(self._default_filenames[1])
        chunk_filename = f"{base}_{chunk_num}{ext}"
        np.save(os.path.join(self.directory,chunk_filename), data)
            
    def write_sample_metadata(self): #incomplete
        sample_metadata = [self.dimensions,self.offset,self.chunk_dimensions,self.chunk_total]
        
    def load_chunk_positions(self,chunk_number,gpu=True):
        # Load a specific chunk as either GPU or CPU array
        base, ext = os.path.splitext(self._default_filenames[0])
        positions_filename = f"{base}_{chunk_number}{ext}"
        if gpu:
            positions_chunk = cp.load(os.path.join(self.directory,positions_filename))
        else:    
            positions_chunk = np.load(os.path.join(self.directory,positions_filename))
        return positions_chunk
    
    def load_chunk_species(self,chunk_number,gpu=True):
        base, ext = os.path.splitext(self._default_filenames[1])
        species_filename = f"{base}_{chunk_number}{ext}"
        if gpu:
            species_chunk = cp.load(os.path.join(self.directory,species_filename))
        else:    
            species_chunk = np.load(os.path.join(self.directory,species_filename))
        return species_chunk
    
    ## Static Functions
    @staticmethod
    def get_unit_corners():
        unit_corners = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]])
        return unit_corners
    
    @staticmethod
    def get_flat_grid(dimensions,gpu=False):
        if gpu:
            # Grid chunk points
            ii, jj, kk = cp.meshgrid(cp.arange(dimensions[0], dtype=cp.float32), 
                                    cp.arange(dimensions[1], dtype=cp.float32), 
                                    cp.arange(dimensions[2], dtype=cp.float32), indexing='ij')
            # Generate positions in the crystal frame
            flat_grid_cp = cp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
            return flat_grid_cp
        else:
            # Grid chunk points
            ii, jj, kk = np.meshgrid(np.arange(dimensions[0], dtype=np.float32), 
                                    np.arange(dimensions[1], dtype=np.float32), 
                                    np.arange(dimensions[2], dtype=np.float32), indexing='ij')
            # Generate positions in the crystal frame
            flat_grid = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
            return flat_grid
    
    @staticmethod    
    def compile_parallelepipeds_intersect_batch_cffi():
        '''
        C++ code using 15-axis SAT method for determining if a set of cornerpoints intersects with another,
        made to run a batch operation of corner point against a single reference.
        '''
        c_source = r'''
        #include <math.h>
        #include <stdlib.h> // for malloc/free if needed

        // Dot product
        static double dot3(const double *a, const double *b){
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        }

        // Cross product out = a x b
        static void cross3(const double *a, const double *b, double *out){
            out[0] = a[1]*b[2] - a[2]*b[1];
            out[1] = a[2]*b[0] - a[0]*b[2];
            out[2] = a[0]*b[1] - a[1]*b[0];
        }

        // Norm of a 3D vector
        static double norm3(const double *v){
            return sqrt(dot3(v,v));
        }

        // Project 8 points onto an axis
        // out[0] = min, out[1] = max of the projection
        static void project_points(const double *pts8x3, const double *axis, double eps, double *out){
            double axis_len = norm3(axis);
            if(axis_len < eps){
                // Degenerate axis -> all points project to zero
                out[0] = 0.0; 
                out[1] = 0.0;
                return;
            }
            double ax[3] = { axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len };

            double val = dot3(pts8x3, ax); // first corner
            double minv = val, maxv = val;
            for(int i=1; i<8; i++){
                val = dot3(pts8x3 + 3*i, ax);
                if(val < minv) minv = val;
                if(val > maxv) maxv = val;
            }
            out[0] = minv;
            out[1] = maxv;
        }

        // Check if intervals [a0,a1] and [b0,b1] overlap
        static int intervals_overlap(const double *a, const double *b){
            // If one interval is strictly to the left of the other, no overlap
            if(a[1] < b[0] || b[1] < a[0]) 
                return 0;
            return 1;
        }

        // single_intersect: checks intersection for one pair of parallelepipeds
        // pts1, pts2 each has 8 corners -> 24 doubles
        static int single_intersect(const double *pts1, const double *pts2, double eps)
        {
            // 1) Identify shape1 edges from the known corner ordering
            //    c1 = pts1[0], e1 = pts1[1] - pts1[0], e2 = pts1[2] - pts1[0], e3 = pts1[3] - pts1[0].
            double c1[3]  = { pts1[0], pts1[1], pts1[2] };
            double e1[3]  = { pts1[3] - c1[0], pts1[4] - c1[1], pts1[5] - c1[2] };
            double e2[3]  = { pts1[6] - c1[0], pts1[7] - c1[1], pts1[8] - c1[2] };
            double e3[3]  = { pts1[9] - c1[0], pts1[10] - c1[1], pts1[11] - c1[2] };

            // 2) Identify shape2 edges similarly
            double c2[3]  = { pts2[0], pts2[1], pts2[2] };
            double f1[3]  = { pts2[3] - c2[0], pts2[4] - c2[1], pts2[5] - c2[2] };
            double f2[3]  = { pts2[6] - c2[0], pts2[7] - c2[1], pts2[8] - c2[2] };
            double f3[3]  = { pts2[9] - c2[0], pts2[10] - c2[1], pts2[11] - c2[2] };

            // 3) Rebuild all 8 corners for shape1 (so we can do standard projection)
            //    shape1[i] = c1 + alpha1 * e1 + alpha2 * e2 + alpha3 * e3,
            //    where alphaN is either 0 or 1. The corner ordering matches get_unit_corners().
            double shape1[24];
            for(int i=0; i<8; i++){
                int a1 = (i & 1) ? 1 : 0; // bit 0
                int a2 = (i & 2) ? 1 : 0; // bit 1
                int a3 = (i & 4) ? 1 : 0; // bit 2
                shape1[3*i + 0] = c1[0] + a1*e1[0] + a2*e2[0] + a3*e3[0];
                shape1[3*i + 1] = c1[1] + a1*e1[1] + a2*e2[1] + a3*e3[1];
                shape1[3*i + 2] = c1[2] + a1*e1[2] + a2*e2[2] + a3*e3[2];
            }

            // 4) Rebuild all 8 corners for shape2
            double shape2[24];
            for(int i=0; i<8; i++){
                int a1 = (i & 1) ? 1 : 0;
                int a2 = (i & 2) ? 1 : 0;
                int a3 = (i & 4) ? 1 : 0;
                shape2[3*i + 0] = c2[0] + a1*f1[0] + a2*f2[0] + a3*f3[0];
                shape2[3*i + 1] = c2[1] + a1*f1[1] + a2*f2[1] + a3*f3[1];
                shape2[3*i + 2] = c2[2] + a1*f1[2] + a2*f2[2] + a3*f3[2];
            }

            // 5) Compute the 15 candidate axes:
            //    -- 3 face normals from shape1
            //    -- 3 face normals from shape2
            //    -- 9 cross products of edges from shape1 x edges from shape2

            // shape1 face normals
            double n1[3], n2[3], n3[3];
            cross3(e1, e2, n1);
            cross3(e2, e3, n2);
            cross3(e3, e1, n3);

            // shape2 face normals
            double m1[3], m2[3], m3[3];
            cross3(f1, f2, m1);
            cross3(f2, f3, m2);
            cross3(f3, f1, m3);

            double edges1[3][3] = {{e1[0], e1[1], e1[2]},
                                {e2[0], e2[1], e2[2]},
                                {e3[0], e3[1], e3[2]}};
            double edges2[3][3] = {{f1[0], f1[1], f1[2]},
                                {f2[0], f2[1], f2[2]},
                                {f3[0], f3[1], f3[2]}};

            // Fill up to 15 axes in an array
            double axes[15][3];
            int axisCount = 0;

            // shape1 face normals
            axes[axisCount][0] = n1[0]; axes[axisCount][1] = n1[1]; axes[axisCount][2] = n1[2]; axisCount++;
            axes[axisCount][0] = n2[0]; axes[axisCount][1] = n2[1]; axes[axisCount][2] = n2[2]; axisCount++;
            axes[axisCount][0] = n3[0]; axes[axisCount][1] = n3[1]; axes[axisCount][2] = n3[2]; axisCount++;

            // shape2 face normals
            axes[axisCount][0] = m1[0]; axes[axisCount][1] = m1[1]; axes[axisCount][2] = m1[2]; axisCount++;
            axes[axisCount][0] = m2[0]; axes[axisCount][1] = m2[1]; axes[axisCount][2] = m2[2]; axisCount++;
            axes[axisCount][0] = m3[0]; axes[axisCount][1] = m3[1]; axes[axisCount][2] = m3[2]; axisCount++;

            // cross products of edges
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    double c12[3];
                    cross3(edges1[i], edges2[j], c12);
                    double len_c12 = norm3(c12);
                    if(len_c12 > eps){  // skip near-degenerate
                        axes[axisCount][0] = c12[0];
                        axes[axisCount][1] = c12[1];
                        axes[axisCount][2] = c12[2];
                        axisCount++;
                    }
                }
            }

            // 6) Run the SAT test
            double proj1[2], proj2[2];
            for(int a=0; a<axisCount; a++){
                project_points(shape1, axes[a], eps, proj1);
                project_points(shape2, axes[a], eps, proj2);
                if(!intervals_overlap(proj1, proj2)){
                    // Found a separating axis -> no intersection
                    return 0;
                }
            }
            // No separating axis found => shapes intersect
            return 1;
        }

        // --------------------------------------------------------------------
        // BATCH function: parallelepipeds_intersect for n parallelepipeds
        // all_pts1: length 24*n (each block of 8 corners = 24 floats)
        // pts2    : just one shape of 8 corners = 24 floats
        // out_intersect[i] = 0 or 1
        // --------------------------------------------------------------------
        int check_parallelepipeds_intersect_batch(const double *all_pts1,
                                                const double *pts2,
                                                double eps,
                                                int n,
                                                int *out_intersect)
        {
            for(int i=0; i<n; i++){
                const double *shape_i = all_pts1 + 24*i; 
                out_intersect[i] = single_intersect(shape_i, pts2, eps);
            }
            return 0; // success
        }
        '''

        # Create CFFI instance
        ffi_obj = FFI()

        # Provide the C prototypes
        ffi_obj.cdef("""int check_parallelepipeds_intersect_batch(const double *all_pts1,const double *pts2,double eps,int n,int *out_intersect);""")

        # Compile and link in-memory
        C_mod = ffi_obj.verify(c_source,extra_compile_args=["-O3"],libraries=[])

        return ffi_obj, C_mod
        
    ## Main Functions
    def get_chunk_positions(self,material):
        '''
        Gets the list of clipped chunk positions in real space the and chunk dimensions in unit cell lengths
        Works for any arbitrary sample dimensions or unit cell.
        Inputs:
            material -> crystal class object
        Outputs:
            chunk_positions_S -> chunk corner positions in the sampke frame
            chunk_dimensions -> chunk dimensions in unit cell lengths
        '''
        lattice_matrix = material.lattice_matrix.T
        lattice_volume = material.lattice_volume
        #lattice_lengths = material.lattice_lengths
        # Get number of lattice units along sample in crystal frame
        lattice_units = np.ceil(np.max(self.corners@np.linalg.inv(lattice_matrix),axis=0)-np.min(self.corners@np.linalg.inv(lattice_matrix),axis=0))
        # Get default chunk size in number of unit cells for each direction.
        # Note: use np.ceil or np.min(([1,1,1],xxx)) to prevent 0 chunk size error, foor is to always make chunk smaller than requested volume
        chunk_dimensions = np.zeros(lattice_units.shape)+np.floor((self.chunk_volume/lattice_volume)**(1/3))
        # Check if any dimensions are smaller than sample for more efficient chunking
        size_check = lattice_units>chunk_dimensions
        if ~np.all(size_check):
            # Adjust size if dimensions are smaller.
            chunk_dimensions[~size_check] = np.min((chunk_dimensions,lattice_units),axis=0)[~size_check]
            chunk_dimensions[size_check] = np.floor(((self.chunk_volume/lattice_volume)/np.prod(chunk_dimensions[~size_check]))**(1/np.sum(size_check)))
            # Try to make dimensions an integer number of unit cells
            chunk_dimensions[size_check] = np.floor(lattice_units[size_check]/np.ceil(lattice_units[size_check]/chunk_dimensions[size_check]))
        chunk_units = np.ceil(lattice_units/chunk_dimensions)
        # Generate positions in the crystal frame
        chunk_positions_C = self.get_flat_grid(chunk_units,gpu=False)*chunk_dimensions # note this is still in crystal coordinates
        # Convert to sample frame
        # Adjust chunk positions to the center of the sample and center of a chunk
        chunk_positions_S = (chunk_positions_C-(lattice_units/2-self.dimensions@np.linalg.inv(lattice_matrix)/2))@lattice_matrix #+ (self.dimensions)/2 #- ((lattice_units+1)@lattice_matrix)/2
        # Trim chunks on cpu
        # Generate corners array
        # Note: Could do "((self.get_unit_corners()*(chunk_dimensions+2)-1)" for padding of 1 unit cell on all sides
        chunk_corners_S = chunk_positions_S[:,np.newaxis,:] + ((self.get_unit_corners()*(chunk_dimensions))@lattice_matrix)[np.newaxis,:,:]
        # Compile and call function
        ffi_object, intersect_function = self.compile_parallelepipeds_intersect_batch_cffi()
        # Using self.get_unit_corners()@self.matrix for sample corner positions to avoid unecessary offset calculations on the CPU
        mask_arr = self.parallelepipeds_intersect_cffi(intersect_function,ffi_object,chunk_corners_S,self.get_unit_corners()@self.matrix,eps=1e-12)
        chunk_positions_S = chunk_positions_S[mask_arr,:]
        return chunk_positions_S, chunk_dimensions #chunk_positions_S, chunk_dimensions
        
    def parallelepipeds_intersect_cffi(self,compiled_code,ffi_object,pts1, pts2, eps=1e-12):
        '''
        Code to check if two parallelepipeds intersect (in this case seeing if a chunk intersects with the sample).
        Inputs:
            compiled_code, ffi_object -> required inputs to call fast C code
            pts1 -> set of n chunk corner points
            pts2 -> set of sample corner points
        Outputs:
            mask_arr -> a mask of which chunks intersect the sample
        '''
        # Convert Python inputs to a contiguous C array of fp32
        # Variable pts1 should be lists of shape (n,8,3), pts2 should be lists of shape (8,3).
        pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
        pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
        n = pts1.shape[0]
        arr_all = pts1.ravel()
        arr2 = pts2.ravel()
        # Bring variables to C
        c_all   = ffi_object.new("double[]", arr_all.tolist())  # 24*n floats
        c_arr2  = ffi_object.new("double[]", arr2.tolist())     # 24 floats
        results_int = np.zeros(n, dtype=np.int32)
        c_out = ffi_object.cast("int *", results_int.ctypes.data)
        # Call the compiled function
        compiled_code.check_parallelepipeds_intersect_batch(c_all,c_arr2,float(eps),n,c_out)
        mask_arr = (results_int == 1)
        return mask_arr
    
    def get_atomic_data(self,material,chunk_position,chunk_dimensions):
        '''
        Gets the location of all lattice points in the sample frame in a given chunk.
        Inputs:
            material -> crystal class object
            chunk_position -> vector of chunk origin position.
            chunk_dimension -> array of number of unit cells per chunk direction.
        Outputs:
            atomic_positions_S -> the position of lattice sites in the sample frame
        '''
        ## Todo: Add species to this.
        lattice_atom_cartesian_cp = cp.asarray(material.lattice_atom_cartesian,dtype=cp.float32)
        lattice_positions_cp = self.get_lattice_positions(material,chunk_position,chunk_dimensions)
        # Cast atomic positions together with lattice positions to get full array of positions
        atomic_positions_S = (lattice_positions_cp[:, cp.newaxis, :] + lattice_atom_cartesian_cp[cp.newaxis, :, :]).reshape(-1, 3)
        atomic_species = np.tile(material.species, lattice_positions_cp.shape[0])
        mask = (((atomic_positions_S[:,0] >= 0) & (atomic_positions_S[:,0] <= self.dimensions[0])) & 
        ((atomic_positions_S[:,1] >= 0) & (atomic_positions_S[:,1] <= self.dimensions[1])) & 
        ((atomic_positions_S[:,2] >= 0) & (atomic_positions_S[:,2] <= self.dimensions[2])))
        atomic_positions_S = atomic_positions_S[mask,:]
        atomic_species = atomic_species[mask.get()]
        # Offset positions
        atomic_positions_S += cp.asarray(self.offset) - cp.asarray(self.dimensions/2)
        return atomic_positions_S,atomic_species

    def get_lattice_positions(self,material,chunk_position,chunk_dimensions):
        '''
        Gets the location of all lattice points in the sample frame in a given chunk.
        Inputs:
            material -> crystal class object
            chunk_position -> vector of chunk origin position.
            chunk_dimension -> array of number of unit cells per chunk direction.
        Outputs:
            lattice_positions_S -> the position of lattice sites in the sample frame
        '''
        lattice_matrix = material.lattice_matrix.T
        # Convert values to GPU
        lattice_matrix_cp = cp.asarray(lattice_matrix,dtype=cp.float32)
        chunk_position_cp = cp.asarray(chunk_position,dtype=cp.float32)
        # Create lattice position grid
        lattice_positions_C = self.get_flat_grid(chunk_dimensions,gpu=True)
        lattice_positions_S = lattice_positions_C@lattice_matrix_cp + chunk_position_cp
        return lattice_positions_S
    
    def generate_sample(self,material): #incomplete
        ## Todo: Add accumulation of chunks (modify chunk total and save large enough chunks on the fly)
        # Also need to output species
        self._chunk_positions, self._chunk_dimensions = self.get_chunk_positions(material)
        self._chunk_total = self.chunk_positions.shape[0] #need to change chunk total once accumulation gets implemented
        for i in range(self.chunk_total):
            atomic_positions,atomic_species = self.get_atomic_data(material,self.chunk_positions[i,:],self.chunk_dimensions)
            self.write_chunk_positions(atomic_positions,i+1)
            self.write_chunk_species(atomic_species,i+1)
        return
    
    def plot_sample(self,elev=0, azim=0):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_proj_type('ortho')
        ax1.axis('equal')
        plt.tight_layout()
        for i in range(self.chunk_total):
                positions_chunk_np = self.load_chunk_positions(i + 1, gpu=False)
                ax1.scatter(positions_chunk_np[:, 0], positions_chunk_np[:, 1], positions_chunk_np[:, 2], c='b',marker='.')
        # plt.show()
        return fig, ax1
        
    ## Properties
    @property
    def matrix(self): #fix for new convention
        """
        Return the 3x3 matrix of vectors (as a NumPy array).
        """
        if self._matrix is None:
            self._matrix = np.diag(self.dimensions)
        return self._matrix

    @property
    def corners(self): #fix for new convention
        """
        Return the lengths of the a, b, c lattice vectors (in Angstroms).
        """
        if self._corners is None:
            self._corners = self.get_unit_corners()@self.matrix - self.dimensions/2 + self.offset
        return self._corners
    
    @property
    def chunk_positions(self):
        """
        Return the 3x3 matrix of vectors (as a NumPy array).
        """
        if self._chunk_positions is None:
            print("self._chunk_positions has not been initialized yet")
        return self._chunk_positions
    
    @property
    def chunk_dimensions(self):
        """
        Return the 3x3 matrix of vectors (as a NumPy array).
        """
        if self._chunk_dimensions is None:
            print("self._chunk_dimensions has not been initialized yet")
        return self._chunk_dimensions
    
    @property
    def chunk_total(self):
        """
        Return the total number of chunks in the sample.
        """
        if self._chunk_total is None:
            print("self._chunk_total has not been initialized yet")
        return self._chunk_total