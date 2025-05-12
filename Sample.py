# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import os
import gc
import json
try:
    import cupy as cp
except ImportError:
    cp = None
from cffi import FFI

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
        self._dimensions = None
        self._offset = None
        self._rotation = None
        self._chunk_volume = None
        self._chunk_total = None 
        self._matrix = None
        self._corners = None
        self._default_filenames = np.array([
            "atomic_positions.npy",
            "atomic_species.npy",
            "sample_metadata.npy"
        ])  # sample_metadata will be a struct
        self._ffi_object, self._intersect_function = self.compile_parallelepipeds_intersect_batch_cffi()
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
    def create_sample(self, dimensions, offset=[0,0,0], chunk_volume=(600*600*600)):
        self._dimensions = np.array(dimensions, dtype=np.float32)
        self._offset = np.array(offset, dtype=np.float32)
        self._rotation = np.eye(3, dtype=np.float32)
        self._chunk_volume = np.array(chunk_volume, dtype=np.float32)
        self._matrix = np.diag(self.dimensions)
        # Slightly rewritten for small overhead reduction (no functional change)
        self._corners = (self.get_unit_corners() @ self.matrix) - (self.dimensions * 0.5) + self.offset
        
    def read_sample_metadata(self):
        """
        Reads the metadata JSON file from disk and restores
        this sample object's state.
        """
        metadata_filename = os.path.join(self.directory, "sample_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            sample_metadata = json.load(f)

        # Convert lists back to NumPy arrays
        if sample_metadata["dimensions"] is not None:
            self._dimensions = np.array(sample_metadata["dimensions"], dtype=np.float32)
        if sample_metadata["offset"] is not None:
            self._offset = np.array(sample_metadata["offset"], dtype=np.float32)
        if sample_metadata["rotation"] is not None:
            self._rotation = np.array(sample_metadata["rotation"], dtype=np.float32)
        if sample_metadata["chunk_total"] is not None:
            self._chunk_total = int(sample_metadata["chunk_total"])
        
    ## Data Handling Functions
    def write_chunk_positions(self, data, chunk_num, override_directory=None):
        base, ext = os.path.splitext(self._default_filenames[0])
        chunk_filename = f"{base}_{chunk_num}{ext}"
        if override_directory is not None:
            np.save(os.path.join(override_directory, chunk_filename), data)
        else:
            np.save(os.path.join(self.directory, chunk_filename), data)
    
    def write_chunk_species(self, data, chunk_num, override_directory=None):
        base, ext = os.path.splitext(self._default_filenames[1])
        chunk_filename = f"{base}_{chunk_num}{ext}"
        if override_directory is not None:
            np.save(os.path.join(override_directory, chunk_filename), data)
        else:
            np.save(os.path.join(self.directory, chunk_filename), data)
            
    def write_sample_metadata(self, override_directory=None):
        """
        Serializes the sample object's critical internal fields to disk 
        as human-readable JSON so that the state can be restored later.
        """
        # Convert NumPy arrays to Python lists so JSON can handle them
        sample_metadata = {
            "dimensions": self._dimensions.tolist() if self._dimensions is not None else None,
            "offset": self._offset.tolist() if self._offset is not None else None,
            "rotation": self._rotation.tolist() if self._rotation is not None else None,
            "chunk_total": int(self._chunk_total) if self._chunk_total is not None else None,
        }

        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "sample_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "sample_metadata.json")

        # Write as nicely formatted JSON
        with open(metadata_filename, "w") as f:
            json.dump(sample_metadata, f, indent=4)
        print(f"Metadata written to {metadata_filename} in JSON format.")

    def load_chunk_positions(self, chunk_number, use_gpu=True):
        """
        Load positions from disk. If use_gpu=True and cupy is available, return a cp.ndarray.
        Otherwise, return an np.ndarray.
        """
        base, ext = os.path.splitext(self._default_filenames[0])
        positions_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, positions_filename)
        if use_gpu and (cp is not None):
            return cp.load(full_path)
        else:
            return np.load(full_path)

    def load_chunk_species(self, chunk_number, use_gpu=True):
        """
        Load species from disk. If use_gpu=True and cupy is available, return a cp.ndarray.
        Otherwise, return an np.ndarray.
        """
        base, ext = os.path.splitext(self._default_filenames[1])
        species_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, species_filename)
        if use_gpu and (cp is not None):
            return cp.load(full_path)
        else:
            return np.load(full_path)
        
    def import_atomic_data(self, import_file, element_list, header_lines=9, ID_column=1, position_columns=[2,3,4], scale=1e-10, flush_size=100000000, override_directory=None):
        """
        Reads the atoms from a large text file, skipping the first 9 lines, and
        chunks them into binary .npy files of size flush_size in the desired folder.
        
        The atomic positions are assumed to be in columns 3,4,5 of each line (1-based indexing).
        Also recovers 'dimensions', 'offset', and 'chunk_total' from the bounding box
        of these atomic positions.
        """
        chunk_num = 0
        # Track min/max in x,y,z to calculate dimensions and offset afterward
        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')
        with open(import_file, "r") as f:
            # Skip the first 9 lines
            for _ in range(header_lines):
                next(f)
            while True:
                # Read up to flush_size lines at a time
                lines = []
                for _ in range(flush_size):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                # If no lines were read, we're at EOF
                if not lines:
                    break
                # Parse positions from columns 3,4,5
                data_arr = np.zeros((len(lines), 3), dtype=np.float32)
                species_arr = []
                for i, line in enumerate(lines):
                    split_line = line.strip().split()
                    species_arr.append(element_list[int(split_line[ID_column])-1])
                    data_arr[i, 0] = float(split_line[position_columns[0]])*float(scale/1e-10)
                    data_arr[i, 1] = float(split_line[position_columns[1]])*float(scale/1e-10)
                    data_arr[i, 2] = float(split_line[position_columns[2]])*float(scale/1e-10)
                    # Update bounding box
                    if data_arr[i, 0] < x_min: x_min = data_arr[i, 0]
                    if data_arr[i, 0] > x_max: x_max = data_arr[i, 0]
                    if data_arr[i, 1] < y_min: y_min = data_arr[i, 1]
                    if data_arr[i, 1] > y_max: y_max = data_arr[i, 1]
                    if data_arr[i, 2] < z_min: z_min = data_arr[i, 2]
                    if data_arr[i, 2] > z_max: z_max = data_arr[i, 2]
                # Increment chunk number and save the positions
                chunk_num += 1
                self.write_chunk_positions(data_arr, chunk_num, override_directory=override_directory)
                self.write_chunk_species(species_arr, chunk_num, override_directory=override_directory)
        # Record how many chunks were created
        self._chunk_total = chunk_num
        # Infer dimensions from bounding box
        self._dimensions = np.array([x_max - x_min, 
                                    y_max - y_min, 
                                    z_max - z_min], dtype=np.float32)
        # Offset is the midpoint of the bounding box (center)
        self._offset = np.array([(x_min + x_max) / 2.0,
                                (y_min + y_max) / 2.0,
                                (z_min + z_max) / 2.0], dtype=np.float32)
        self._rotation = np.eye(3)

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
            [1, 1, 1]], dtype=np.float32)
        return unit_corners
    
    @staticmethod
    def get_rotation(axis,angle):
        """
        Return the 3x3 rotation matrix for rotation by 'angle' radians
        around the (normalized) 'axis'.
        """
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        d = 1.0 - c
        x, y, z = axis
        return np.array([[c + d*x*x,     d*x*y - z*s,   d*x*z + y*s],
                         [d*y*x + z*s,   c + d*y*y,     d*y*z - x*s],
                         [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]])
    
    @staticmethod
    def get_flat_grid(dimensions, use_gpu=False):
        """
        Create a 3D grid of integer coordinates. If use_gpu=True and cupy is available,
        use CuPy arrays; otherwise use NumPy arrays.
        """
        if use_gpu and (cp is not None):
            # GPU path
            ii, jj, kk = cp.meshgrid(
                cp.arange(dimensions[0], dtype=cp.float32),
                cp.arange(dimensions[1], dtype=cp.float32),
                cp.arange(dimensions[2], dtype=cp.float32),
                indexing='ij'
            )
            flat_grid_cp = cp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
            return flat_grid_cp
        else:
            # CPU path
            # Force float32 for CPU so it matches GPU's single precision
            dims_np = np.array(dimensions, dtype=np.float32)
            ii, jj, kk = np.meshgrid(
                np.arange(dims_np[0], dtype=np.float32),
                np.arange(dims_np[1], dtype=np.float32),
                np.arange(dims_np[2], dtype=np.float32),
                indexing='ij'
            )
            flat_grid = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
            return flat_grid
    
    @staticmethod    
    def compile_parallelepipeds_intersect_batch_cffi():
        '''
        C++ code using 15-axis SAT method for determining if a set of cornerpoints intersects
        with another, made to run a batch operation of corner points against a single reference.
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

            // 3) Rebuild all 8 corners for shape1
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
        int check_parallelepipeds_intersect_batch(
            const double *all_pts1,
            const double *pts2,
            double eps,
            int n,
            int *out_intersect
        )
        {
            for(int i=0; i<n; i++){
                const double *shape_i = all_pts1 + 24*i; 
                out_intersect[i] = single_intersect(shape_i, pts2, eps);
            }
            return 0; // success
        }
        '''
        ffi_obj = FFI()
        ffi_obj.cdef("""int check_parallelepipeds_intersect_batch(
            const double *all_pts1,
            const double *pts2,
            double eps,
            int n,
            int *out_intersect);
        """)
        C_mod = ffi_obj.verify(c_source, extra_compile_args=["-O3"], libraries=[])
        return ffi_obj, C_mod
        
    ## Main Functions
    def get_chunk_positions(self, material):
        '''
        Gets the list of clipped chunk positions in real space the and chunk dimensions in unit cell lengths
        Works for any arbitrary sample dimensions or unit cell.
        Inputs:
            material -> crystal class object
        Outputs:
            chunk_positions_S -> chunk corner positions in the sample frame
            chunk_dimensions -> chunk dimensions in unit cell lengths
        '''
        lattice_matrix = material.lattice_matrix.T
        lattice_volume = material.lattice_volume
        
        # Precompute for performance
        inv_lattice_matrix = np.linalg.inv(lattice_matrix)
        corners_in_lattice = self.corners @ inv_lattice_matrix
        
        # Get number of lattice units along sample in crystal frame
        lattice_units = np.ceil(np.max(corners_in_lattice, axis=0) - np.min(corners_in_lattice, axis=0))
        
        # Get default chunk size in number of unit cells for each direction.
        chunk_dimensions = np.zeros(lattice_units.shape) + np.floor((self.chunk_volume / lattice_volume)**(1/3))
        
        # Check if any dimensions are smaller than sample for more efficient chunking
        size_check = lattice_units > chunk_dimensions
        if not np.all(size_check):
            chunk_dimensions[~size_check] = np.min((chunk_dimensions, lattice_units), axis=0)[~size_check]
            chunk_dimensions[size_check] = np.floor(
                ((self.chunk_volume/lattice_volume) / np.prod(chunk_dimensions[~size_check])) ** 
                (1/np.sum(size_check))
            )
            chunk_dimensions[size_check] = np.floor(lattice_units[size_check] / np.ceil(lattice_units[size_check] / chunk_dimensions[size_check]))
        
        chunk_units = np.ceil(lattice_units / chunk_dimensions)
        
        # Generate positions in the crystal frame (CPU by default)
        chunk_positions_C = self.get_flat_grid(chunk_units, use_gpu=False) * chunk_dimensions
        
        # Convert to sample frame, adjusting positions to center
        adj_val = (lattice_units * 0.5) - (self.dimensions @ inv_lattice_matrix * 0.5)
        chunk_positions_S = (chunk_positions_C - adj_val) @ lattice_matrix
        
        # Generate corners array
        chunk_corners_S = chunk_positions_S[:, np.newaxis, :] + ((self.get_unit_corners() * chunk_dimensions) @ lattice_matrix)[np.newaxis, :, :]
        
        # Using self.get_unit_corners() @ self.matrix for sample corner positions
        mask_arr = self.parallelepipeds_intersect_cffi(
            self._intersect_function,
            self._ffi_object,
            chunk_corners_S,
            (self.get_unit_corners() @ self.matrix),
            eps=1e-12
        )
        chunk_positions_S = chunk_positions_S[mask_arr, :]
        return chunk_positions_S, chunk_dimensions
        
    def parallelepipeds_intersect_cffi(self, compiled_code, ffi_object, pts1, pts2, eps=1e-12):
        '''
        Code to check if two parallelepipeds intersect (in this case seeing if
        a chunk intersects with the sample).
        
        Inputs:
            compiled_code, ffi_object -> required inputs to call fast C code
            pts1 -> set of n chunk corner points
            pts2 -> set of sample corner points
        Outputs:
            mask_arr -> a mask of which chunks intersect the sample
        '''
        pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
        pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
        n = pts1.shape[0]
        
        arr_all = pts1.ravel().tolist()  # cffi needs a Python list
        arr2 = pts2.ravel().tolist()
        
        c_all   = ffi_object.new("double[]", arr_all)
        c_arr2  = ffi_object.new("double[]", arr2)
        results_int = np.zeros(n, dtype=np.int32)
        c_out = ffi_object.cast("int *", results_int.ctypes.data)
        
        compiled_code.check_parallelepipeds_intersect_batch(c_all, c_arr2, float(eps), n, c_out)
        mask_arr = (results_int == 1)
        return mask_arr

    def get_lattice_positions(self, material, chunk_position, chunk_dimensions, use_gpu=True):
        '''
        Gets the location of lattice points in the sample frame in a given chunk.
        
        If use_gpu=True and cupy is installed, returns a cp.ndarray.
        Otherwise returns a np.ndarray.
        '''
        lattice_matrix = material.lattice_matrix.T

        if use_gpu and (cp is not None):
            # GPU path
            lattice_matrix_cp = cp.asarray(lattice_matrix, dtype=cp.float32)
            chunk_position_cp = cp.asarray(chunk_position, dtype=cp.float32)

            lattice_positions_C = self.get_flat_grid(chunk_dimensions, use_gpu=True)
            lattice_positions_S = lattice_positions_C @ lattice_matrix_cp + chunk_position_cp
            return lattice_positions_S

        else:
            # CPU path
            # Ensure single-precision on CPU
            lattice_matrix_np = lattice_matrix.astype(np.float32)
            chunk_position_np = np.array(chunk_position, dtype=np.float32)

            lattice_positions_C = self.get_flat_grid(chunk_dimensions, use_gpu=False)
            lattice_positions_S = lattice_positions_C @ lattice_matrix_np + chunk_position_np
            return lattice_positions_S

    def get_atomic_data(self, material, chunk_position, chunk_dimensions, use_gpu=True):
        '''
        Gets the location of all lattice points in the sample frame in a given chunk,
        plus the species. Returns (positions, species).
        
        - If use_gpu=True and cupy is available, positions will be a cp.ndarray
          (until masking finishes, then we bring them partially back).
        - If use_gpu=False or cupy is unavailable, positions will be an np.ndarray.
        '''
        # If we have a GPU available and user wants GPU, do it on GPU
        use_gpu = (use_gpu and (cp is not None))

        if use_gpu:
            # GPU branch
            lattice_atom_cartesian_cp = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
            lattice_positions_cp = self.get_lattice_positions(material, chunk_position, chunk_dimensions, use_gpu=True)
            
            atomic_positions_S = (
                lattice_positions_cp[:, cp.newaxis, :] + 
                lattice_atom_cartesian_cp[cp.newaxis, :, :]
            ).reshape(-1, 3)
            
            atomic_species = np.tile(material.species, lattice_positions_cp.shape[0])
            mask = (
                (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= self.dimensions[0]) &
                (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= self.dimensions[1]) &
                (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= self.dimensions[2])
            )
            mask_np = mask.get()  # bring mask back to CPU
            
            atomic_positions_S = atomic_positions_S[mask, :]  # still cp array
            atomic_species = atomic_species[mask_np]
            
            offset_gpu = cp.array(self.offset, dtype=cp.float32)
            dim_half_gpu = cp.array(self.dimensions * 0.5, dtype=cp.float32)
            atomic_positions_S += (offset_gpu - dim_half_gpu)

            # Return final positions to CPU
            atomic_positions_S = atomic_positions_S.get()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            return atomic_positions_S, atomic_species

        else:
            # CPU branch
            # Convert all relevant data to float32 to match GPU path
            lattice_atom_cartesian_np = material.lattice_atom_cartesian.astype(np.float32)
            lattice_positions_np = self.get_lattice_positions(material, chunk_position, chunk_dimensions, use_gpu=False)
            
            atomic_positions_S = (
                lattice_positions_np[:, np.newaxis, :].astype(np.float32) +
                lattice_atom_cartesian_np[np.newaxis, :, :]
            ).reshape(-1, 3)
            
            atomic_species = np.tile(material.species, lattice_positions_np.shape[0])
            # Mask
            mask = (
                (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= self.dimensions[0]) &
                (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= self.dimensions[1]) &
                (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= self.dimensions[2])
            )
            atomic_positions_S = atomic_positions_S[mask, :].astype(np.float32)
            atomic_species = atomic_species[mask]

            # Offset in float32
            offset_np = self.offset.astype(np.float32)
            dim_half_np = (self.dimensions * 0.5).astype(np.float32)
            atomic_positions_S += (offset_np - dim_half_np)
            gc.collect()
            return atomic_positions_S, atomic_species

    def generate_sample(self, material, flush_size=100000000, use_gpu=True):
        """
        Accumulates the atomic positions/species from each geometric chunk.
        Each written chunk will contain exactly `flush_size` atoms, except
        for the last chunk if there are fewer than `flush_size` atoms left.

        The `gpu` parameter controls whether to use GPU acceleration (if available)
        or force CPU-only. 
        """
        # 1) Determine the geometric chunk positions
        self._chunk_positions, self._chunk_dimensions = self.get_chunk_positions(material)
        self._chunk_total = self.chunk_positions.shape[0]
        
        # 2) Prepare accumulators (lists) in CPU memory
        acc_positions = []
        acc_species = []
        
        # We'll use this to name each *written* chunk
        file_chunk_index = 0
        # Keep track of total atoms in accumulator
        total_accumulated = 0
        
        # 3) Loop over all geometric chunks
        use_gpu = (use_gpu and (cp is not None))
        for i in range(self.chunk_total):
            # -- a) Get atomic data
            atomic_positions, atomic_species = self.get_atomic_data(
                material,
                self.chunk_positions[i, :],
                self._chunk_dimensions,
                use_gpu=use_gpu
            )
            
            # -- b) If this chunk alone is bigger than flush_size, split immediately
            if atomic_positions.shape[0] >= flush_size:
                start_idx = 0
                while start_idx < atomic_positions.shape[0]:
                    end_idx = start_idx + flush_size
                    chunk_positions = atomic_positions[start_idx:end_idx]
                    chunk_species   = atomic_species[start_idx:end_idx]

                    file_chunk_index += 1
                    self.write_chunk_positions(chunk_positions, file_chunk_index)
                    self.write_chunk_species(chunk_species, file_chunk_index)

                    start_idx = end_idx
                # Move on to next geometric chunk
                continue
            
            # Otherwise, accumulate
            acc_positions.append(atomic_positions)
            acc_species.append(atomic_species)
            total_accumulated += atomic_positions.shape[0]
            
            # -- c) While total atoms >= flush_size, write out exactly flush_size
            while total_accumulated >= flush_size:
                cat_positions = np.concatenate(acc_positions, axis=0)
                cat_species   = np.concatenate(acc_species,   axis=0)

                chunk_positions = cat_positions[:flush_size]
                chunk_species   = cat_species[:flush_size]

                file_chunk_index += 1
                self.write_chunk_positions(chunk_positions, file_chunk_index)
                self.write_chunk_species(chunk_species, file_chunk_index)

                leftover_positions = cat_positions[flush_size:]
                leftover_species   = cat_species[flush_size:]

                acc_positions = [leftover_positions] if leftover_positions.size > 0 else []
                acc_species   = [leftover_species] if leftover_species.size > 0 else []
                total_accumulated = leftover_positions.shape[0] if leftover_positions.size > 0 else 0
        
        # 4) After processing all geometric chunks, check leftover
        leftover_atoms = total_accumulated
        if leftover_atoms > 0:
            cat_positions = np.concatenate(acc_positions, axis=0)
            cat_species   = np.concatenate(acc_species, axis=0)

            file_chunk_index += 1
            self.write_chunk_positions(cat_positions, file_chunk_index)
            self.write_chunk_species(cat_species, file_chunk_index)
        
        self._chunk_total = file_chunk_index
        return
    
    def zero_sample_position(self, use_gpu=True):
        """
        Re-loads each chunk of atomic positions, subtracts the current self.offset
        from every position (centering them), and writes them back out.
        Finally sets self.offset to [0,0,0].
        """
        if self._offset is None:
            raise ValueError("Offset is not initialized. Please set self._offset or load metadata first.")

        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate sample or import atoms first.")
        
        offset_np = self.offset.astype(np.float32)

        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                positions_chunk -= cp.array(offset_np)
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk -= offset_np
                self.write_chunk_positions(positions_chunk, i + 1)

        self._offset = np.zeros(3, dtype=np.float32)
        print("All atomic positions re-centered. Offset is now [0, 0, 0].")
        
    def zero_sample_rotation(self, use_gpu=True):
        """
        Re-loads each chunk of atomic positions, rotates all chunks by the inverse
        of the current self._rotation, and writes them back out.
        Finally sets self._rotation to the 3x3 identity matrix.
        """
        if self._rotation is None:
            raise ValueError("No sample rotation matrix is set. Please initialize or load it first.")
        
        R_inv = self._rotation.T.astype(np.float32)
        
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                R_inv_cp = cp.asarray(R_inv)
                positions_chunk = positions_chunk @ R_inv_cp
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk = positions_chunk @ R_inv
                self.write_chunk_positions(positions_chunk, i + 1)
        
        self._rotation = np.eye(3, dtype=np.float32)
        print("All atomic positions de-rotated. Sample rotation is now the identity matrix.")
        
    def zero_sample(self, use_gpu=True):
        self.zero_sample_position(use_gpu=use_gpu)
        self.zero_sample_rotation(use_gpu=use_gpu)
        
    def rotate_sample_relative(self, axis, dangle, degrees=True, use_gpu=True):
        """
        Re-loads each chunk of atomic positions, rotates it according to self.get_rotation(axis, dangle),
        writes them back out, and then updates self._rotation by left-multiplying with the new rotation.
        """
        if degrees:
            dangle = np.deg2rad(dangle)
        
        R = self.get_rotation(axis, dangle).astype(np.float32)
        
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                R_cp = cp.asarray(R)
                positions_chunk = positions_chunk @ R_cp
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk = positions_chunk @ R
                self.write_chunk_positions(positions_chunk, i + 1)
        
        self._rotation = R @ self._rotation
        print(f"Sample rotated by {dangle:.4f} radians about axis {axis}. "
              f"Updated sample rotation matrix:\n{self._rotation}")

    def translate_sample_relative(self, offset_vector, use_gpu=True): # update this to use dx, dy, dz
        """
        Re-loads each chunk of atomic positions, adds the offset_vector to every position,
        and writes them back out.
        Finally adds offset_vector to self._offset.
        """
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        offset_np = np.array(offset_vector, dtype=np.float32)
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                positions_chunk += cp.asarray(offset_np)
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk += offset_np
                self.write_chunk_positions(positions_chunk, i + 1)
        
        if self._offset is None:
            self._offset = offset_np
        else:
            self._offset += offset_np
        
        print(f"Sample translated by {offset_vector}. New offset is {self._offset}.")

    def plot_sample(self, elev=0, azim=0):
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
            positions_chunk_np = self.load_chunk_positions(i + 1, use_gpu=False)
            ax1.scatter(
                positions_chunk_np[:, 0],
                positions_chunk_np[:, 1],
                positions_chunk_np[:, 2],
                c='b', marker='.'
            )
        return fig, ax1
    
    ## Properties
    @property
    def dimensions(self):
        """
        Return the dimensions array (length 3).
        """
        if self._dimensions is None:
            print("self._dimensions has not been initialized yet")
        return self._dimensions

    @property
    def offset(self):
        """
        Return the offset array (length 3).
        """
        if self._offset is None:
            print("self._offset has not been initialized yet")
        return self._offset
    
    @property
    def rotation(self):
        """
        Return the rotation matrix (3x3).
        """
        if self._rotation is None:
            print("self._rotation has not been initialized yet")
        return self._rotation

    @property
    def chunk_volume(self):
        """
        Return the chunk volume.
        """
        if self._chunk_volume is None:
            print("self._chunk_volume has not been initialized yet")
        return self._chunk_volume

    @property
    def matrix(self):
        """
        Return the sample matrix (3x3).
        """
        if self._matrix is None:
            self._matrix = np.diag(self.dimensions)
        return self._matrix

    @property
    def corners(self):
        """
        Return the corners of the sample parallelepiped (8x3).
        """
        if self._corners is None:
            self._corners = (self.get_unit_corners() @ self.matrix) - (self.dimensions * 0.5) + self.offset
        return self._corners
    
    @property
    def chunk_positions(self):
        """
        Return the array of chunk positions (Nx3).
        """
        if self._chunk_positions is None:
            print("self._chunk_positions has not been initialized yet")
        return self._chunk_positions
    
    @property
    def chunk_dimensions(self):
        """
        Return the chunk dimensions (in lattice units).
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
