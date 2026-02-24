# -----------------------------------------------------------------------------
# Diffraction Calculator
# -----------------------------------------------------------------------------
"""
Utility module for calculating diffraction geometry.

Provides calculations for:
- Bragg angles from Miller indices
- Q-vector transformations (crystal -> lab frame)
- Detector angle calculations
- Accessible reflection enumeration
- Stage alignment calculations
"""

import numpy as np
from typing import Optional, Tuple, List, Generator


class DiffractionCalculator:
    """Calculate diffraction geometry from crystal, beam, and stage objects.

    This class provides methods to:
    - Calculate Bragg angles for given Miller indices
    - Transform Q-vectors from crystal to lab frame
    - Determine detector positions for specific reflections
    - Enumerate accessible reflections at current energy
    - Calculate stage rotations for alignment

    Attributes:
        crystal: Crystal object with lattice information
        beam: Beam object with energy/wavelength
        stage: Stage object with rotation information
    """

    def __init__(self, crystal=None, beam=None, stage=None):
        """Initialize the diffraction calculator.

        Args:
            crystal: Crystal object (optional, can be set later)
            beam: Beam object (optional, can be set later)
            stage: Stage object (optional, can be set later)
        """
        self.crystal = crystal
        self.beam = beam
        self.stage = stage

    def set_crystal(self, crystal):
        """Set or update the crystal object."""
        self.crystal = crystal

    def set_beam(self, beam):
        """Set or update the beam object."""
        self.beam = beam

    def set_stage(self, stage):
        """Set or update the stage object."""
        self.stage = stage

    def get_wavelength(self) -> float:
        """Return beam wavelength in Angstroms.

        Returns:
            Wavelength in Angstroms.

        Raises:
            ValueError: If beam is not set or wavelength is not available.
        """
        if self.beam is None:
            raise ValueError("Beam object not set")
        if self.beam._wavelength is None:
            raise ValueError("Beam wavelength not initialized")
        # Convert from meters to Angstroms
        return self.beam._wavelength * 1e10

    def get_energy_eV(self) -> float:
        """Return beam energy in eV.

        Returns:
            Energy in electron volts.

        Raises:
            ValueError: If beam is not set or energy is not available.
        """
        if self.beam is None:
            raise ValueError("Beam object not set")
        if self.beam._energy is None:
            raise ValueError("Beam energy not initialized")
        return self.beam._energy

    def get_d_spacing(self, hkl: Tuple[int, int, int]) -> float:
        """Calculate d-spacing for given Miller indices.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Interplanar spacing in Angstroms.

        Raises:
            ValueError: If crystal is not set.
        """
        if self.crystal is None:
            raise ValueError("Crystal object not set")
        return self.crystal.get_dhkl(list(hkl))

    def get_bragg_angle(self, hkl: Tuple[int, int, int]) -> Optional[float]:
        """Calculate Bragg angle for given Miller indices.

        Uses Bragg's law: 2d*sin(theta) = lambda

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Bragg angle theta in radians, or None if reflection is inaccessible
            (when d < lambda/2, making sin(theta) > 1).
        """
        d = self.get_d_spacing(hkl)
        wavelength = self.get_wavelength()

        sin_theta = wavelength / (2 * d)
        if abs(sin_theta) > 1:
            return None  # Reflection is inaccessible at this energy

        return np.arcsin(sin_theta)

    def get_two_theta(self, hkl: Tuple[int, int, int]) -> Optional[float]:
        """Calculate 2-theta scattering angle for given Miller indices.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            2-theta angle in radians, or None if reflection is inaccessible.
        """
        theta = self.get_bragg_angle(hkl)
        if theta is None:
            return None
        return 2 * theta

    def is_accessible(self, hkl: Tuple[int, int, int]) -> bool:
        """Check if a reflection is accessible at current energy.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            True if the reflection can be measured, False otherwise.
        """
        return self.get_bragg_angle(hkl) is not None

    def _get_reciprocal_lattice_matrix(self) -> np.ndarray:
        """Calculate reciprocal lattice matrix from conventional cell.

        For direct lattice matrix L (with rows as lattice vectors, as used by
        Crystal.py), the reciprocal lattice matrix is B = (L^-1)^T.
        The COLUMNS of B are the reciprocal lattice vectors b1, b2, b3.

        Note: This correctly handles the transformation L' = R @ L used by
        Crystal.py when rotating the crystal. When L' = R @ L, we get
        B' = (L'^-1)^T = (L^-1 @ R^-1)^T = R^-T @ (L^-1)^T = R @ B
        (for orthogonal R), so Q' = R @ Q as expected.

        Returns:
            3x3 reciprocal lattice matrix with columns as reciprocal vectors.
        """
        if self.crystal is None:
            raise ValueError("Crystal object not set")

        L = self.crystal.lattice_matrix_conventional
        return np.linalg.inv(L).T

    def get_q_vector_crystal(self, hkl: Tuple[int, int, int]) -> np.ndarray:
        """Calculate Q-vector in crystal reference frame (sample frame).

        The Q-vector (scattering vector) is G = h*b1 + k*b2 + l*b3
        where bi are the reciprocal lattice vectors (columns of B matrix).

        This can be computed as Q = B @ hkl where B = (L^-1)^T.

        Note: Since Crystal.py rotates the lattice as L' = R @ L, the
        resulting Q is already in the sample reference frame (after any
        crystal rotations from align_axes or rotate_crystal).

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Q-vector in sample frame (Angstrom^-1).
        """
        B = self._get_reciprocal_lattice_matrix()
        hkl_vec = np.array(hkl, dtype=float)
        return B @ hkl_vec

    def get_q_vector_lab(self, hkl: Tuple[int, int, int]) -> np.ndarray:
        """Calculate Q-vector transformed to laboratory frame.

        Applies stage rotation to transform the Q-vector from sample
        coordinates to lab coordinates.

        Note: The crystal's lattice_matrix_conventional is already rotated
        by _cumulative_rotation, so Q_crystal computed from it is already
        in the sample frame. We only need to apply the stage rotation.

        Convention note: Stage.get_rotation() returns R designed for row-vector
        convention (pos_lab = pos_sample @ R, used in Beam.py). For column-vector
        Q-vectors, we need: Q_lab = R.T @ Q_sample.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Q-vector in lab frame (Angstrom^-1).
        """
        # Q_crystal is computed from lattice_matrix_conventional which already
        # incorporates the crystal rotation (_cumulative_rotation).
        # Therefore Q_crystal is already in the sample frame.
        Q_sample = self.get_q_vector_crystal(hkl)

        # Apply stage rotation to transform from sample frame to lab frame
        # Stage.get_rotation() is designed for row-vectors: pos_lab = pos_sample @ R
        # For column-vector Q: Q_lab = R.T @ Q_sample
        R_stage = np.eye(3)
        if self.stage is not None:
            R_stage = self.stage.get_rotation()

        return R_stage.T @ Q_sample

    def get_detector_angles(self, hkl: Tuple[int, int, int]) -> Optional[Tuple[float, float]]:
        """Calculate detector angles to capture a specific reflection.

        Determines the (2theta, eta) position for the detector to
        intercept the diffracted beam from the specified reflection.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Tuple of (two_theta, eta) in radians, or None if inaccessible.
            - two_theta: scattering angle (angle from incident beam)
            - eta: azimuthal angle in y-z plane
        """
        theta = self.get_bragg_angle(hkl)
        if theta is None:
            return None

        two_theta = 2 * theta

        # Get Q-vector direction in lab frame
        Q_lab = self.get_q_vector_lab(hkl)
        Q_norm = Q_lab / np.linalg.norm(Q_lab)

        # eta is the azimuthal angle of Q in the y-z plane
        # When Q points along +z, eta = 0; along +y, eta = pi/2
        eta = np.arctan2(Q_norm[1], Q_norm[2])

        return (two_theta, eta)

    def enumerate_accessible_reflections(
        self,
        max_h: int = 5,
        max_k: int = 5,
        max_l: int = 5,
        unique_only: bool = True
    ) -> Generator[Tuple[int, int, int], None, None]:
        """Enumerate all reflections accessible at current energy.

        Args:
            max_h: Maximum |h| index to consider.
            max_k: Maximum |k| index to consider.
            max_l: Maximum |l| index to consider.
            unique_only: If True, only yield positive (h,k,l) to avoid
                        Friedel pairs (h,k,l) and (-h,-k,-l).

        Yields:
            Tuples of accessible Miller indices (h, k, l).
        """
        for h in range(-max_h, max_h + 1):
            for k in range(-max_k, max_k + 1):
                for l in range(-max_l, max_l + 1):
                    if h == k == l == 0:
                        continue

                    # Skip Friedel pairs if unique_only
                    if unique_only:
                        # Only yield if this is the "positive" representative
                        # (first non-zero index is positive)
                        first_nonzero = next((x for x in (h, k, l) if x != 0), 0)
                        if first_nonzero < 0:
                            continue

                    if self.is_accessible((h, k, l)):
                        yield (h, k, l)

    def get_structure_factor(self, hkl: Tuple[int, int, int]) -> complex:
        """Calculate kinematic structure factor for given Miller indices.

        Uses the formula: F(hkl) = Σ_j f_j * exp(2πi * (h*x_j + k*y_j + l*z_j))
        where f_j is approximated by atomic number (for rough intensity estimates)
        and (x_j, y_j, z_j) are fractional coordinates.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Complex structure factor F(hkl).

        Raises:
            ValueError: If crystal is not set.
        """
        if self.crystal is None:
            raise ValueError("Crystal object not set")

        h, k, l = hkl

        # Get fractional coordinates and species from CONVENTIONAL cell
        # (Miller indices are defined in the conventional cell basis)
        frac_coords = None
        species = None

        # Prefer conventional cell data if available
        if hasattr(self.crystal, '_lattice_atom_fractional_conventional') and \
           self.crystal._lattice_atom_fractional_conventional is not None:
            frac_coords = self.crystal._lattice_atom_fractional_conventional
            species = getattr(self.crystal, '_species_conventional', None)
        elif hasattr(self.crystal, '_lattice_atom_fractional') and \
             self.crystal._lattice_atom_fractional is not None:
            # Fallback to primitive cell data (may give incorrect results)
            frac_coords = self.crystal._lattice_atom_fractional
            species = getattr(self.crystal, '_species', None)
        else:
            raise ValueError("Crystal fractional coordinates not available")

        # Atomic number lookup for common elements (as f_j approximation)
        # This is a rough approximation - real calculations use tabulated
        # atomic scattering factors that depend on sin(θ)/λ
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Pb': 82,
            'Bi': 83, 'U': 92,
        }

        F = 0.0 + 0.0j
        for i, (x, y, z) in enumerate(frac_coords):
            # Get atomic scattering factor (approximated by Z)
            if species is not None and i < len(species):
                element = str(species[i])
                # Strip any charge notation (e.g., "Fe2+" -> "Fe")
                element = ''.join(c for c in element if c.isalpha())
                f_j = atomic_numbers.get(element, 10)  # Default to 10 if unknown
            else:
                f_j = 10  # Default

            # Phase factor: exp(2πi * (h*x + k*y + l*z))
            phase = 2 * np.pi * (h * x + k * y + l * z)
            F += f_j * np.exp(1j * phase)

        return F

    def get_structure_factor_magnitude(self, hkl: Tuple[int, int, int]) -> float:
        """Calculate |F(hkl)| - the magnitude of the structure factor.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Magnitude of structure factor.
        """
        F = self.get_structure_factor(hkl)
        return np.abs(F)

    def get_reflection_info(self, hkl: Tuple[int, int, int]) -> dict:
        """Get comprehensive information about a reflection.

        Args:
            hkl: Tuple of Miller indices (h, k, l).

        Returns:
            Dictionary containing:
            - hkl: Miller indices
            - d_spacing: Interplanar spacing (Angstroms)
            - accessible: Whether reflection is accessible
            - bragg_angle: Bragg angle theta (radians), or None
            - two_theta: 2-theta angle (radians), or None
            - bragg_angle_deg: Bragg angle (degrees), or None
            - two_theta_deg: 2-theta angle (degrees), or None
            - q_magnitude: |Q| (Angstrom^-1)
            - q_lab: Q-vector in lab frame
            - detector_angles: (two_theta, eta) or None
            - structure_factor: |F(hkl)| magnitude
        """
        d = self.get_d_spacing(hkl)
        theta = self.get_bragg_angle(hkl)
        accessible = theta is not None

        info = {
            'hkl': hkl,
            'd_spacing': d,
            'accessible': accessible,
            'bragg_angle': theta,
            'two_theta': 2 * theta if theta else None,
            'bragg_angle_deg': np.degrees(theta) if theta else None,
            'two_theta_deg': np.degrees(2 * theta) if theta else None,
        }

        # Q-vector information
        Q_crystal = self.get_q_vector_crystal(hkl)
        info['q_magnitude'] = np.linalg.norm(Q_crystal)

        # Structure factor
        try:
            info['structure_factor'] = self.get_structure_factor_magnitude(hkl)
        except Exception:
            info['structure_factor'] = None

        if accessible:
            info['q_lab'] = self.get_q_vector_lab(hkl)
            det_angles = self.get_detector_angles(hkl)
            info['detector_angles'] = det_angles
            if det_angles:
                info['detector_two_theta_deg'] = np.degrees(det_angles[0])
                info['detector_eta_deg'] = np.degrees(det_angles[1])
        else:
            info['q_lab'] = None
            info['detector_angles'] = None

        return info

    def calculate_alignment_motor_values(
        self,
        hkl: Tuple[int, int, int],
        target_eta: float = 0.0
    ) -> Optional[dict]:
        """Calculate stage motor values to align a reflection to Bragg condition.

        This method determines the stage rotation needed to bring the
        specified (h,k,l) reflection into the Bragg condition with
        scattering into the desired azimuthal direction.

        Args:
            hkl: Target Miller indices (h, k, l).
            target_eta: Desired azimuthal angle for scattering (radians).
                       eta=0 is vertical (XZ plane), eta=90° is horizontal (XY plane).

        Returns:
            Dictionary with motor values, or None if alignment is impossible.
            Contains:
            - two_theta: Required detector 2-theta (radians)
            - eta: Required detector eta (radians)
            - rotation_matrix: Rotation to apply to stage
            - motor_values: Estimated motor positions (if stage has standard motors)
        """
        theta = self.get_bragg_angle(hkl)
        if theta is None:
            return None  # Cannot align inaccessible reflection

        two_theta = 2 * theta

        # Get Q-vector in sample frame.
        # Note: get_q_vector_crystal() uses lattice_matrix_conventional which
        # already incorporates the crystal's _cumulative_rotation, so the
        # returned vector is already in the sample frame.
        Q_sample = self.get_q_vector_crystal(hkl)
        Q_sample_norm = Q_sample / np.linalg.norm(Q_sample)

        # Target Q direction for Bragg condition in lab frame
        # Q_target = (-sin(θ), cos(θ)*sin(η), cos(θ)*cos(η))
        Q_target_norm = np.array([
            -np.sin(theta),
            np.cos(theta) * np.sin(target_eta),
            np.cos(theta) * np.cos(target_eta),
        ])

        # Directly compute Euler angles (phi, chi, omega) that align Q_sample with Q_target
        # Stage rotation: R = R_Z(-ω) @ R_X(-χ) @ R_Y(-φ)
        motor_values = self._compute_euler_angles_for_alignment(
            Q_sample_norm, Q_target_norm
        )

        if motor_values is None:
            return None

        # Reconstruct the rotation matrix from motor values
        phi_rad = np.radians(motor_values['phi'])
        chi_rad = np.radians(motor_values['chi'])
        omega_rad = np.radians(motor_values['omega'])
        rotation_matrix = self._build_stage_rotation(phi_rad, chi_rad, omega_rad)

        result = {
            'two_theta': two_theta,
            'eta': target_eta,
            'two_theta_deg': np.degrees(two_theta),
            'eta_deg': np.degrees(target_eta),
            'rotation_matrix': rotation_matrix,
            'motor_values': motor_values,
        }

        return result

    def _build_stage_rotation(
        self, phi: float, chi: float, omega: float
    ) -> np.ndarray:
        """Build stage rotation matrix from motor angles.

        This properly accounts for the kinematic chain coupling used in the
        standard 4-circle goniometer where omega's axis is transformed by
        phi and chi rotations.

        Motor coupling (with mu=0):
        - phi: axis [0, -1, 0] (fixed)
        - chi: axis [-1, 0, 0] (fixed, only coupled to mu)
        - omega: axis transformed by phi and chi rotations

        Args:
            phi: Phi angle in radians
            chi: Chi angle in radians
            omega: Omega angle in radians

        Returns:
            3x3 rotation matrix matching Stage.get_rotation()
        """
        # Fixed axes for phi and chi (with mu=0)
        phi_axis = np.array([0.0, -1.0, 0.0])
        chi_axis = np.array([-1.0, 0.0, 0.0])
        omega_axis_raw = np.array([0.0, 0.0, -1.0])

        # Build R_phi: rotation around phi_axis by phi
        R_phi = self._axis_angle_rotation(phi_axis, phi)

        # Build R_chi: rotation around chi_axis by chi
        R_chi = self._axis_angle_rotation(chi_axis, chi)

        # Omega's axis is transformed by phi and chi (coupling [0, 1, 2] with mu=0)
        # omega_axis_transformed = R_chi @ R_phi @ omega_axis_raw
        omega_axis_transformed = R_chi @ R_phi @ omega_axis_raw

        # Build R_omega: rotation around transformed omega axis by omega
        R_omega = self._axis_angle_rotation(omega_axis_transformed, omega)

        # Total rotation: apply phi first, then chi, then omega
        # R_total = R_omega @ R_chi @ R_phi
        return R_omega @ R_chi @ R_phi

    def _axis_angle_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Compute rotation matrix for rotation around an axis by an angle.

        Uses Rodrigues' rotation formula, matching Stage.get_axis_rotation().

        Args:
            axis: Rotation axis (will be normalized)
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        d = 1.0 - c
        x, y, z = axis

        return np.array([
            [c + d*x*x,     d*x*y - z*s,   d*x*z + y*s],
            [d*y*x + z*s,   c + d*y*y,     d*y*z - x*s],
            [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]
        ])

    def _compute_euler_angles_for_alignment(
        self,
        Q_sample: np.ndarray,
        Q_target: np.ndarray
    ) -> Optional[dict]:
        """Compute Euler angles to align Q_sample with Q_target direction.

        Directly solves for (phi, chi, omega) such that:
        R_Z(-ω) @ R_X(-χ) @ R_Y(-φ) @ Q_sample is parallel to Q_target

        Uses numerical optimization to find the angles.

        Args:
            Q_sample: Normalized Q-vector in sample frame
            Q_target: Normalized target Q direction in lab frame

        Returns:
            Dictionary with phi, chi, omega in degrees, or None if failed.
        """
        Qs = Q_sample / np.linalg.norm(Q_sample)
        Qt = Q_target / np.linalg.norm(Q_target)

        def cost_function(angles):
            """Cost = 1 - cos(angle between rotated Q and target)."""
            phi, chi, omega = angles
            R = self._build_stage_rotation(phi, chi, omega)
            # Use R.T because _build_stage_rotation returns row-vector convention matrix
            # (matching Stage.get_rotation()), but we use column-vector Q
            Q_lab = R.T @ Qs
            Q_lab_norm = Q_lab / np.linalg.norm(Q_lab)
            # Return 1 - dot product (0 when aligned, 2 when opposite)
            return 1.0 - np.dot(Q_lab_norm, Qt)

        def gradient(angles, eps=1e-6):
            """Numerical gradient of cost function."""
            grad = np.zeros(3)
            f0 = cost_function(angles)
            for i in range(3):
                angles_plus = angles.copy()
                angles_plus[i] += eps
                grad[i] = (cost_function(angles_plus) - f0) / eps
            return grad

        # Simple gradient descent with multiple starting points
        best_cost = np.inf
        best_angles = None

        # First, try the analytical solution for Q along Z-axis
        # If Q_sample ≈ [0, 0, 1], then a pure phi rotation works
        if abs(Qs[0]) < 0.01 and abs(Qs[1]) < 0.01 and abs(Qs[2]) > 0.99:
            # Q is along Z, compute phi directly
            # Q_target = [-sin(θ), cos(θ)*sin(η), cos(θ)*cos(η)]
            # For phi rotation around [0,-1,0]: R.T @ [0,0,1] = [sin(phi), 0, cos(phi)]
            # To match Q_target[0] = -sin(θ): sin(phi) = -sin(θ) => phi = -θ
            # Since Qt[0] = -sin(θ), we have phi = arcsin(Qt[0])
            analytical_phi = np.arcsin(Qt[0])  # Qt[0] = -sin(θ) => phi = -θ
            test_angles = np.array([analytical_phi, 0.0, 0.0])
            test_cost = cost_function(test_angles)
            if test_cost < 1e-8:
                # Analytical solution works!
                return {
                    'mu': 0.0,
                    'phi': np.degrees(analytical_phi),
                    'chi': 0.0,
                    'omega': 0.0,
                }

        # Try multiple starting points to find global minimum
        start_points = [
            [0, 0, 0],
            [np.pi/4, 0, 0],
            [-np.pi/4, 0, 0],
            [np.pi/2, 0, 0],
            [-np.pi/2, 0, 0],
            [0, np.pi/4, 0],
            [0, -np.pi/4, 0],
            [np.pi/4, np.pi/4, 0],
            [-np.pi/4, np.pi/4, 0],
            [np.pi/4, -np.pi/4, 0],
            [-np.pi/4, -np.pi/4, 0],
        ]

        for start in start_points:
            angles = np.array(start, dtype=np.float64)

            # Gradient descent
            learning_rate = 0.5
            for iteration in range(200):
                cost = cost_function(angles)
                if cost < 1e-10:  # Converged
                    break

                grad = gradient(angles)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break

                # Line search with backtracking
                step = learning_rate
                for _ in range(10):
                    new_angles = angles - step * grad
                    new_cost = cost_function(new_angles)
                    if new_cost < cost:
                        angles = new_angles
                        break
                    step *= 0.5
                else:
                    # No improvement found
                    break

            final_cost = cost_function(angles)
            if final_cost < best_cost:
                best_cost = final_cost
                best_angles = angles.copy()

        if best_angles is None or best_cost > 0.01:
            # Failed to find good alignment
            return None

        # Normalize angles to [-180, 180] range
        phi = np.degrees(best_angles[0])
        chi = np.degrees(best_angles[1])
        omega = np.degrees(best_angles[2])

        # Wrap to [-180, 180]
        phi = ((phi + 180) % 360) - 180
        chi = ((chi + 180) % 360) - 180
        omega = ((omega + 180) % 360) - 180

        return {
            'mu': 0.0,  # Explicitly set mu to 0
            'phi': phi,
            'chi': chi,
            'omega': omega,
        }

    def _rotation_between_vectors(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> np.ndarray:
        """Calculate rotation matrix to rotate v1 to align with v2.

        Uses Rodrigues' formula for rotation about the axis
        perpendicular to both vectors.

        Args:
            v1: Source unit vector.
            v2: Target unit vector.

        Returns:
            3x3 rotation matrix R such that R @ v1 is parallel to v2.
        """
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        cos_theta = np.dot(v1, v2)

        # Handle nearly parallel vectors
        if np.abs(cos_theta - 1.0) < 1e-10:
            return np.eye(3)

        # Handle nearly antiparallel vectors
        if np.abs(cos_theta + 1.0) < 1e-10:
            # Find orthogonal axis
            if abs(v1[0]) < 0.9:
                axis = np.cross(v1, np.array([1, 0, 0]))
            else:
                axis = np.cross(v1, np.array([0, 1, 0]))
            axis = axis / np.linalg.norm(axis)
            return self._rodrigues_rotation(axis, np.pi)

        # General case
        axis = np.cross(v1, v2)
        sin_theta = np.linalg.norm(axis)
        axis = axis / sin_theta
        angle = np.arctan2(sin_theta, cos_theta)

        return self._rodrigues_rotation(axis, angle)

    def _rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Compute rotation matrix using Rodrigues' formula.

        Args:
            axis: Unit rotation axis.
            angle: Rotation angle in radians.

        Returns:
            3x3 rotation matrix.
        """
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        d = 1.0 - c
        x, y, z = axis

        return np.array([
            [c + d*x*x,     d*x*y - z*s,   d*x*z + y*s],
            [d*y*x + z*s,   c + d*y*y,     d*y*z - x*s],
            [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]
        ], dtype=np.float64)

    def _estimate_motor_values(self, rotation_matrix: np.ndarray) -> Optional[dict]:
        """Estimate motor values from a rotation matrix.

        Decomposes the rotation into Euler angles matching the stage's
        motor configuration from Stage.py:
        - phi: rotation around -Y axis [0, -1, 0]
        - chi: rotation around -X axis [-1, 0, 0]
        - omega: rotation around -Z axis [0, 0, -1]

        The stage applies: R = R_omega @ R_chi @ R_phi
        Where each motor rotates around its NEGATIVE axis:
        - R_phi(φ) = R_Y(-φ) (rotation around [0,-1,0])
        - R_chi(χ) = R_X(-χ) (rotation around [-1,0,0])
        - R_omega(ω) = R_Z(-ω) (rotation around [0,0,-1])

        So: R = R_Z(-ω) @ R_X(-χ) @ R_Y(-φ)

        Matrix elements derived from this product:
        R[0,0] = cos(ω)cos(φ) - sin(ω)sin(χ)sin(φ)
        R[0,1] = sin(ω)cos(χ)
        R[0,2] = cos(ω)sin(φ) + sin(ω)sin(χ)cos(φ)
        R[1,0] = -sin(ω)cos(φ) - cos(ω)sin(χ)sin(φ)
        R[1,1] = cos(ω)cos(χ)
        R[1,2] = -sin(ω)sin(φ) + cos(ω)sin(χ)cos(φ)
        R[2,0] = -cos(χ)sin(φ)
        R[2,1] = -sin(χ)
        R[2,2] = cos(χ)cos(φ)

        Args:
            rotation_matrix: Target rotation matrix.

        Returns:
            Dictionary of motor name -> angle (degrees), or None.
        """
        try:
            R = np.asarray(rotation_matrix)

            # Validate rotation matrix shape
            if R.ndim != 2 or R.shape[0] != 3 or R.shape[1] != 3:
                return None

            # Extract chi from R[2,1] = -sin(chi)
            r21 = float(R[2, 1])
            r21 = np.clip(r21, -1.0, 1.0)  # Clamp for numerical stability

            # Check for gimbal lock (chi near ±90°)
            if abs(r21) < 0.9999:
                # chi = -arcsin(R[2,1]) since R[2,1] = -sin(chi)
                chi = -np.arcsin(r21)
                cos_chi = np.cos(chi)

                # Extract phi from R[2,0] and R[2,2]
                # R[2,0] = -cos(chi)*sin(phi)
                # R[2,2] = cos(chi)*cos(phi)
                # tan(phi) = -R[2,0] / R[2,2] = sin(phi) / cos(phi)
                phi = np.arctan2(-float(R[2, 0]), float(R[2, 2]))

                # Extract omega from R[0,1] and R[1,1]
                # R[0,1] = sin(omega)*cos(chi)
                # R[1,1] = cos(omega)*cos(chi)
                # tan(omega) = R[0,1] / R[1,1]
                omega = np.arctan2(float(R[0, 1]), float(R[1, 1]))
            else:
                # Gimbal lock case (chi = ±90°)
                # When chi = ±90°, cos(chi) ≈ 0, and the rotation becomes:
                # R = R_Z(-ω) @ R_X(∓90°) @ R_Y(-φ)
                # This couples phi and omega into a single effective rotation
                phi = 0.0  # Set phi to 0 arbitrarily
                if r21 < 0:  # R[2,1] = -sin(chi), so r21 < 0 means chi > 0
                    chi = np.pi / 2
                    # At chi = 90°: R becomes rotation around combined axis
                    # R[0,0] = cos(ω-φ), R[1,0] = -sin(ω-φ)
                    omega = np.arctan2(-float(R[1, 0]), float(R[0, 0]))
                else:  # r21 > 0 means chi < 0
                    chi = -np.pi / 2
                    # At chi = -90°: R[0,0] = cos(ω+φ), R[1,0] = sin(ω+φ)
                    omega = np.arctan2(float(R[1, 0]), float(R[0, 0]))

            return {
                'phi': np.degrees(phi),
                'chi': np.degrees(chi),
                'omega': np.degrees(omega),
            }
        except Exception:
            return None

    def get_q_positions_for_display(
        self,
        max_h: int = 3,
        max_k: int = 3,
        max_l: int = 3,
        scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Q-vector positions for 3D display.

        Returns arrays of positions suitable for rendering as markers
        in the 3D viewport.

        Args:
            max_h, max_k, max_l: Maximum Miller indices.
            scale: Scale factor for visualization (larger = smaller display).

        Returns:
            Tuple of (positions, colors, sizes) where:
            - positions: Nx3 array of Q positions (scaled)
            - colors: Nx4 array of RGBA colors
            - accessible: N-length boolean array
        """
        positions = []
        colors = []
        accessible_list = []

        GREEN = np.array([0.2, 0.8, 0.2, 1.0])  # Accessible
        RED = np.array([0.8, 0.2, 0.2, 0.5])    # Inaccessible

        for h in range(-max_h, max_h + 1):
            for k in range(-max_k, max_k + 1):
                for l in range(-max_l, max_l + 1):
                    if h == k == l == 0:
                        continue

                    hkl = (h, k, l)
                    Q_lab = self.get_q_vector_lab(hkl)

                    # Scale Q for visualization
                    pos = Q_lab / scale
                    positions.append(pos)

                    accessible = self.is_accessible(hkl)
                    accessible_list.append(accessible)
                    colors.append(GREEN if accessible else RED)

        if not positions:
            return np.zeros((0, 3)), np.zeros((0, 4)), np.zeros(0, dtype=bool)

        return (
            np.array(positions),
            np.array(colors),
            np.array(accessible_list)
        )
