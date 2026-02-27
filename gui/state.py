# -----------------------------------------------------------------------------
# Simulation State Management
# -----------------------------------------------------------------------------
"""
Central state management for the X-ray simulator GUI using the observer pattern.

The SimulationState class holds references to all simulation objects and notifies
registered observers when objects are created, modified, or deleted.
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with aliases to avoid naming conflicts with properties
try:
    from Crystal import crystal as CrystalType
    from Sample import sample as SampleType
    from Beam import beam as BeamType
    from Detector import detector as DetectorType
    from Stage import stage as StageType
    from Optics import optics as OpticsType
    from Defects import defects as DefectsType
    from Deformation import deformation as DeformationType
    from Experiment import experiment as ExperimentType
    from Analysis import analysis as AnalysisType
except ImportError as e:
    # Allow GUI to run without all modules (for development/testing)
    print(f"Warning: Could not import simulation modules: {e}")
    CrystalType = Any
    SampleType = Any
    BeamType = Any
    DetectorType = Any
    StageType = Any
    OpticsType = Any
    DefectsType = Any
    DeformationType = Any
    ExperimentType = Any
    AnalysisType = Any


@dataclass
class ProjectMetadata:
    """Metadata for a simulation project."""
    name: str = "Untitled"
    description: str = ""
    author: str = ""
    created: str = ""
    modified: str = ""
    file_path: Optional[str] = None


class SimulationState:
    """
    Central state management with observer pattern for GUI synchronization.

    This class manages all simulation objects and provides:
    - Observer registration for state change notifications
    - Object creation/modification/deletion with automatic notifications
    - Project save/load functionality
    - State serialization for presets

    Attributes:
        crystal: Crystal structure object
        sample: Sample object with atomic positions
        beam: X-ray beam configuration
        detector: Detector object
        stage: Goniometer/stage object
        optics: Optical components stack
        defects: Defect definitions
        deformation: Deformation field
        experiment: Experiment orchestration
        analysis: Analysis tools
    """

    # Event types for observer notifications
    EVENTS = [
        "crystal_changed",
        "sample_changed",
        "beam_changed",
        "detector_changed",
        "stage_changed",
        "optics_changed",
        "defects_changed",
        "deformation_changed",
        "experiment_changed",
        "analysis_changed",
        "project_changed",
        "simulation_started",
        "simulation_finished",
        "simulation_progress",
        "error_occurred",
        "global_working_directory_changed",
    ]

    def __init__(self):
        """Initialize the simulation state with default values."""
        # Simulation objects
        self._crystal: Optional[CrystalType] = None
        self._sample: Optional[SampleType] = None
        self._beam: Optional[BeamType] = None
        self._detector: Optional[DetectorType] = None
        self._stage: Optional[StageType] = None
        self._optics: Optional[OpticsType] = None
        self._defects: Optional[DefectsType] = None
        self._deformation: Optional[DeformationType] = None
        self._experiment: Optional[ExperimentType] = None
        self._analysis: Optional[AnalysisType] = None

        # Project metadata
        self._metadata = ProjectMetadata()

        # Observer callbacks: event_name -> list of callbacks
        self._observers: Dict[str, List[Callable]] = defaultdict(list)

        # Dirty flag for unsaved changes
        self._dirty = False

        # Current working directory for file operations
        self._working_directory = os.getcwd()

        # Global working directory (default for all directory options)
        self._global_working_directory = ""

    # -------------------------------------------------------------------------
    # Observer Pattern Implementation
    # -------------------------------------------------------------------------

    def register_observer(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a specific event.

        Args:
            event: Event name (must be in EVENTS list)
            callback: Function to call when event occurs
        """
        if event not in self.EVENTS:
            raise ValueError(f"Unknown event: {event}. Valid events: {self.EVENTS}")
        self._observers[event].append(callback)

    def unregister_observer(self, event: str, callback: Callable) -> None:
        """
        Unregister a callback for a specific event.

        Args:
            event: Event name
            callback: Function to remove
        """
        if callback in self._observers[event]:
            self._observers[event].remove(callback)

    def _notify(self, event: str, data: Any = None) -> None:
        """
        Notify all observers of an event.

        Args:
            event: Event name
            data: Optional data to pass to callbacks
        """
        for callback in self._observers[event]:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in observer callback for {event}: {e}")

    def notify_observers(self, event: str, data: Any = None) -> None:
        """
        Public method to notify observers of an event.

        Args:
            event: Event name (e.g., "detector_changed", "preset_loaded")
            data: Optional data to pass to callbacks
        """
        # Add event to list if not already present (for custom events)
        if event not in self.EVENTS:
            self.EVENTS.append(event)
        self._notify(event, data)

    # -------------------------------------------------------------------------
    # Property Accessors with Change Notifications
    # -------------------------------------------------------------------------

    @property
    def crystal(self) -> Optional[CrystalType]:
        """Get the crystal object."""
        return self._crystal

    @crystal.setter
    def crystal(self, value: Optional[CrystalType]) -> None:
        """Set the crystal object and notify observers."""
        self._crystal = value
        self._dirty = True
        self._notify("crystal_changed", value)

    @property
    def sample(self) -> Optional[SampleType]:
        """Get the sample object."""
        return self._sample

    @sample.setter
    def sample(self, value: Optional[SampleType]) -> None:
        """Set the sample object and notify observers."""
        self._sample = value
        self._dirty = True
        self._notify("sample_changed", value)

    @property
    def beam(self) -> Optional[BeamType]:
        """Get the beam object."""
        return self._beam

    @beam.setter
    def beam(self, value: Optional[BeamType]) -> None:
        """Set the beam object and notify observers."""
        self._beam = value
        self._dirty = True
        self._notify("beam_changed", value)

    @property
    def detector(self) -> Optional[DetectorType]:
        """Get the detector object."""
        return self._detector

    @detector.setter
    def detector(self, value: Optional[DetectorType]) -> None:
        """Set the detector object and notify observers."""
        self._detector = value
        self._dirty = True
        self._notify("detector_changed", value)

    @property
    def stage(self) -> Optional[StageType]:
        """Get the stage object."""
        return self._stage

    @stage.setter
    def stage(self, value: Optional[StageType]) -> None:
        """Set the stage object and notify observers."""
        self._stage = value
        self._dirty = True
        self._notify("stage_changed", value)

    @property
    def optics(self) -> Optional[OpticsType]:
        """Get the optics object."""
        return self._optics

    @optics.setter
    def optics(self, value: Optional[OpticsType]) -> None:
        """Set the optics object and notify observers."""
        self._optics = value
        self._dirty = True
        self._notify("optics_changed", value)

    @property
    def defects(self) -> Optional[DefectsType]:
        """Get the defects object."""
        return self._defects

    @defects.setter
    def defects(self, value: Optional[DefectsType]) -> None:
        """Set the defects object and notify observers."""
        self._defects = value
        self._dirty = True
        self._notify("defects_changed", value)

    @property
    def deformation(self) -> Optional[DeformationType]:
        """Get the deformation object."""
        return self._deformation

    @deformation.setter
    def deformation(self, value: Optional[DeformationType]) -> None:
        """Set the deformation object and notify observers."""
        self._deformation = value
        self._dirty = True
        self._notify("deformation_changed", value)

    @property
    def experiment(self) -> Optional[ExperimentType]:
        """Get the experiment object."""
        return self._experiment

    @experiment.setter
    def experiment(self, value: Optional[ExperimentType]) -> None:
        """Set the experiment object and notify observers."""
        self._experiment = value
        self._dirty = True
        self._notify("experiment_changed", value)

    @property
    def analysis(self) -> Optional[AnalysisType]:
        """Get the analysis object."""
        return self._analysis

    @analysis.setter
    def analysis(self, value: Optional[AnalysisType]) -> None:
        """Set the analysis object and notify observers."""
        self._analysis = value
        self._dirty = True
        self._notify("analysis_changed", value)

    @property
    def metadata(self) -> ProjectMetadata:
        """Get the project metadata."""
        return self._metadata

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty

    @property
    def working_directory(self) -> str:
        """Get the current working directory."""
        return self._working_directory

    @working_directory.setter
    def working_directory(self, value: str) -> None:
        """Set the working directory."""
        self._working_directory = value

    @property
    def global_working_directory(self) -> str:
        """Get the global working directory (default for all directory options)."""
        return self._global_working_directory

    @global_working_directory.setter
    def global_working_directory(self, value: str) -> None:
        """Set the global working directory and notify observers."""
        self._global_working_directory = value
        self._notify("global_working_directory_changed", value)

    def get_default_directory(self) -> str:
        """
        Get the default directory to use for file operations.

        Returns the global working directory if set, otherwise the current
        working directory.

        Returns:
            Directory path string
        """
        if self._global_working_directory:
            return self._global_working_directory
        return self._working_directory

    # -------------------------------------------------------------------------
    # Convenience Methods for Object Creation
    # -------------------------------------------------------------------------

    def create_crystal(self, cif_path: str, **kwargs) -> CrystalType:
        """
        Create a new crystal from a CIF file.

        Args:
            cif_path: Path to CIF file
            **kwargs: Additional arguments for crystal constructor

        Returns:
            The created crystal object
        """
        self.crystal = CrystalType(cif_path, **kwargs)
        return self.crystal

    def create_sample(self, dimensions: tuple, **kwargs) -> SampleType:
        """
        Create a new sample.

        Args:
            dimensions: Sample dimensions (x, y, z) in Angstroms
            **kwargs: Additional arguments for sample constructor

        Returns:
            The created sample object
        """
        self.sample = SampleType(dimensions, **kwargs)
        return self.sample

    def create_beam(self, energy: float, **kwargs) -> BeamType:
        """
        Create a new beam.

        Args:
            energy: Beam energy in eV
            **kwargs: Additional arguments for beam constructor

        Returns:
            The created beam object
        """
        self.beam = BeamType()
        self.beam.create_beam(energy=energy, **kwargs)
        return self.beam

    def create_detector(self, shape: tuple, pixel_size: tuple, **kwargs) -> DetectorType:
        """
        Create a new detector.

        Args:
            shape: Detector shape (Ny, Nz) in pixels — Ny = width, Nz = height
            pixel_size: Pixel size (dy, dz) — dy = width spacing, dz = height spacing
            **kwargs: Additional arguments for detector constructor

        Returns:
            The created detector object
        """
        self.detector = DetectorType()
        self.detector.create_detector(shape=shape, pixel_size=pixel_size, **kwargs)
        return self.detector

    def create_stage(self, **kwargs) -> StageType:
        """
        Create a new stage.

        Args:
            **kwargs: Arguments for stage constructor

        Returns:
            The created stage object
        """
        self.stage = StageType()
        self.stage.create_stage(**kwargs)
        return self.stage

    def create_optics(self) -> OpticsType:
        """
        Create a new optics stack.

        Returns:
            The created optics object
        """
        self.optics = OpticsType()
        return self.optics

    def create_defects(self) -> DefectsType:
        """
        Create a new defects object.

        Returns:
            The created defects object
        """
        self.defects = DefectsType()
        return self.defects

    def create_deformation(self) -> DeformationType:
        """
        Create a new deformation object.

        Returns:
            The created deformation object
        """
        self.deformation = DeformationType()
        return self.deformation

    def create_experiment(self, **kwargs) -> ExperimentType:
        """
        Create a new experiment.

        Args:
            **kwargs: Arguments for experiment constructor

        Returns:
            The created experiment object
        """
        self.experiment = ExperimentType(**kwargs)
        return self.experiment

    def create_analysis(self, directory: str = None) -> AnalysisType:
        """
        Create a new analysis object.

        Args:
            directory: Output directory for analysis results

        Returns:
            The created analysis object
        """
        if directory is None:
            directory = self._working_directory
        self.analysis = AnalysisType(directory=directory)
        return self.analysis

    # -------------------------------------------------------------------------
    # Notification Helpers
    # -------------------------------------------------------------------------

    def notify_object_modified(self, object_name: str) -> None:
        """
        Manually notify observers that an object has been modified.

        Use this when modifying object attributes directly without
        reassigning the object itself.

        Args:
            object_name: Name of the object (e.g., "crystal", "sample")
        """
        event = f"{object_name}_changed"
        if event in self.EVENTS:
            self._dirty = True
            obj = getattr(self, f"_{object_name}", None)
            self._notify(event, obj)

    def notify_simulation_started(self) -> None:
        """Notify observers that a simulation has started."""
        self._notify("simulation_started")

    def notify_simulation_finished(self, result: Any = None) -> None:
        """Notify observers that a simulation has finished."""
        self._notify("simulation_finished", result)

    def notify_simulation_progress(self, progress: float, message: str = "") -> None:
        """
        Notify observers of simulation progress.

        Args:
            progress: Progress value (0.0 to 1.0)
            message: Optional status message
        """
        self._notify("simulation_progress", {"progress": progress, "message": message})

    def notify_error(self, error: Exception) -> None:
        """
        Notify observers that an error occurred.

        Args:
            error: The exception that occurred
        """
        self._notify("error_occurred", error)

    # -------------------------------------------------------------------------
    # Project Save/Load
    # -------------------------------------------------------------------------

    def new_project(self) -> None:
        """Reset the state for a new project."""
        self._crystal = None
        self._sample = None
        self._beam = None
        self._detector = None
        self._stage = None
        self._optics = None
        self._defects = None
        self._deformation = None
        self._experiment = None
        self._analysis = None
        self._metadata = ProjectMetadata()
        self._dirty = False
        self._notify("project_changed", None)

    def mark_clean(self) -> None:
        """Mark the project as saved (no unsaved changes)."""
        self._dirty = False

    def to_dict(self) -> Dict:
        """
        Serialize the simulation state to a dictionary.

        Returns:
            Dictionary representation of the state
        """
        from datetime import datetime

        state_dict = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "metadata": {
                "name": self._metadata.name,
                "description": self._metadata.description,
                "author": self._metadata.author,
                "file_path": self._metadata.file_path,
            },
            "global_working_directory": self._global_working_directory,
            "crystal": self._serialize_crystal(),
            "sample": self._serialize_sample(),
            "beam": self._serialize_beam(),
            "detector": self._serialize_detector(),
            "stage": self._serialize_stage(),
            "optics": self._serialize_optics(),
        }
        return state_dict

    def _serialize_crystal(self) -> Optional[Dict]:
        """Serialize crystal parameters."""
        if self._crystal is None:
            return None

        # Get cumulative rotation as list
        cumulative_rotation = None
        if hasattr(self._crystal, '_cumulative_rotation') and self._crystal._cumulative_rotation is not None:
            cumulative_rotation = self._crystal._cumulative_rotation.tolist()

        # Get lattice matrix
        lattice_matrix = None
        if hasattr(self._crystal, '_lattice_matrix_conventional') and self._crystal._lattice_matrix_conventional is not None:
            lattice_matrix = self._crystal._lattice_matrix_conventional.tolist()

        return {
            "cif_path": getattr(self._crystal, "filepath", None),
            "cumulative_rotation": cumulative_rotation,
            "lattice_matrix_conventional": lattice_matrix,
        }

    def _serialize_sample(self) -> Optional[Dict]:
        """Serialize sample parameters."""
        if self._sample is None:
            return None

        dimensions = None
        if hasattr(self._sample, '_dimensions') and self._sample._dimensions is not None:
            dimensions = self._sample._dimensions.tolist()

        offset = None
        if hasattr(self._sample, '_offset') and self._sample._offset is not None:
            offset = self._sample._offset.tolist()

        return {
            "directory": getattr(self._sample, 'directory', None),
            "dimensions": dimensions,
            "offset": offset,
            "sample_type": getattr(self._sample, '_sample_type', "single"),
            "chunk_total": getattr(self._sample, '_chunk_total', None),
        }

    def _serialize_beam(self) -> Optional[Dict]:
        """Serialize beam parameters."""
        if self._beam is None:
            return None

        beam_size = None
        if hasattr(self._beam, '_beam_size') and self._beam._beam_size is not None:
            beam_size = list(self._beam._beam_size) if hasattr(self._beam._beam_size, '__iter__') else self._beam._beam_size

        beam_samples = None
        if hasattr(self._beam, '_beam_samples') and self._beam._beam_samples is not None:
            beam_samples = list(self._beam._beam_samples) if hasattr(self._beam._beam_samples, '__iter__') else self._beam._beam_samples

        return {
            "directory": getattr(self._beam, 'directory', None),
            "energy": getattr(self._beam, '_energy', None),
            "wavelength": getattr(self._beam, '_wavelength', None),
            "beam_shape": getattr(self._beam, '_beam_shape', "rectangular"),
            "beam_size": beam_size,
            "beam_samples": beam_samples,
            "beam_profile": getattr(self._beam, '_beam_profile', "uniform"),
            "pol_perp_rate": getattr(self._beam, '_pol_perp_rate', 0.5),
        }

    def _serialize_detector(self) -> Optional[Dict]:
        """Serialize detector parameters."""
        if self._detector is None:
            return None

        shape = None
        if hasattr(self._detector, '_shape') and self._detector._shape is not None:
            shape = list(self._detector._shape) if hasattr(self._detector._shape, '__iter__') else self._detector._shape

        pixel_size = None
        if hasattr(self._detector, '_pixel_size') and self._detector._pixel_size is not None:
            pixel_size = self._detector._pixel_size.tolist() if hasattr(self._detector._pixel_size, 'tolist') else list(self._detector._pixel_size)

        center = None
        if hasattr(self._detector, '_center') and self._detector._center is not None:
            center = self._detector._center.tolist() if hasattr(self._detector._center, 'tolist') else list(self._detector._center)

        # Handle angular_range serialization
        angular_range = getattr(self._detector, '_angular_range', None)
        if angular_range is not None:
            angular_range = list(angular_range)

        return {
            "directory": getattr(self._detector, 'directory', None),
            "shape": shape,
            "pixel_size": pixel_size,
            "center": center,
            "distance": getattr(self._detector, '_distance', None),
            "two_theta": getattr(self._detector, '_two_theta', None),
            "eta": getattr(self._detector, '_eta', None),
            "geometry": getattr(self._detector, '_geometry', "rectangular"),
            "construction_mode": getattr(self._detector, '_construction_mode', "plane"),
            "input_mode": getattr(self._detector, '_input_mode', "spatial"),
            "angular_range": angular_range,
        }

    def _serialize_stage(self) -> Optional[Dict]:
        """Serialize stage parameters."""
        if self._stage is None:
            return None

        motor_name = None
        if hasattr(self._stage, '_motor_name') and self._stage._motor_name is not None:
            motor_name = self._stage._motor_name.tolist() if hasattr(self._stage._motor_name, 'tolist') else list(self._stage._motor_name)

        motor_type = None
        if hasattr(self._stage, '_motor_type') and self._stage._motor_type is not None:
            motor_type = self._stage._motor_type.tolist() if hasattr(self._stage._motor_type, 'tolist') else list(self._stage._motor_type)

        motor_value = None
        if hasattr(self._stage, '_motor_value') and self._stage._motor_value is not None:
            motor_value = self._stage._motor_value.tolist() if hasattr(self._stage._motor_value, 'tolist') else list(self._stage._motor_value)

        motor_axis = None
        if hasattr(self._stage, '_motor_axis') and self._stage._motor_axis is not None:
            motor_axis = self._stage._motor_axis.tolist() if hasattr(self._stage._motor_axis, 'tolist') else list(self._stage._motor_axis)

        rotation = None
        if hasattr(self._stage, '_rotation') and self._stage._rotation is not None:
            rotation = self._stage._rotation.tolist() if hasattr(self._stage._rotation, 'tolist') else list(self._stage._rotation)

        translation = None
        if hasattr(self._stage, '_translation') and self._stage._translation is not None:
            translation = self._stage._translation.tolist() if hasattr(self._stage._translation, 'tolist') else list(self._stage._translation)

        return {
            "directory": getattr(self._stage, 'directory', None),
            "name": getattr(self._stage, '_name', None),
            "motor_name": motor_name,
            "motor_type": motor_type,
            "motor_value": motor_value,
            "motor_axis": motor_axis,
            "motor_coupling": getattr(self._stage, '_motor_coupling', None),
            "rotation": rotation,
            "translation": translation,
        }

    def _serialize_optics(self) -> Optional[Dict]:
        """Serialize optics parameters."""
        if self._optics is None:
            return None

        stack = []
        if hasattr(self._optics, '_stack') and self._optics._stack is not None:
            for component in self._optics._stack:
                if isinstance(component, dict):
                    stack.append(component)
                else:
                    # Try to extract component info
                    stack.append({"type": str(type(component).__name__)})

        return {
            "directory": getattr(self._optics, 'directory', None),
            "stack": stack,
        }

    def save_to_file(self, filepath: str, save_object_data: bool = True) -> None:
        """
        Save the simulation state to a file.

        Args:
            filepath: Path to save the file
            save_object_data: If True, also save object metadata files to their directories
        """
        # Save object metadata files first (if they have directories)
        if save_object_data:
            self._save_all_object_metadata()

        state_dict = self.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, default=str)

        self._metadata.file_path = filepath
        self._dirty = False

    def _save_all_object_metadata(self) -> None:
        """Save metadata files for all simulation objects."""
        # Save sample metadata
        if self._sample is not None:
            try:
                if hasattr(self._sample, 'write_sample_metadata'):
                    self._sample.write_sample_metadata()
            except Exception as e:
                print(f"Warning: Could not save sample metadata: {e}")

        # Save beam metadata
        if self._beam is not None:
            try:
                if hasattr(self._beam, 'write_beam_metadata'):
                    self._beam.write_beam_metadata()
            except Exception as e:
                print(f"Warning: Could not save beam metadata: {e}")

        # Save detector metadata and pixel data
        if self._detector is not None:
            try:
                if hasattr(self._detector, 'write_detector_metadata'):
                    self._detector.write_detector_metadata()
                # Also save pixel values if present
                self._save_detector_pixels()
            except Exception as e:
                print(f"Warning: Could not save detector metadata: {e}")

        # Save stage metadata
        if self._stage is not None:
            try:
                if hasattr(self._stage, 'write_stage_metadata'):
                    self._stage.write_stage_metadata()
            except Exception as e:
                print(f"Warning: Could not save stage metadata: {e}")

        # Save optics metadata
        if self._optics is not None:
            try:
                if hasattr(self._optics, 'write_optics_metadata'):
                    self._optics.write_optics_metadata()
            except Exception as e:
                print(f"Warning: Could not save optics metadata: {e}")

    def _save_detector_pixels(self) -> None:
        """Save detector pixel data to the detector's directory."""
        if self._detector is None:
            return

        directory = getattr(self._detector, 'directory', None)
        if not directory:
            return

        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save pixel values (complex array)
        if hasattr(self._detector, '_pixel_values') and self._detector._pixel_values is not None:
            try:
                np.save(dir_path / "detector_pixel_values.npy", self._detector._pixel_values)
            except Exception as e:
                print(f"Warning: Could not save pixel values: {e}")

        # Save pixel amplitude
        if hasattr(self._detector, '_pixel_amplitude') and self._detector._pixel_amplitude is not None:
            try:
                np.save(dir_path / "detector_pixel_amplitude.npy", self._detector._pixel_amplitude)
            except Exception as e:
                print(f"Warning: Could not save pixel amplitude: {e}")

        # Save pixel phase
        if hasattr(self._detector, '_pixel_phase') and self._detector._pixel_phase is not None:
            try:
                np.save(dir_path / "detector_pixel_phase.npy", self._detector._pixel_phase)
            except Exception as e:
                print(f"Warning: Could not save pixel phase: {e}")

        # Save pixel intensity
        if hasattr(self._detector, '_pixel_intensity') and self._detector._pixel_intensity is not None:
            try:
                np.save(dir_path / "detector_pixel_intensity.npy", self._detector._pixel_intensity)
            except Exception as e:
                print(f"Warning: Could not save pixel intensity: {e}")

    def load_from_file(self, filepath: str) -> None:
        """
        Load the simulation state from a file.

        Args:
            filepath: Path to load the file from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)

        self.from_dict(state_dict)
        self._metadata.file_path = filepath
        self._dirty = False

    def from_dict(self, state_dict: Dict) -> None:
        """
        Deserialize the simulation state from a dictionary.

        Args:
            state_dict: Dictionary representation of the state
        """
        # Load metadata
        if 'metadata' in state_dict:
            meta = state_dict['metadata']
            self._metadata.name = meta.get('name', 'Untitled')
            self._metadata.description = meta.get('description', '')
            self._metadata.author = meta.get('author', '')

        # Load global working directory
        if 'global_working_directory' in state_dict:
            self._global_working_directory = state_dict['global_working_directory'] or ''

        # Load crystal
        if state_dict.get('crystal'):
            self._deserialize_crystal(state_dict['crystal'])
        else:
            self._crystal = None

        # Load sample
        if state_dict.get('sample'):
            self._deserialize_sample(state_dict['sample'])
        else:
            self._sample = None

        # Load beam
        if state_dict.get('beam'):
            self._deserialize_beam(state_dict['beam'])
        else:
            self._beam = None

        # Load detector
        if state_dict.get('detector'):
            self._deserialize_detector(state_dict['detector'])
        else:
            self._detector = None

        # Load stage
        if state_dict.get('stage'):
            self._deserialize_stage(state_dict['stage'])
        else:
            self._stage = None

        # Load optics
        if state_dict.get('optics'):
            self._deserialize_optics(state_dict['optics'])
        else:
            self._optics = None

        # Notify all observers
        self._notify("project_changed", self)
        self._notify("crystal_changed", self._crystal)
        self._notify("sample_changed", self._sample)
        self._notify("beam_changed", self._beam)
        self._notify("detector_changed", self._detector)
        self._notify("stage_changed", self._stage)
        self._notify("optics_changed", self._optics)

    def _deserialize_crystal(self, data: Dict) -> None:
        """Deserialize crystal from dictionary."""
        cif_path = data.get('cif_path')
        if not cif_path or not Path(cif_path).exists():
            print(f"Warning: CIF file not found: {cif_path}")
            self._crystal = None
            return

        try:
            self._crystal = CrystalType(cif_path)
            self._crystal.get_lattice_from_cif()

            # Restore cumulative rotation if available
            if data.get('cumulative_rotation'):
                self._crystal._cumulative_rotation = np.array(data['cumulative_rotation'])
                # Also update the lattice matrix if rotation was applied
                if data.get('lattice_matrix_conventional'):
                    self._crystal._lattice_matrix_conventional = np.array(data['lattice_matrix_conventional'])
        except Exception as e:
            print(f"Error loading crystal: {e}")
            self._crystal = None

    def _deserialize_sample(self, data: Dict) -> None:
        """Deserialize sample from dictionary."""
        directory = data.get('directory') or self._global_working_directory or os.getcwd()
        if not directory:
            self._sample = None
            return

        try:
            self._sample = SampleType(directory)
            # Try to load existing sample metadata
            if Path(directory).exists():
                try:
                    self._sample.read_sample_metadata()
                except Exception:
                    # If no metadata, just set basic properties
                    if data.get('dimensions'):
                        self._sample._dimensions = np.array(data['dimensions'], dtype=np.float32)
                    if data.get('sample_type'):
                        self._sample._sample_type = data['sample_type']
        except Exception as e:
            print(f"Error loading sample: {e}")
            self._sample = None

    def _deserialize_beam(self, data: Dict) -> None:
        """Deserialize beam from dictionary."""
        directory = data.get('directory') or self._global_working_directory or os.getcwd()

        try:
            self._beam = BeamType(directory)

            # Try to load from metadata file first
            metadata_loaded = False
            if directory and Path(directory).exists():
                metadata_path = Path(directory) / "beam_metadata.npy"
                if metadata_path.exists() and hasattr(self._beam, 'read_beam_metadata'):
                    try:
                        self._beam.read_beam_metadata()
                        metadata_loaded = True
                    except Exception:
                        pass

            # Fall back to creating from saved parameters
            if not metadata_loaded:
                energy = data.get('energy')
                if energy:
                    beam_shape = data.get('beam_shape', 'rectangular')
                    beam_size = tuple(data.get('beam_size', (1000, 1000)))
                    beam_samples = tuple(data.get('beam_samples', (64, 64)))
                    beam_profile = data.get('beam_profile', 'uniform')
                    pol_perp_rate = data.get('pol_perp_rate', 0.5)

                    self._beam.create_beam(
                        energy=energy,
                        beam_shape=beam_shape,
                        beam_size=beam_size,
                        beam_samples=beam_samples,
                        beam_profile=beam_profile,
                        pol_perp_rate=pol_perp_rate
                    )
        except Exception as e:
            print(f"Error loading beam: {e}")
            self._beam = None

    def _deserialize_detector(self, data: Dict) -> None:
        """Deserialize detector from dictionary."""
        directory = data.get('directory') or self._global_working_directory or os.getcwd()

        try:
            self._detector = DetectorType(directory)

            # Try to load from metadata file first
            metadata_loaded = False
            if directory and Path(directory).exists():
                metadata_path = Path(directory) / "detector_metadata.json"
                if metadata_path.exists() and hasattr(self._detector, 'read_detector_metadata'):
                    try:
                        self._detector.read_detector_metadata()
                        metadata_loaded = True
                    except Exception:
                        pass

            # Fall back to creating from saved parameters
            if not metadata_loaded:
                shape = data.get('shape')
                pixel_size = data.get('pixel_size')

                if shape and pixel_size:
                    geometry = data.get('geometry', 'rectangular')
                    construction_mode = data.get('construction_mode', 'plane')
                    input_mode = data.get('input_mode', 'spatial')

                    # For angular mode, shape may need to be a tuple (not np.array)
                    if input_mode == 'angular':
                        shape_arg = tuple(shape)
                    else:
                        shape_arg = np.array(shape)

                    self._detector.create_detector(
                        shape_arg,
                        np.array(pixel_size),
                        geometry=geometry,
                        construction_mode=construction_mode,
                        input_mode=input_mode
                    )

                    # Position detector if values available
                    distance = data.get('distance')
                    two_theta = data.get('two_theta')
                    eta = data.get('eta')

                    if distance is not None and two_theta is not None and eta is not None:
                        self._detector.position_detector_absolute(distance, two_theta, eta, degrees=False)

            # Load pixel data if available
            self._load_detector_pixels()

        except Exception as e:
            print(f"Error loading detector: {e}")
            self._detector = None

    def _load_detector_pixels(self) -> None:
        """Load detector pixel data from the detector's directory."""
        if self._detector is None:
            return

        directory = getattr(self._detector, 'directory', None)
        if not directory:
            return

        dir_path = Path(directory)
        if not dir_path.exists():
            return

        # Load pixel values (complex array)
        pixel_values_path = dir_path / "detector_pixel_values.npy"
        if pixel_values_path.exists():
            try:
                self._detector._pixel_values = np.load(pixel_values_path)
            except Exception as e:
                print(f"Warning: Could not load pixel values: {e}")

        # Load pixel amplitude
        pixel_amplitude_path = dir_path / "detector_pixel_amplitude.npy"
        if pixel_amplitude_path.exists():
            try:
                self._detector._pixel_amplitude = np.load(pixel_amplitude_path)
            except Exception as e:
                print(f"Warning: Could not load pixel amplitude: {e}")

        # Load pixel phase
        pixel_phase_path = dir_path / "detector_pixel_phase.npy"
        if pixel_phase_path.exists():
            try:
                self._detector._pixel_phase = np.load(pixel_phase_path)
            except Exception as e:
                print(f"Warning: Could not load pixel phase: {e}")

        # Load pixel intensity
        pixel_intensity_path = dir_path / "detector_pixel_intensity.npy"
        if pixel_intensity_path.exists():
            try:
                self._detector._pixel_intensity = np.load(pixel_intensity_path)
            except Exception as e:
                print(f"Warning: Could not load pixel intensity: {e}")

    def _deserialize_stage(self, data: Dict) -> None:
        """Deserialize stage from dictionary."""
        directory = data.get('directory') or self._global_working_directory or os.getcwd()

        try:
            self._stage = StageType(directory)

            # Try to load from metadata file first
            metadata_loaded = False
            if directory and Path(directory).exists():
                metadata_path = Path(directory) / "stage_metadata.npy"
                if metadata_path.exists() and hasattr(self._stage, 'read_stage_metadata'):
                    try:
                        self._stage.read_stage_metadata()
                        metadata_loaded = True
                    except Exception:
                        pass

            # Fall back to creating default stage and setting motor values
            if not metadata_loaded:
                self._stage.create_stage()

                # Restore motor values if available
                motor_value = data.get('motor_value')
                if motor_value and hasattr(self._stage, 'set_motor_value_absolute'):
                    self._stage.set_motor_value_absolute(motor_value_abs=motor_value, degrees=False)
        except Exception as e:
            print(f"Error loading stage: {e}")
            self._stage = None

    def _deserialize_optics(self, data: Dict) -> None:
        """Deserialize optics from dictionary."""
        directory = data.get('directory') or self._global_working_directory or os.getcwd()

        try:
            self._optics = OpticsType(directory)

            # Try to load from metadata file first
            metadata_loaded = False
            if directory and Path(directory).exists():
                metadata_path = Path(directory) / "optics_metadata.npy"
                if metadata_path.exists() and hasattr(self._optics, 'read_optics_metadata'):
                    try:
                        self._optics.read_optics_metadata()
                        metadata_loaded = True
                    except Exception:
                        pass

            # Fall back to restoring stack from saved data
            if not metadata_loaded:
                stack = data.get('stack', [])
                for component in stack:
                    if isinstance(component, dict):
                        comp_type = component.get('type', '')
                        # Add components based on type
                        if 'free_space' in comp_type.lower():
                            distance = component.get('distance', 0.01)
                            self._optics.add_free_space(distance)
                        elif 'angular_filter' in comp_type.lower():
                            acceptance = component.get('acceptance', 5e-3)
                            mode = component.get('mode', '1d')
                            self._optics.add_angular_filter(acceptance, mode=mode)
                        # Add more component types as needed
        except Exception as e:
            print(f"Error loading optics: {e}")
            self._optics = None

    def get_all_objects(self) -> Dict[str, Any]:
        """
        Get all simulation objects as a dictionary.

        Returns:
            Dictionary mapping object names to objects
        """
        return {
            "crystal": self._crystal,
            "sample": self._sample,
            "beam": self._beam,
            "detector": self._detector,
            "stage": self._stage,
            "optics": self._optics,
            "defects": self._defects,
            "deformation": self._deformation,
            "experiment": self._experiment,
            "analysis": self._analysis,
        }
