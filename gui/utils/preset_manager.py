# -----------------------------------------------------------------------------
# Preset Manager
# -----------------------------------------------------------------------------
"""
Manager for simulation presets.

Handles:
- Loading/saving presets from JSON files
- User and built-in preset directories
- Preset validation and migration
- Export to Python scripts
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class PresetMetadata:
    """Metadata for a preset."""
    name: str
    category: str
    description: str
    date: str
    version: str = "1.0"


class PresetManager:
    """
    Manager for simulation presets.

    Handles both built-in and user presets, with support for
    saving, loading, and exporting configurations.
    """

    CURRENT_VERSION = "1.0"

    def __init__(self, builtin_dir: Optional[Path] = None, user_dir: Optional[Path] = None):
        """
        Initialize the preset manager.

        Args:
            builtin_dir: Directory for built-in presets (defaults to gui/presets)
            user_dir: Directory for user presets (defaults to ~/.xray_simulator/presets)
        """
        self.builtin_dir = builtin_dir or (Path(__file__).parent.parent / "presets")
        self.user_dir = user_dir or (Path.home() / ".xray_simulator" / "presets")

        # Ensure directories exist
        self.builtin_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> Dict[str, List[str]]:
        """
        List all available presets by category.

        Returns:
            Dict mapping category names to lists of preset names
        """
        presets = {"Built-in": [], "User": []}

        # Built-in presets
        for preset_file in sorted(self.builtin_dir.glob("*.json")):
            presets["Built-in"].append(preset_file.stem)

        # User presets by category
        for preset_file in sorted(self.user_dir.glob("*.json")):
            try:
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                category = data.get("category", "User")
                if category not in presets:
                    presets[category] = []
                presets[category].append(preset_file.stem)
            except Exception:
                presets["User"].append(preset_file.stem)

        # Remove empty categories
        return {k: v for k, v in presets.items() if v}

    def get_preset_path(self, name: str, user_only: bool = False) -> Optional[Path]:
        """
        Get the path to a preset file.

        Args:
            name: Preset name (without .json extension)
            user_only: If True, only search user presets

        Returns:
            Path to preset file, or None if not found
        """
        # Check user presets first
        user_path = self.user_dir / f"{name}.json"
        if user_path.exists():
            return user_path

        # Check built-in presets
        if not user_only:
            builtin_path = self.builtin_dir / f"{name}.json"
            if builtin_path.exists():
                return builtin_path

        return None

    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a preset by name.

        Args:
            name: Preset name

        Returns:
            Preset data dict, or None if not found
        """
        path = self.get_preset_path(name)
        if path is None:
            return None

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Validate and migrate if needed
            data = self._migrate_preset(data)
            return data

        except Exception as e:
            print(f"Error loading preset '{name}': {e}")
            return None

    def save_preset(self, name: str, data: Dict[str, Any],
                    category: str = "User", description: str = "",
                    overwrite: bool = False) -> bool:
        """
        Save a preset to the user directory.

        Args:
            name: Preset name
            data: Preset parameters dict
            category: Category for organization
            description: Optional description
            overwrite: Whether to overwrite existing preset

        Returns:
            True if saved successfully
        """
        # Sanitize name
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        if not safe_name:
            return False

        path = self.user_dir / f"{safe_name}.json"

        if path.exists() and not overwrite:
            return False

        try:
            preset_data = {
                "name": name,
                "category": category,
                "description": description,
                "date": datetime.now().isoformat(),
                "version": self.CURRENT_VERSION,
                "parameters": data
            }

            with open(path, 'w') as f:
                json.dump(preset_data, f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving preset '{name}': {e}")
            return False

    def delete_preset(self, name: str) -> bool:
        """
        Delete a user preset.

        Args:
            name: Preset name

        Returns:
            True if deleted successfully
        """
        path = self.user_dir / f"{name}.json"

        if not path.exists():
            return False

        try:
            path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting preset '{name}': {e}")
            return False

    def export_preset(self, name: str, output_path: Path) -> bool:
        """
        Export a preset to an external file.

        Args:
            name: Preset name
            output_path: Destination file path

        Returns:
            True if exported successfully
        """
        data = self.load_preset(name)
        if data is None:
            return False

        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting preset '{name}': {e}")
            return False

    def import_preset(self, file_path: Path, name: Optional[str] = None,
                      overwrite: bool = False) -> bool:
        """
        Import a preset from an external file.

        Args:
            file_path: Source file path
            name: Optional name override
            overwrite: Whether to overwrite existing preset

        Returns:
            True if imported successfully
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Validate structure
            if "parameters" not in data:
                print("Invalid preset file: missing 'parameters' key")
                return False

            preset_name = name or data.get("name", file_path.stem)
            return self.save_preset(
                preset_name,
                data["parameters"],
                category=data.get("category", "Imported"),
                description=data.get("description", ""),
                overwrite=overwrite
            )

        except Exception as e:
            print(f"Error importing preset: {e}")
            return False

    def _migrate_preset(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate preset to current version if needed.

        Args:
            data: Preset data

        Returns:
            Migrated preset data
        """
        version = data.get("version", "1.0")

        # Currently only version 1.0, no migrations needed
        # Future versions would add migration logic here

        data["version"] = self.CURRENT_VERSION
        return data

    def get_preset_info(self, name: str) -> Optional[PresetMetadata]:
        """
        Get metadata for a preset without loading full parameters.

        Args:
            name: Preset name

        Returns:
            PresetMetadata or None if not found
        """
        path = self.get_preset_path(name)
        if path is None:
            return None

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            return PresetMetadata(
                name=data.get("name", name),
                category=data.get("category", "Unknown"),
                description=data.get("description", ""),
                date=data.get("date", "Unknown"),
                version=data.get("version", "1.0")
            )

        except Exception:
            return None

    def create_preset_from_state(self, state, components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create preset parameters from current simulation state.

        Args:
            state: SimulationState instance
            components: List of components to include, or None for all

        Returns:
            Parameters dict
        """
        all_components = ["crystal", "sample", "beam", "detector", "stage", "optics", "defects", "deformation"]
        if components is None:
            components = all_components

        params = {}

        if "crystal" in components and state.crystal:
            crystal = state.crystal
            params["crystal"] = {
                "lattice_parameters": {
                    "a": getattr(crystal, 'a', None),
                    "b": getattr(crystal, 'b', None),
                    "c": getattr(crystal, 'c', None),
                    "alpha": getattr(crystal, 'alpha', None),
                    "beta": getattr(crystal, 'beta', None),
                    "gamma": getattr(crystal, 'gamma', None),
                },
                "space_group": getattr(crystal, 'space_group', None),
            }

        if "sample" in components and state.sample:
            sample = state.sample
            params["sample"] = {
                "dimensions": {
                    "Lx": getattr(sample, 'Lx', None),
                    "Ly": getattr(sample, 'Ly', None),
                    "Lz": getattr(sample, 'Lz', None),
                },
                "type": getattr(sample, 'sample_type', 'single'),
            }

        if "beam" in components and state.beam:
            beam = state.beam
            params["beam"] = {
                "energy": getattr(beam, 'energy', None),
                "shape": getattr(beam, 'shape', None),
                "Ny": getattr(beam, 'Ny', None),
                "Nz": getattr(beam, 'Nz', None),
                "Ly": getattr(beam, 'Ly', None),
                "Lz": getattr(beam, 'Lz', None),
                "profile": getattr(beam, 'profile', None),
                "polarization_rate": getattr(beam, 'polarization_rate', None),
            }

        if "detector" in components and state.detector:
            detector = state.detector
            # Get shape (handle both old Ny/Nz and new shape attributes)
            shape = getattr(detector, 'shape', None)
            if shape is None:
                Ny = getattr(detector, 'Ny', None)
                Nz = getattr(detector, 'Nz', None)
                shape = [Ny, Nz] if Ny is not None and Nz is not None else None
            elif hasattr(shape, '__iter__'):
                shape = list(shape)

            # Handle angular_range serialization
            angular_range = getattr(detector, '_angular_range', None)
            if angular_range is not None:
                angular_range = list(angular_range)

            params["detector"] = {
                "shape": shape,
                "pixel_size": getattr(detector, 'pixel_size', None),
                "geometry": getattr(detector, '_geometry', 'rectangular'),
                "construction_mode": getattr(detector, '_construction_mode', 'plane'),
                "input_mode": getattr(detector, '_input_mode', 'spatial'),
                "angular_range": angular_range,
                "distance": getattr(detector, 'distance', None),
                "two_theta": getattr(detector, 'two_theta', None),
                "eta": getattr(detector, 'eta', None),
            }

        if "stage" in components and state.stage:
            stage = state.stage
            if hasattr(stage, 'motors'):
                params["stage"] = {
                    "motors": {name: getattr(motor, 'value', 0) for name, motor in stage.motors.items()}
                }

        if "optics" in components and state.optics:
            params["optics"] = {
                "components": []  # Would serialize optics stack
            }

        return params

    def apply_preset_to_state(self, state, params: Dict[str, Any]) -> None:
        """
        Apply preset parameters to simulation state.

        Args:
            state: SimulationState instance
            params: Parameters dict from preset
        """
        # Crystal
        if "crystal" in params and state.crystal:
            crystal_params = params["crystal"]
            if "cif_file" in crystal_params:
                state.crystal.load_cif(crystal_params["cif_file"])

        # Sample
        if "sample" in params and state.sample:
            sample_params = params["sample"]
            dims = sample_params.get("dimensions", {})
            if "Lx" in dims:
                state.sample.Lx = dims["Lx"]
            if "Ly" in dims:
                state.sample.Ly = dims["Ly"]
            if "Lz" in dims:
                state.sample.Lz = dims["Lz"]

        # Beam
        if "beam" in params and state.beam:
            beam_params = params["beam"]
            for key in ["energy", "shape", "Ny", "Nz", "Ly", "Lz", "profile", "polarization_rate"]:
                if key in beam_params and beam_params[key] is not None:
                    setattr(state.beam, key, beam_params[key])

        # Detector
        if "detector" in params and state.detector:
            det_params = params["detector"]
            for key in ["Ny", "Nz", "pixel_size", "distance", "two_theta", "eta"]:
                if key in det_params and det_params[key] is not None:
                    setattr(state.detector, key, det_params[key])

        # Stage
        if "stage" in params and state.stage:
            stage_params = params["stage"]
            if "motors" in stage_params and hasattr(state.stage, 'motors'):
                for name, value in stage_params["motors"].items():
                    if name in state.stage.motors:
                        state.stage.motors[name].value = value

        # Notify observers
        state.notify_observers("preset_loaded")
