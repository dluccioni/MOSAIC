# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import os
import gc

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class optics:

    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=None):
        self.directory = directory
        if self.directory is not None and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # List of optical components
        self._components = []
        self._direction  = None
        self._origin     = None

    def read_optics_metadata(self):
        """
        Stub for reading an optics JSON or other meta file from self.directory
        """
        pass

    def write_optics_metadata(self):
        """
        Stub for writing an optics JSON or other meta file to self.directory
        """
        pass

    def add_free_space(self, length_mm):
        """
        Add a free-space propagation segment of length in millimeters.
        """
        self._components.append({
            'kind'   : 'free space',
            'length' : float(length_mm)
        })

    def add_CRL_box(self, number, focal_length_mm, thickness_mm,
                    absorption_sigma=np.inf):
        """
        A simplified compound refractive lens (CRL) "box" specification.
        For example, 'number' CRLs in series, each of focal_length_mm in
        thin-lens approximation, thickness_mm for absorption, etc.
        """
        self._components.append({
            'kind'           : 'lens box',
            'number'         : int(number),
            'focal_length'   : float(focal_length_mm),
            'thickness'      : float(thickness_mm),
            'absorption_sigma': float(absorption_sigma)
        })

    def add_aperture(self, width_mm, shape='square'):
        """
        Hard aperture (default: square of given width in mm).
        """
        self._components.append({
            'kind'  : 'aperture',
            'type'  : shape.lower(),
            'width' : float(width_mm)
        })

    def add_custom_component(self, component):
        """
        Add any arbitrary custom component (dict) to the optics stack.
        """
        self._components.append(component)

    @property
    def components(self):
        """
        Return the internal list of components.
        """
        return self._components