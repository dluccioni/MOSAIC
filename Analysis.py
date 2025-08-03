# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import os
import sys
import json
import time
import logging
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.DEBUG)

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class analysis:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=os.getcwd()):
        """
        Initialize the analysis object.

        Args:
            directory (str, optional): Path where analysis results will be stored. Defaults to current working directory.
        """
        try:
            self.directory = directory
            if not os.path.isdir(self.directory):
                os.makedirs(self.directory)
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    # Static Function
    @staticmethod
    def surf_plot(X, Y, Z, title, xlabel="Frequency (1/px)", ylabel="Distance", zlabel="FFT Amplitude", figsize=(12, 12)):
        """
        Generate a 3D surface plot.

        Args:
            X (ndarray): 2D array of x-coordinates.
            Y (ndarray): 2D array of y-coordinates.
            Z (ndarray): 2D array of z-values.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            zlabel (str): Z-axis label.
            figsize (tuple): Size of the figure.

        Returns:
            tuple: matplotlib Figure and Axes3DSubplot objects.
        """
        start_time = time.perf_counter()
        try:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho')
            ax.set_title(title)
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', linewidth=0, antialiased=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            ax.view_init(elev=90, azim=0)
            fig.colorbar(surf, shrink=0.5, aspect=8)
            return fig, ax
        except Exception as e:
            logging.error(f"Error in surf_plot: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            logging.debug(f"Execution time for surf_plot: {end_time - start_time:.4f} s")

    @staticmethod
    def line_plot(x, y, title, xlabel="Detector X", ylabel="Pixel Value", figsize=(12, 12)):
        """
        Generate a 2D line plot.

        Args:
            x (ndarray): X values.
            y (ndarray): Y values.
            title (str): Plot title.
            xlabel (str): Label for X-axis.
            ylabel (str): Label for Y-axis.
            figsize (tuple): Size of the figure.

        Returns:
            tuple: matplotlib Figure and AxesSubplot objects.
        """
        start_time = time.perf_counter()
        try:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.set_title(title)
            ax.plot(x, y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return fig, ax
        except Exception as e:
            logging.error(f"Error in line_plot: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            logging.debug(f"Execution time for line_plot: {end_time - start_time:.4f} s")

    def _save_figure(self, fig, filename):
        """
        Save a matplotlib figure and close it.

        Args:
            fig (matplotlib.figure.Figure): Figure object to save.
            filename (str): Filename to save the figure under.
        """
        try:
            fig.savefig(os.path.join(self.directory, filename))
            plt.close(fig)
        except Exception as e:
            logging.error(f"Failed to save figure {filename}: {e}")
            raise

    def integrate_detector_along_axis(self, detector, data_type="Intensity", axis="x", system="cartesian", degrees=True, bins=200, aggregator="mean", plot=True, title="Integrated Detector Data", xlabel=None, ylabel="Integrated Value", figsize=(8, 6)):
        """
        Integrate detector data along a specified axis.

        Args:
            detector: Detector object.
            data_type (str): Data type to integrate ("Intensity", "Amplitude", etc).
            axis (str): Axis along which to integrate.
            system (str): Coordinate system ("cartesian" or "angular").
            degrees (bool): If True, output angular units in degrees.
            bins (int): Number of bins for integration.
            aggregator (str): Aggregation method ("mean" or "sum").
            plot (bool): Whether to generate a plot.
            title (str): Plot title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            figsize (tuple): Size of the figure.

        Returns:
            tuple: Bin centers and integrated values.
        """
        start_time = time.perf_counter()
        try:
            data_map = {
                "Intensity": detector.pixel_intensity,
                "Amplitude": detector.pixel_amplitude,
                "Phase": detector.pixel_phase
            }
            if data_type not in data_map:
                raise ValueError(f"Invalid data_type: {data_type}")
            data = data_map[data_type]
            coords = detector.pixel_coordinates

            if system == "angular":
                coords = detector.coordinate_conversion(coords, input_system="cartesian", output_system="angular", units="deg" if degrees else "rad")
            axis_map = {"x": 0, "y": 1, "z": 2, "eta": 0, "2theta": 1, "distance": 2}
            idx = axis_map.get(axis.lower())
            if idx is None:
                raise ValueError(f"Invalid axis: {axis}")

            coord_vals = coords[idx]
            flattened_data = data.flatten()

            bins_range = (coord_vals.min(), coord_vals.max())
            hist, bin_edges = np.histogram(coord_vals, bins=bins, range=bins_range, weights=flattened_data)

            if aggregator == "mean":
                counts, _ = np.histogram(coord_vals, bins=bins, range=bins_range)
                with np.errstate(divide='ignore', invalid='ignore'):
                    hist = np.divide(hist, counts, out=np.zeros_like(hist), where=counts != 0)

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            if plot:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(bin_centers, hist)
                ax.set_title(title)
                ax.set_xlabel(xlabel or axis)
                ax.set_ylabel(ylabel)
                self._save_figure(fig, f"Integrated_{data_type}_{axis}.png")

            return bin_centers, hist
        except Exception as e:
            logging.error(f"Error in integrate_detector_along_axis: {e}")
            raise
        finally:
            end_time = time.perf_counter()
            logging.debug(f"Execution time for integrate_detector_along_axis: {end_time - start_time:.4f} s")
