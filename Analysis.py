# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import os
import sys
import json
import time
from Logging import logging
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class analysis(logging):
    
    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "distance_fft_dependance",
        "integrate_detector_along_axis",
        "surf_plot",
        "line_plot",
    )
    
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
        super().__init__(log_name="analysis")
        self.directory = directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

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
            raise e

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
        try:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.set_title(title)
            ax.plot(x, y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return fig, ax
        except Exception as e:
            raise e

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
            raise e
    ## Main Functions
    def distance_fft_dependance(self,sample,beam,stage,detector,distance_array,plot_prefix="Test",output_pixel_values=False,offset_list=None):
        """
        Compute amplitude/phase summations and their FFTs at various detector distances.

        Moves the detector to each distance in distance_array, computes scattering,
        generates 1D and 2D plots for amplitude/phase, and creates combined 3D surface
        plots. All figures are saved to self.directory.

        Args:
            sample: Sample object for scattering computation.
            beam: Beam object for atomic direct interaction.
            stage: Stage object for positioning.
            detector: Detector object to capture scattering data.
            distance_array (ndarray): Array of detector distances to iterate over.
            plot_prefix (str): Prefix for saved plot filenames. Defaults to "Test".
            output_pixel_values (bool): If True, return pixel values list. Defaults to False.
            offset_list (list, optional): List of offset arrays to subtract from pixel values
                at each distance. Defaults to None.

        Returns:
            tuple: If output_pixel_values is False, returns (X_fft, Y_fft, Z_fft_amp, Z_fft_pha).
                If output_pixel_values is True, returns (X_fft, Y_fft, Z_fft_amp, Z_fft_pha, pixel_values_list).
        """
        freq_array = None
        real_amplitude_list = []
        real_phase_list = []
        fft_amplitude_list = []
        fft_phase_list = []
        pixel_values_list = []
        
        for idx, d in enumerate(distance_array):
            print(f"Processing distance: {d} || {idx+1}/{distance_array.size}")
            # Move detector to new distance and compute scattering
            detector.position_detector_absolute(d,detector.two_theta,detector.eta)
            beam.atomic_direct_interaction(sample,detector,stage,scattering=True, scattering_params=[None], transmission=False, transmission_params=[1.7,1.0], use_gpu=True)
            if output_pixel_values:
                pixel_values_list.append(detector.pixel_values)
            if offset_list != None:
                detector.input_pixel_values(detector.pixel_values-offset_list[idx])
            # 1) Plot Intensity
            fig_int, ax_int = detector.plot_detector(type="Intensity")
            fig_int.savefig(os.path.join(self.directory,plot_prefix + f"_Intensity_2D_Real_Distance_{d}.png"))
            plt.close(fig_int)
            # 2) Plot Phase
            fig_pha, ax_pha = detector.plot_detector(type="Phase")
            fig_pha.savefig(os.path.join(self.directory,plot_prefix + f"_Phase_2D_Real_Distance_{d}.png"))
            plt.close(fig_pha)
            # 3) Plot Amplitude
            fig_amp, ax_amp = detector.plot_detector(type="Amplitude")
            fig_amp.savefig(os.path.join(self.directory,plot_prefix + f"_Amplitude_2D_Real_Distance_{d}.png"))
            plt.close(fig_amp)
            # -------------------------------------------------------------------
            # Summed amplitude & phase -> compute FFT
            # -------------------------------------------------------------------
            # Amplitude
            summed_amplitude = np.mean(detector.pixel_amplitude, axis=0)
            fft_values_amplitude = np.abs(np.fft.fft(summed_amplitude)[1:summed_amplitude.size // 2])
            # fft_values_amplitude /= np.max(fft_values_amplitude)
            real_amplitude_list.append(summed_amplitude)
            fft_amplitude_list.append(fft_values_amplitude)
            # Phase
            summed_phase = np.mean(detector.pixel_phase, axis=0)
            fft_values_phase = np.abs(np.fft.fft(summed_phase)[1:summed_phase.size // 2])
            # fft_values_phase /= np.max(fft_values_phase)
            real_phase_list.append(summed_phase)
            fft_phase_list.append(fft_values_phase)
            # Compute frequency array only once
            if freq_array is None:
                freq_array = np.fft.fftfreq(summed_amplitude.size,d=1/detector.pixel_size[0])[1:summed_amplitude.size // 2]
            # -------------------------------------------------------------------
            # Create the line plots and save them
            # -------------------------------------------------------------------
            x_pos = np.linspace(-detector.shape[0]/2*detector.pixel_size[0],
                                detector.shape[0]/2*detector.pixel_size[0],
                                detector.shape[0])
            # Amplitude trace
            fig_amp_line, ax_amp_line = self.line_plot(x_pos,summed_amplitude,title=f"Amplitude Trace Plot (Distance = {d})",xlabel="Detector X (mm)",ylabel="Summed Amplitude")
            fig_amp_line.savefig(os.path.join(self.directory,plot_prefix + f"_Amplitude_1D_Real_Distance_{d}.png"))
            plt.close(fig_amp_line)
            # Amplitude FFT trace
            fig_amp_fft_line, ax_amp_fft_line = self.line_plot(freq_array,fft_values_amplitude,title=f"Amplitude FFT Trace Plot (Distance = {d})",xlabel="Frequency (1/px)",ylabel="Log(|FFT(Amplitude)|)")
            fig_amp_fft_line.savefig(os.path.join(self.directory,plot_prefix + f"_Amplitude_1D_FFT_Distance_{d}.png"))
            plt.close(fig_amp_fft_line)
            # Phase trace
            fig_phase_line, ax_phase_line = self.line_plot(x_pos,summed_phase,title=f"Phase Trace Plot (Distance = {d})",xlabel="Detector X (mm)",ylabel="Summed Phase")
            fig_phase_line.savefig(os.path.join(self.directory,plot_prefix + f"_Phase_1D_Real_Distance_{d}.png"))
            plt.close(fig_phase_line)
            # Phase FFT trace
            fig_phase_fft_line, ax_phase_fft_line = self.line_plot(freq_array,fft_values_phase,title=f"Phase FFT Trace Plot (Distance = {d})",xlabel="Frequency (1/px)",ylabel="Log(|FFT(Phase)|)")
            fig_phase_fft_line.savefig(os.path.join(self.directory,plot_prefix + f"_Phase_1D_FFT_Distance_{d}.png"))
            plt.close(fig_phase_fft_line)
        # -----------------------------------------------------------------------
        # After the loop, create 3D surface plots for amplitude and phase
        # -----------------------------------------------------------------------
        Z_fft_amp = np.array(fft_amplitude_list)
        Z_fft_pha = np.array(fft_phase_list)
        Z_real_amp = np.array(real_amplitude_list)
        Z_real_pha = np.array(real_phase_list)
        # Make X,Y array
        X_fft, Y_fft = np.meshgrid(freq_array, distance_array)
        detector_x = np.linspace(-detector.shape[0]/2*detector.pixel_size[0],detector.shape[0]/2*detector.pixel_size[0],detector.shape[0])
        X_real, Y_real = np.meshgrid(detector_x, distance_array)
        # Combined Amplitude FFT Surface
        fig_fft_amp_surf, ax_fft_amp_surf = self.surf_plot(X_fft, Y_fft, Z_fft_amp,"Combined Amplitude FFT Surface Plot",xlabel="Frequency (1/px)",ylabel="Distance",zlabel="Log(|FFT(Amplitude)|)")
        fig_fft_amp_surf.savefig(os.path.join(self.directory,plot_prefix + f"_Combined_Amplitude_FFT_Surface.png"))
        plt.close(fig_fft_amp_surf)
        # Combined Phase FFT Surface
        fig_fft_pha_surf, ax_fft_pha_surf = self.surf_plot(X_fft, Y_fft, Z_fft_pha,"Combined Phase FFT Surface Plot",xlabel="Frequency (1/px)",ylabel="Distance",zlabel="Log(|FFT(Phase)|)")
        fig_fft_pha_surf.savefig(os.path.join(self.directory,plot_prefix + f"_Combined_Phase_FFT_Surface.png"))
        plt.close(fig_fft_pha_surf)
        # Combined Amplitude Real Surface
        fig_real_amp_surf, ax_real_amp_surf = self.surf_plot(X_real, Y_real, Z_real_amp,"Combined Amplitude Real Surface Plot",xlabel="Detector X",ylabel="Distance",zlabel="Amplitude",cmap='gist_yarg',log=False)
        fig_real_amp_surf.savefig(os.path.join(self.directory,plot_prefix + f"_Combined_Amplitude_Real_Surface.png"))
        plt.close(fig_real_amp_surf)
        # Combined Phase Real Surface
        fig_real_pha_surf, ax_real_pha_surf = self.surf_plot(X_real, Y_real, Z_real_pha,"Combined Phase Real Surface Plot",xlabel="Detector X",ylabel="Distance",zlabel="Phase",cmap='gist_yarg',log=False)
        fig_real_pha_surf.savefig(os.path.join(self.directory,plot_prefix + f"_Combined_Phase_Real_Surface.png"))
        plt.close(fig_real_pha_surf)
        
        if output_pixel_values:
            return X_fft, Y_fft, Z_fft_amp, Z_fft_pha, pixel_values_list
        return X_fft, Y_fft, Z_fft_amp, Z_fft_pha
    
    def integrate_detector_along_axis(self, detector, data_type="Intensity", axis="x", system="cartesian", degrees=True, bins=200, aggregator="mean", plot=True, save_plot=False, title="Integrated Detector Data", xlabel=None, ylabel="Integrated Value", figsize=(8, 6)):
        """
        Integrate detector data along a specified axis.

        Args:
            detector: Detector object containing pixel data and coordinates.
            data_type (str): Data type to integrate ("Intensity", "Amplitude", or "Phase").
                Defaults to "Intensity".
            axis (str): Axis along which to integrate ("x", "y", "z" for cartesian,
                or "eta", "2theta", "distance" for angular). Defaults to "x".
            system (str): Coordinate system ("cartesian" or "angular"). Defaults to "cartesian".
            degrees (bool): If True, output angular units in degrees. Defaults to True.
            bins (int): Number of bins for integration. Defaults to 200.
            aggregator (str): Aggregation method ("mean" or "sum"). Defaults to "mean".
            plot (bool): Whether to generate a plot. Defaults to True.
            save_plot (bool): Whether to save the plot to file. Defaults to False.
            title (str): Plot title. Defaults to "Integrated Detector Data".
            xlabel (str, optional): X-axis label. Defaults to axis name if None.
            ylabel (str): Y-axis label. Defaults to "Integrated Value".
            figsize (tuple): Size of the figure. Defaults to (8, 6).

        Returns:
            tuple: Bin centers (ndarray) and integrated values (ndarray).
        """
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
                if save_plot:
                    self._save_figure(fig, f"Integrated_{data_type}_{axis}.png")

            return bin_centers, hist
        except Exception as e:
            raise e
