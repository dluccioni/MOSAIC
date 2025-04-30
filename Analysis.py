# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class analysis:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=os.getcwd()):
        self.directory = directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    # Static Function
    @staticmethod
    def surf_plot(X,Y,Z,title,xlabel="Frequency (1/px)",ylabel="Distance",zlabel="FFT Amplitude",figsize=(12, 12)):
        # Now create the 3D surface plot for fig1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        ax.set_title(title)
        # Create a meshgrid so X is frequency, Y is distance
        # Plot the surface with interpolated colors
        surf = ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            edgecolor='none',      # no grid lines
            linewidth=0,
            antialiased=True
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.view_init(elev=90, azim=0)
        # Add a color bar for the surface
        fig.colorbar(surf, shrink=0.5, aspect=8)
        # fig1.show()
        return fig,ax
    
    @staticmethod
    def line_plot(x,y,title,xlabel="Detector X",ylabel="Pixel Value",figsize=(12, 12)):
         # Now create the 3D surface plot for fig1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        plot = ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # fig1.show()
        return fig,ax

    ## Main Functions
    def distance_fft_dependance(self, sample, beam, detector, distance_array, plot_prefix="Test"):
        """
        Computes amplitude/phase summations and their FFTs at various detector
        distances, plots them, and saves all generated figures to self.directory.
        """
        freq_array = None
        fft_amplitude_list = []
        fft_phase_list = []
        
        for idx, d in enumerate(distance_array):
            print(f"Processing distance: {d} || {idx+1}/{distance_array.size}")
            # Move detector to new distance and compute scattering
            detector.position_detector_absolute(d, detector.two_theta, detector.nu)
            beam.atomic_direct_scattering(sample, detector)
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
            summed_amplitude = np.sum(detector.pixel_amplitude, axis=0)
            fft_values_amplitude = np.log(np.abs(np.fft.fft(summed_amplitude)[1:summed_amplitude.size // 2]) + 1e-6)
            fft_amplitude_list.append(fft_values_amplitude)
            # Phase
            summed_phase = np.sum(detector.pixel_phase, axis=0)
            fft_values_phase = np.log(np.abs(np.fft.fft(summed_phase)[1:summed_phase.size // 2]) + 1e-6)
            fft_phase_list.append(fft_values_phase)
            # Compute frequency array only once
            if freq_array is None:
                freq_array = np.fft.fftfreq(summed_amplitude.size,d=1/detector.pixel_size[0])[1:summed_amplitude.size // 2]
            # -------------------------------------------------------------------
            # Create the line plots and save them
            # -------------------------------------------------------------------
            x_pos = np.linspace(-detector.shape[0]*detector.pixel_size[0],
                                detector.shape[0]*detector.pixel_size[0],
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
        Z_amp = np.array(fft_amplitude_list)
        Z_pha = np.array(fft_phase_list)
        # Make X,Y array
        X, Y = np.meshgrid(freq_array, distance_array)
        # Combined Amplitude FFT Surface
        fig_amp_surf, ax_amp_surf = self.surf_plot(X, Y, Z_amp,"Combined Amplitude FFT Surface Plot",xlabel="Frequency (1/px)",ylabel="Distance",zlabel="Log(|FFT(Amplitude)|)")
        fig_amp_surf.savefig(os.path.join(self.directory,plot_prefix + f"_Combined_Amplitude_FFT_Surface_Distance_{d}.png"))
        plt.close(fig_amp_surf)
        # Combined Phase FFT Surface
        fig_pha_surf, ax_pha_surf = self.surf_plot(X, Y, Z_pha,"Combined Phase FFT Surface Plot",xlabel="Frequency (1/px)",ylabel="Distance",zlabel="Log(|FFT(Phase)|)")
        fig_pha_surf.savefig(os.path.join(self.directory,plot_prefix + f"Combined_Phase_FFT_Surface_Distance_{d}.png"))
        plt.close(fig_pha_surf)
        return X, Y, Z_amp, Z_pha
    
    def integrate_axis(self, data, axis_data=None, integration_axis=0,title="Integrated Detector", xlabel="X-axis", ylabel="Y-axis",scaling="linear",figsize=(8, 6)):
        """
        Integrates a detector data array (data is a 2D array with dimensions [Nx, Ny])
        along a chosen axis and plots the resulting 1D line scan or 2D surface,
        depending on the dimensionality of the result. If axis_data is provided
        (and is consistent with data shape), those values are used for the plot axes;
        otherwise integer indices are used.

        Examples
        --------
        If data.shape = (Nx, Ny), and you set integration_axis=0:
            -> integrated_data has shape (Ny,).
            -> A 1D line plot is generated: integrated_data vs. Y-axis.

        If you set integration_axis=1:
            -> integrated_data has shape (Nx,).
            -> A 1D line plot is generated: integrated_data vs. X-axis.

        If you do not sum at all (not typical here, but if data is already 2D),
        you would end up plotting a 2D surface. (But in this case, a single call to
        np.sum(..., axis=integration_axis) always reduces one dimension for 2D data,
        resulting in 1D.)

        Parameters
        ----------
        data : ndarray, shape (Nx, Ny)
            The 2D detector data to be integrated and plotted.
        axis_data : None or tuple/list of arrays
            Coordinate arrays for each dimension. For example:
                axis_data[0] : x-coordinates, shape (Nx,)
                axis_data[1] : y-coordinates, shape (Ny,)
            If None, integer indices are used.
        integration_axis : int
            The axis along which to integrate (0 or 1). Defaults to 0.
        title : str
            Plot title.
        xlabel : str
            Label for the X-axis on the plot.
        ylabel : str
            Label for the Y-axis on the plot.

        Returns
        -------
        integrated_data : ndarray
            The 1D result of integrating `data` along `integration_axis`.
        """
        # 1) Integrate data along the chosen axis.
        integrated_data = np.mean(data, axis=integration_axis)
        axis_data = np.mean(axis_data, axis=integration_axis)
        if scaling == "log":
            integrated_data = np.log10(integrated_data)
        # 2) Figure out shapes for plotting. integrated_data is now 1D: shape (N,).
        #    We'll do a line plot: x-axis vs. integrated_data.
        if integrated_data.ndim == 1:
            # If axis_data is given, pick the matching dimension
            if axis_data is not None:
                # If integration_axis=0 => we are left with dimension 1 => shape (Ny,).
                # If integration_axis=1 => we are left with dimension 0 => shape (Nx,).
                x_axis = axis_data
            else:
                # Fallback: integer indices
                x_axis = np.arange(integrated_data.size)

            # 1D line plot
            fig, ax = self.line_plot(
                x_axis,
                integrated_data,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                figsize=figsize
            )
            plt.show()

        else:
            # If by design, you'd prefer no summation => data remains 2D => surface plot
            # For standard usage in this docstring, we always get 1D after summation,
            # but here's a fallback for a 2D plot if that case arises:
            if axis_data is not None:
                X_vals = axis_data[0]
                Y_vals = axis_data[1]
            else:
                X_vals = np.arange(data.shape[0])
                Y_vals = np.arange(data.shape[1])

            Xmesh, Ymesh = np.meshgrid(X_vals, Y_vals, indexing='ij')
            fig, ax = self.surf_plot(
                Xmesh,
                Ymesh,
                integrated_data,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel="Integrated Intensity",
                figsize=figsize
            )
            plt.show()

        return integrated_data
