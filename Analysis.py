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
        """
        Initialize the analysis object.

        Parameters
        ----------
        directory : str, optional
            Path to the directory where analysis results will be stored.
            Defaults to the current working directory.
        """
        self.directory = directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    # Static Function
    @staticmethod
    def surf_plot(X,Y,Z,title,xlabel="Frequency (1/px)",ylabel="Distance",zlabel="FFT Amplitude",figsize=(12, 12)):
        """
        Generate a 3D surface plot of Z as a function of X and Y.

        Parameters
        ----------
        X : ndarray
            2D array of x-coordinates.
        Y : ndarray
            2D array of y-coordinates.
        Z : ndarray
            2D array of z-values (the surface height).
        title : str
            Title for the plot.
        xlabel : str, optional
            Label for the X-axis. Default is "Frequency (1/px)".
        ylabel : str, optional
            Label for the Y-axis. Default is "Distance".
        zlabel : str, optional
            Label for the Z-axis. Default is "FFT Amplitude".
        figsize : tuple, optional
            Figure size. Default is (12, 12).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The 3D axes containing the surface plot.
        """
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
        """
        Generate a 2D line plot of y vs x.

        Parameters
        ----------
        x : ndarray
            1D array for the X-axis.
        y : ndarray
            1D array for the Y-axis.
        title : str
            Title for the plot.
        xlabel : str, optional
            Label for the X-axis. Default is "Detector X".
        ylabel : str, optional
            Label for the Y-axis. Default is "Pixel Value".
        figsize : tuple, optional
            Figure size. Default is (12, 12).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes._subplots.AxesSubplot
            The 2D axes containing the line plot.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        plot = ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # fig1.show()
        return fig,ax

    ## Main Functions
    def distance_fft_dependance(self,sample,beam,stage,detector,distance_array,plot_prefix="Test",output_pixel_values=False,offset_list=None):
        """
        Computes amplitude/phase summations and their FFTs at various detector
        distances, plots them, and saves all generated figures to self.directory.
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
    
    def integrate_detector_along_axis(self,detector,data_type="Intensity",axis="x",system="cartesian",degrees=True,bins=200,aggregator="mean",plot=True,title="Integrated Detector Data",xlabel=None,ylabel="Integrated Value",figsize=(8, 6)):
        """
        Integrate detector data along a chosen axis (x, y in Cartesian or
        2theta, eta in angular coordinates) by binning and summation/averaging.

        Parameters
        ----------
        detector : detector
            The detector object containing pixel data and geometry.
        data_type : str, optional
            Which detector quantity to integrate. Must be one of:
            "Intensity", "Amplitude", or "Phase".
        axis : str, optional
            The axis along which to integrate. For `system="cartesian"`,
            choose from ["x", "y", "z"]. For `system="angular"`,
            choose from ["eta", "2theta"].
            Default is "x".
        system : str, optional
            Coordinate system to use: "cartesian" or "angular".
            If "angular", the returned angles can be in degrees or radians.
            Default is "cartesian".
        degrees : bool, optional
            If `system="angular"`, whether to convert angles to degrees.
            Default is True. If False, angles are in radians.
        bins : int, optional
            Number of bins for histogram integration. Default is 200.
        aggregator : str, optional
            How to combine pixel values within each bin:
            "sum" or "mean". Default is "sum".
        plot : bool, optional
            If True, produce a line plot of the integrated data vs. the chosen axis.
            Default is True.
        title : str, optional
            Plot title if `plot=True`.
        xlabel : str, optional
            X-axis label for the plot. If not provided, one is auto-generated.
        ylabel : str, optional
            Y-axis label for the plot. Default is "Integrated Value".
        figsize : tuple, optional
            Figure size for the plot. Default is (8, 6).

        Returns
        -------
        bin_centers : ndarray, shape (bins,)
            The midpoints of each bin along the chosen axis.
        integrated_vals : ndarray, shape (bins,)
            The integrated or averaged detector data values corresponding
            to each bin.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # -------------------------------------------------------------------------
        # 1) Get the requested pixel data (Intensity, Amplitude, or Phase).
        # -------------------------------------------------------------------------
        if data_type.lower() == "intensity":
            data_array = detector.pixel_intensity
        elif data_type.lower() == "amplitude":
            data_array = detector.pixel_amplitude
        elif data_type.lower() == "phase":
            data_array = detector.pixel_phase
        else:
            raise ValueError(f"Unknown data_type '{data_type}'; choose from Intensity, Amplitude, Phase.")

        # Flatten into 1D
        data_vals = data_array.ravel()

        # -------------------------------------------------------------------------
        # 2) Get the appropriate coordinate array: either cartesian or angular.
        #    detector.get_detector_axis(...) returns shape = (3, Nx, Ny).
        #       - "cartesian": coords = [ x, y, z ]
        #       - "angular" : coords = [ eta, two_theta, distance ]  (in radians by default)
        # -------------------------------------------------------------------------
        if system.lower() not in ["cartesian", "angular"]:
            raise ValueError(f"system must be 'cartesian' or 'angular', got '{system}'.")

        coords_3xN = detector.get_detector_axis(system=system, units="deg" if degrees else "rad")
        # coords_3xN is shape (3, Ny, Nx). Flatten to shape (3, Nx*Ny).
        coords_3xN = coords_3xN.reshape(3, -1)

        # Decide which row in coords_3xN is the "axis" dimension we care about.
        if system.lower() == "cartesian":
            axis = axis.lower()
            if axis not in ["x", "y", "z"]:
                raise ValueError("For system='cartesian', axis must be one of ['x', 'y', 'z'].")
            axis_idx_map = {"x": 0, "y": 1, "z": 2}
            coord_vals = coords_3xN[axis_idx_map[axis], :]
            if xlabel is None:
                xlabel = f"{axis.upper()} (mm or Å)"
        else:
            # 'angular' system
            axis = axis.lower()
            # coords_3xN: index 0=eta, 1=2theta, 2=distance
            if axis == "eta":
                coord_vals = coords_3xN[0, :]
                if xlabel is None:
                    xlabel = r"$\eta$ ({}{})".format("°" if degrees else "rad", "")
            elif axis in ["2theta", "2θ"]:
                coord_vals = coords_3xN[1, :]
                if xlabel is None:
                    xlabel = r"$2\theta$ ({}{})".format("°" if degrees else "rad", "")
            else:
                raise ValueError("For system='angular', axis must be one of ['eta', '2theta'].")

        # Flatten coordinate array too
        coord_vals = coord_vals.ravel()

        # -------------------------------------------------------------------------
        # 3) Bin and aggregate (sum or mean) across this axis.
        # -------------------------------------------------------------------------
        # Create the bin edges from min to max of the chosen coordinate.
        # Using np.histogram with weights to accumulate sums,
        # then dividing by counts for mean if aggregator='mean'.
        bin_edges = np.linspace(coord_vals.min(), coord_vals.max(), bins + 1)

        hist_sums, _ = np.histogram(coord_vals, bins=bin_edges, weights=data_vals)
        hist_counts, _ = np.histogram(coord_vals, bins=bin_edges)

        if aggregator.lower() == "sum":
            integrated_vals = hist_sums
        elif aggregator.lower() == "mean":
            # Avoid divide-by-zero for empty bins
            with np.errstate(divide='ignore', invalid='ignore'):
                integrated_vals = np.where(hist_counts > 0, hist_sums / hist_counts, 0.0)
        else:
            raise ValueError(f"Unknown aggregator '{aggregator}'; choose 'sum' or 'mean'.")

        # Compute bin centers for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # -------------------------------------------------------------------------
        # 4) Optional: Plot the integrated curve
        # -------------------------------------------------------------------------
        if plot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot(bin_centers, integrated_vals, marker="o", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            plt.show()

        return bin_centers, integrated_vals

