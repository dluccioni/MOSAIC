# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import sys
sys.path.insert(1, 'X://Dresselhaus Lab/Code/Phase Retreival/Wave_Optics/waveoptics_fwrd_sim/')

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

    ## Data Handling Functions
    def write_analysis_metadata(self): #incomplete
        detector_analysis = []

    # Static Function
    @staticmethod
    def surf_plot(X,Y,Z,title,xlabel="Frequency (1/px)",ylabel="Distance",zlabel="FFT Amplitude"):
        # Now create the 3D surface plot for fig1
        fig = plt.figure(figsize=(12, 12))
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
    def line_plot(x,y,title,xlabel="Detector X",ylabel="Pixel Value"):
         # Now create the 3D surface plot for fig1
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.set_title(title)
        plot = ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # fig1.show()
        return fig,ax

    ## Main Functions   
    def distance_fft_dependance(self, sample, beam, detector, distance_array):
        freq_array = None
        fft_amplitude_list = []  # FFT amplitude matrix
        fft_phase_list = []  # FFT amplitude matrix
        # Loop over distances, compute FFT
        for idx, d in enumerate(distance_array):
            print(f"Processing distance: {d} || {idx+1}/{distance_array.size}")
            detector.position_detector_absolute(d, detector.two_theta, detector.nu)
            beam.atomic_direct_scattering(sample, detector)
            
            detector.plot_detector(type="Intensity")
            detector.plot_detector(type="Phase")
            detector.plot_detector(type="Amplitude")
            
            summed_amplitude = np.sum(detector.pixel_amplitude, axis=0)
            # summed_amplitude /= np.max(summed_amplitude)
            fft_values_amplitude = np.log(np.abs(np.fft.fft(summed_amplitude)[1:summed_amplitude.size//2])+1e-6)
            # Store in the fft_list
            fft_amplitude_list.append(fft_values_amplitude)
            
            summed_phase = np.sum(detector.pixel_phase, axis=0)
            # summed_phase /= np.max(summed_phase)
            fft_values_phase = np.log(np.abs(np.fft.fft(summed_phase)[1:summed_phase.size//2])+1e-6)
            # Store in the fft_list
            fft_phase_list.append(fft_values_phase)
            
            # Compute frequencies only once (assuming detector.pixel_size[0] is constant)
            if freq_array is None:
                freq_array = np.fft.fftfreq(summed_amplitude.size, d=1/detector.pixel_size[0])[1:summed_amplitude.size//2]
            
            # Plot 2D plot of "summed" values
            self.line_plot(np.linspace(-detector.shape[0]*detector.pixel_size[0],detector.shape[0]*detector.pixel_size[0],detector.shape[0]),summed_amplitude,"Amplitude Trace Plot")
            self.line_plot(freq_array,fft_values_amplitude,"Amplitude FFT Trace Plot")
            self.line_plot(np.linspace(-detector.shape[0]*detector.pixel_size[0],detector.shape[0]*detector.pixel_size[0],detector.shape[0]),summed_phase,"Phase Trace Plot")
            self.line_plot(freq_array,fft_values_phase,"Phase FFT Trace Plot")
            

        # Form matrix from list
        Z_amp = np.array(fft_amplitude_list)
        Z_pha = np.array(fft_phase_list)
        
        # Create a meshgrid so X is frequency, Y is distance
        X, Y = np.meshgrid(freq_array, distance_array)
        
        self.surf_plot(X,Y,Z_amp,"Combined Amplitude FFT Surface Plot")
        self.surf_plot(X,Y,Z_pha,"Combined Phase FFT Surface Plot")
        
        return X, Y, Z_amp, Z_pha 
    
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