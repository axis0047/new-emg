'''spectrogram creation'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import math

import random
import string

import logging

import os

import time as tt

def spectrogram_generator_cwt(signal_array,action_class):

    dataset_folder_path = '/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/'


    try:
        print("Spectrogram generator")

            #data / define
        wavelete_scales = [11.4591559, 11.38301533, 11.30787992, 11.23372988, 11.16054598, 11.08830943, 11.01700198, 10.9466058,
            10.87710354, 10.80847828, 10.74071353, 10.67379319, 10.60770159, 10.54242343, 10.47794378, 10.41424807,
            10.3513221, 10.289152, 10.22772422, 10.16702556, 10.10704311, 10.04776427, 9.98917672, 9.93126845,
            9.87402771, 9.81744302, 9.76150318, 9.70619721, 9.65151441, 9.5974443, 9.54397664, 9.49110143,
            9.43880886, 9.38708936, 9.33593356, 9.28533229, 9.23527659, 9.18575768, 9.13676697, 9.08829606,
            9.04033671, 8.99288088, 8.94592067, 8.89944835, 8.85345637, 8.80793731, 8.76288393, 8.7182891,
            8.67414586, 8.63044739, 8.587187, 8.54435814, 8.50195438, 8.45996942, 8.41839709, 8.37723133,
            8.33646622, 8.29609592, 8.25611473, 8.21651706, 8.17729741, 8.13845039, 8.09997072, 8.06185321,
            8.02409277, 7.98668442, 7.94962324, 7.91290442, 7.87652325, 7.84047509, 7.80475539, 7.76935967,
            7.73428356, 7.69952273, 7.66507296, 7.6309301, 7.59709006, 7.56354882, 7.53030245, 7.49734708,
            7.4646789, 7.43229418, 7.40018923, 7.36836046, 7.33680431, 7.3055173, 7.274496, 7.24373703,
            7.21323708, 7.1829929, 7.15300128, 7.12325907, 7.09376318, 7.06451055, 7.03549818, 7.00672314,
            6.97818252, 6.94987346, 6.92179316, 6.89393886, 6.86630785, 6.83889743, 6.811705, 6.78472795,
            6.75796374, 6.73140985, 6.70506383, 6.67892323, 6.65298566, 6.62724877, 6.60171024, 6.57636778,
            6.55121915, 6.52626212, 6.50149453, 6.47691421, 6.45251905, 6.42830697, 6.40427592, 6.38042386,
            6.35674882, 6.33324883, 6.30992194, 6.28676627, 6.26377992, 6.24096105, 6.21830783, 6.19581847,
            6.1734912, 6.15132426, 6.12931595, 6.10746455, 6.08576841, 6.06422587, 6.0428353, 6.02159511,
            6.0005037, 5.97955954, 5.95876107, 5.93810678, 5.91759519, 5.89722481, 5.87699419, 5.85690191,
            5.83694653, 5.81712668, 5.79744097, 5.77788805, 5.75846658, 5.73917523, 5.72001271, 5.70097773,
            5.68206901, 5.66328531, 5.6446254, 5.62608804, 5.60767204, 5.58937621, 5.57119937, 5.55314038,
            5.53519809, 5.51737136, 5.49965909, 5.48206018, 5.46457355, 5.44719812, 5.42993283, 5.41277664,
            5.39572853, 5.37878746, 5.36195245, 5.34522249, 5.3285966, 5.31207382, 5.29565319, 5.27933377,
            5.26311462, 5.24699482, 5.23097346, 5.21504964, 5.19922248, 5.1834911, 5.16785462, 5.1523122,
            5.13686299, 5.12150615, 5.10624086, 5.09106629, 5.07598165, 5.06098614, 5.04607896, 5.03125935,
            5.01652652, 5.00187973, 4.98731822, 4.97284124, 4.95844807, 4.94413797, 4.92991024, 4.91576415,
            4.90169902, 4.88771414, 4.87380884, 4.85998243, 4.84623425, 4.83256363, 4.81896992, 4.80545248,
            4.79201065, 4.77864381, 4.76535134, 4.75213261, 4.73898702, 4.72591395, 4.71291281, 4.69998301,
            4.68712396, 4.67433508, 4.6616158, 4.64896556, 4.63638378, 4.62386993, 4.61142344, 4.59904378,
            4.58673041, 4.5744828, 4.56230042, 4.55018276, 4.53812929, 4.52613952, 4.51421293, 4.50234903,
            4.49054733, 4.47880734, 4.46712857, 4.45551055, 4.44395281, 4.43245487, 4.42101628, 4.40963657,
            4.3983153, 4.387052, 4.37584625, 4.3646976, 4.35360561, 4.34256985, 4.33158991, 4.32066534,
            4.30979574, 4.2989807, 4.28821979, 4.27751263, 4.2668588, 4.25625791, 4.24570956, 4.23521337,
            4.22476895, 4.21437591, 4.20403388, 4.19374249, 4.18350136, 4.17331013, 4.16316843, 4.1530759,
            4.14303218, 4.13303693, 4.12308979, 4.11319041, 4.10333846, 4.09353359, 4.08377546, 4.07406375,
            4.06439812, 4.05477824, 4.0452038, 4.03567446, 4.02618991, 4.01674984, 4.00735394, 3.99800188,
            3.98869338, 3.97942812, 3.97020581, 3.96102614, 3.95188883, 3.94279357, 3.93374009, 3.92472808,
            3.91575727, 3.90682738, 3.89793813, 3.88908923, 3.88028042, 3.87151143, 3.86278198, 3.85409181,
            3.84544065, 3.83682824, 3.82825432, 3.81971863]
        
        def plot_signal(time, signal):
            plt.plot(time, signal, label='ADC Signal')
            plt.title('ADC Signal')
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.show()

        # def resample_signal(input_signal, target_fs):
        #     """
        #     Resample the input signal to the target sampling rate.
        #     """
        #     current_fs = 1000.0 / (input_signal[1, 0] - input_signal[0, 0])
        #     resampled_signal = signal.resample(input_signal[:, 1], int(len(input_signal) * target_fs / current_fs))
        #     resampled_time = np.linspace(input_signal[0, 0], input_signal[-1, 0], len(resampled_signal))
        #     return np.column_stack((resampled_time, resampled_signal))
            
        # def resample_signal(input_signal, target_fs):
        #     """
        #     Resample the input signal to the target sampling rate.
        #     """
        #     current_fs = 1000.0 / (input_signal[1, 0] - input_signal[0, 0])
        #     resampled_length = int(len(input_signal) * target_fs / current_fs)
            
        #     resampled_signal = signal.resample(input_signal[:, 1], resampled_length)
        #     resampled_time = np.linspace(input_signal[0, 0], input_signal[-1, 0], len(resampled_signal))
            
        #     return np.column_stack((resampled_time, resampled_signal))
            
        # def resample_signal(input_signal, target_fs):
        #     """
        #     Resample the input signal to the target sampling rate.
        #     """
        #     # Extract time and signal values
        #     input_time = input_signal[:, 0]
        #     input_values = input_signal[:, 1]

        #     print("after time thing")

        #     # Calculate the current sampling frequency
        #     current_fs = 1.0 / np.mean(np.diff(input_time))
            
        #     print("after fs")

        #     # Resample signal
        #     resampled_length = int(len(input_values) * target_fs / current_fs)
        #     print("___")
        #     resampled_signal = signal.resample(input_values, resampled_length)

        #     print("after resample len")
            
        #     # Resample time
        #     resampled_time = np.linspace(input_time[0], input_time[-1], len(resampled_signal))

        #     print("after resample time")

        #     # Stack time and resampled signal
        #     resampled_data = np.column_stack((resampled_time, resampled_signal))

        #     print("after resample data")

        #     return resampled_data
            
        def adc_to_voltage_signal(signal_array):
            adc_values = signal_array[:,1]
            V_min = -20*(10**(-6))  # Minimum voltage in volts
            V_max = 20*(10**(-6))   # Maximum voltage in volts
            adc_signal =  V_min + ((V_max - V_min) / 1023) * adc_values
            return_signal = np.column_stack((signal_array[:, 0], adc_signal))
            return return_signal
            
        def resample_signal(input_signal, target_fs):
            """
            Resample the input signal to the target sampling rate.
            """
            current_fs = 1000.0 / (input_signal[1, 0] - input_signal[0, 0])
            resampled_signal = signal.resample(input_signal[:, 1], int(len(input_signal) * target_fs / current_fs))
            resampled_time = np.linspace(input_signal[0, 0], input_signal[-1, 0], len(resampled_signal))
            return np.column_stack((resampled_time, resampled_signal))            
        
        def apply_bandpass_filter(input_signal, lowcut, highcut, fs):
            """
            Apply a bandpass filter to the input signal.

            Parameters:
            - input_signal: NumPy array with shape (n, 2) where the first column represents time
                            and the second column represents the signal values.
            - lowcut: Lower cutoff frequency of the bandpass filter.
            - highcut: Upper cutoff frequency of the bandpass filter.
            - fs: Sampling frequency of the input signal.

            Returns:
            - filtered_signal: NumPy array with shape (n, 2) containing the filtered signal.
            """
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal_values = signal.filtfilt(b, a, input_signal[:, 1])
            return np.column_stack((input_signal[:, 0], filtered_signal_values))
        
        def apply_notch_filter(input_signal, notch_freq, q, fs):
            """
            Apply a notch filter to the input signal.

            Parameters:
            - signal: NumPy array with shape (n, 2) where the first column represents time
                    and the second column represents the signal values.
            - notch_freq: Frequency of the notch filter in Hertz.
            - q: Quality factor of the notch filter.

            Returns:
            - filtered_signal: NumPy array with shape (n, 2) containing the filtered signal.
            """
            nyquist = 0.5 * fs
            freq = notch_freq / nyquist
            b, a = signal.iirnotch(freq, q)
            filtered_signal_values = signal.filtfilt(b, a, input_signal[:, 1])
            return np.column_stack((input_signal[:, 0], filtered_signal_values))
        
        def rectify(signal_data):
            """
            Rectify the signal to get the magnitude value only
            """
            rectified_signal = np.abs(signal_data)
            return rectified_signal
        
        def smooth(signal_data, window_size):
            """
            Rectify the signal to get the magnitude value only
            """
            signal_series_time_data = pd.Series(signal_data[:,0])
            signal_series_signal_data = pd.Series(signal_data[:,1])
            smoothed_signal = signal_series_signal_data.rolling(window=window_size).mean()
            smoothed_signal_series = pd.DataFrame({'time':signal_series_time_data, 'signal':smoothed_signal})
            print(smoothed_signal_series)
            smoothed_signal_array = smoothed_signal_series.values
            return smoothed_signal_array
        
        def create_continuous_wavelet_transform(signal, scales, wavelet='cmor', fs=1.0):
            """
            Create a continuous wavelet transform of the input signal.

            Parameters:
            - signal: NumPy array with shape (n, 2) where the first column represents time
                    and the second column represents the signal values.
            - scales: Array of scales at which the wavelet transform is computed.
            - wavelet: String representing the wavelet family to be used. Default is 'cmor'.
            - fs: Sampling frequency of the input signal. Default is 1.0.

            Returns:
            - time: Time values corresponding to the input signal.
            - frequencies: Frequencies corresponding to the wavelet transform.
            - cwt_matrix: Matrix containing the continuous wavelet transform coefficients.
            """

            try:
                logging.info("Starting plot_spectrogram function")
                cwt_matrix, frequencies = pywt.cwt(signal[:, 1], scales, wavelet, sampling_period=1.0/fs)
                time = signal[:, 0]
                return time, frequencies, np.abs(cwt_matrix)
            except Exception as e:
                logging.error(f"Error: {e}")

        
        def plot_spectrogram(time, frequencies, spectrogram,
                        random_string_save
                        ):
            """
            Plot the 2D spectrogram.
            """
            plt.figure(figsize=(10, 6))
            print("Fig")
            plt.imshow(spectrogram, aspect='auto', extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
                        cmap='jet', vmin=0, vmax=1*(10**(-8)))
            # plt.imshow(spectrogram, aspect='auto', extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
            #             cmap='jet')
            print("imshow")
            plt.axis('off')
            # plt.colorbar(label='Magnitude')
            # plt.title(title)
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Frequency (Hz)')
            # plt.show()

            if action_class == 0:
                output_file_path = f"{dataset_folder_path}0/{random_string_save}.jpg"
            elif action_class == 1:
                output_file_path = f"{dataset_folder_path}1/{random_string_save}.jpg"
            elif action_class == 2:
                output_file_path = f"{dataset_folder_path}2/{random_string_save}.jpg"
            elif action_class == 3:
                output_file_path = f"{dataset_folder_path}3/{random_string_save}.jpg"

            print("file pathh")
            plt.savefig(output_file_path, format='jpg', bbox_inches='tight', pad_inches=0.0)
            print("Savefig")
            # plt.colorbar(label='Magnitude')
            # plt.title(title)
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Frequency (Hz)')
            # plt.show()

        # def get_raw_signal(data, channel):
        #     if 1 <= channel <= 8:
        #         signal_array = data[['time', f'channel{channel}']].values
        #         output_signal = signal.resample(signal_array, len(signal_array))
        #         return output_signal
        #     else:
        #         print('''  _____                          ______
        #                 |  __ \                        |  ____|
        #                 | |__) |__ _ _ __   __ _  ___  | |__   _ __ _ __ ___  _ __
        #                 |  _  // _` | '_ \ / _` |/ _ \ |  __| | '__| '__/ _ \| '__|
        #                 | | \ \ (_| | | | | (_| |  __/ | |____| |  | | | (_) | |
        #                 |_|  \_\__,_|_| |_|\__, |\___| |______|_|  |_|  \___/|_|
        #                                     __/ |
        #                                     |___/                                   ''')
            
        def get_raw_signal(data):
            signal_array = data[['time', 'channel']].values
            output_signal = signal.resample(signal_array, len(signal_array))
            return output_signal
        
        def spectrogram_generator_alpha(input_signal, random_string_):
            bandpass_signal_50_150 = apply_bandpass_filter(input_signal, 50, 150, 1000)
            print("bandpass")
            notch_filtered_signal = apply_notch_filter(bandpass_signal_50_150, 60, 70, 1000)
            print("notch")
            resampled_signal = resample_signal(notch_filtered_signal, 600)
            print("resample")
            rectified_signal = rectify(resampled_signal)
            print("rectify")
            # plot_signal(rectified_signal[:,0], rectified_signal[:,1])
            time,freq,cwt_matrix = create_continuous_wavelet_transform(rectified_signal, wavelete_scales, 'cmor100-100', 600)
            print("cwt")
            plot_spectrogram(time,freq,cwt_matrix, random_string_)
            print("plot")
            return time,freq,cwt_matrix
        
        def generate_random_string(length=12):
            characters = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(characters) for _ in range(length))
            return random_string
        
        #mf scpectrogram generation

        print("SIGNAL ARRAY",signal_array)
        
        time_stamps =  np.arange(len(signal_array))

        print("AFTER TIME STAMP")

        df = pd.DataFrame({'channel':signal_array, 'time':time_stamps}, columns = ['time', 'channel'])
        print(df.head)

        print("AFTER DF")

        raw_signal = get_raw_signal(df)

        voltage_signal = adc_to_voltage_signal(raw_signal)

        print("AFTER RAW SIGNAL")

        random_string = generate_random_string()

        print("AFTER RANDOM STRING")

        plot_string = f'{action_class}_{random_string}'

        generated_spectrogram  = spectrogram_generator_alpha(voltage_signal, plot_string)

        print("AFTER SPECTROGRAM")

        print(len(generated_spectrogram[2]), action_class)

        return True


    except Exception as e:
        print(f"error: {e}")
        return False

'''
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
FUNCTION BREAK
'''

def spectrogram_generator_cwt_predict(signal_array,action_class):

    dataset_folder_path = '/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/'


    try:
        print("Spectrogram generator")

            #data / define
        wavelete_scales = [11.4591559, 11.38301533, 11.30787992, 11.23372988, 11.16054598, 11.08830943, 11.01700198, 10.9466058,
            10.87710354, 10.80847828, 10.74071353, 10.67379319, 10.60770159, 10.54242343, 10.47794378, 10.41424807,
            10.3513221, 10.289152, 10.22772422, 10.16702556, 10.10704311, 10.04776427, 9.98917672, 9.93126845,
            9.87402771, 9.81744302, 9.76150318, 9.70619721, 9.65151441, 9.5974443, 9.54397664, 9.49110143,
            9.43880886, 9.38708936, 9.33593356, 9.28533229, 9.23527659, 9.18575768, 9.13676697, 9.08829606,
            9.04033671, 8.99288088, 8.94592067, 8.89944835, 8.85345637, 8.80793731, 8.76288393, 8.7182891,
            8.67414586, 8.63044739, 8.587187, 8.54435814, 8.50195438, 8.45996942, 8.41839709, 8.37723133,
            8.33646622, 8.29609592, 8.25611473, 8.21651706, 8.17729741, 8.13845039, 8.09997072, 8.06185321,
            8.02409277, 7.98668442, 7.94962324, 7.91290442, 7.87652325, 7.84047509, 7.80475539, 7.76935967,
            7.73428356, 7.69952273, 7.66507296, 7.6309301, 7.59709006, 7.56354882, 7.53030245, 7.49734708,
            7.4646789, 7.43229418, 7.40018923, 7.36836046, 7.33680431, 7.3055173, 7.274496, 7.24373703,
            7.21323708, 7.1829929, 7.15300128, 7.12325907, 7.09376318, 7.06451055, 7.03549818, 7.00672314,
            6.97818252, 6.94987346, 6.92179316, 6.89393886, 6.86630785, 6.83889743, 6.811705, 6.78472795,
            6.75796374, 6.73140985, 6.70506383, 6.67892323, 6.65298566, 6.62724877, 6.60171024, 6.57636778,
            6.55121915, 6.52626212, 6.50149453, 6.47691421, 6.45251905, 6.42830697, 6.40427592, 6.38042386,
            6.35674882, 6.33324883, 6.30992194, 6.28676627, 6.26377992, 6.24096105, 6.21830783, 6.19581847,
            6.1734912, 6.15132426, 6.12931595, 6.10746455, 6.08576841, 6.06422587, 6.0428353, 6.02159511,
            6.0005037, 5.97955954, 5.95876107, 5.93810678, 5.91759519, 5.89722481, 5.87699419, 5.85690191,
            5.83694653, 5.81712668, 5.79744097, 5.77788805, 5.75846658, 5.73917523, 5.72001271, 5.70097773,
            5.68206901, 5.66328531, 5.6446254, 5.62608804, 5.60767204, 5.58937621, 5.57119937, 5.55314038,
            5.53519809, 5.51737136, 5.49965909, 5.48206018, 5.46457355, 5.44719812, 5.42993283, 5.41277664,
            5.39572853, 5.37878746, 5.36195245, 5.34522249, 5.3285966, 5.31207382, 5.29565319, 5.27933377,
            5.26311462, 5.24699482, 5.23097346, 5.21504964, 5.19922248, 5.1834911, 5.16785462, 5.1523122,
            5.13686299, 5.12150615, 5.10624086, 5.09106629, 5.07598165, 5.06098614, 5.04607896, 5.03125935,
            5.01652652, 5.00187973, 4.98731822, 4.97284124, 4.95844807, 4.94413797, 4.92991024, 4.91576415,
            4.90169902, 4.88771414, 4.87380884, 4.85998243, 4.84623425, 4.83256363, 4.81896992, 4.80545248,
            4.79201065, 4.77864381, 4.76535134, 4.75213261, 4.73898702, 4.72591395, 4.71291281, 4.69998301,
            4.68712396, 4.67433508, 4.6616158, 4.64896556, 4.63638378, 4.62386993, 4.61142344, 4.59904378,
            4.58673041, 4.5744828, 4.56230042, 4.55018276, 4.53812929, 4.52613952, 4.51421293, 4.50234903,
            4.49054733, 4.47880734, 4.46712857, 4.45551055, 4.44395281, 4.43245487, 4.42101628, 4.40963657,
            4.3983153, 4.387052, 4.37584625, 4.3646976, 4.35360561, 4.34256985, 4.33158991, 4.32066534,
            4.30979574, 4.2989807, 4.28821979, 4.27751263, 4.2668588, 4.25625791, 4.24570956, 4.23521337,
            4.22476895, 4.21437591, 4.20403388, 4.19374249, 4.18350136, 4.17331013, 4.16316843, 4.1530759,
            4.14303218, 4.13303693, 4.12308979, 4.11319041, 4.10333846, 4.09353359, 4.08377546, 4.07406375,
            4.06439812, 4.05477824, 4.0452038, 4.03567446, 4.02618991, 4.01674984, 4.00735394, 3.99800188,
            3.98869338, 3.97942812, 3.97020581, 3.96102614, 3.95188883, 3.94279357, 3.93374009, 3.92472808,
            3.91575727, 3.90682738, 3.89793813, 3.88908923, 3.88028042, 3.87151143, 3.86278198, 3.85409181,
            3.84544065, 3.83682824, 3.82825432, 3.81971863]
        
        def plot_signal(time, signal):
            plt.plot(time, signal, label='ADC Signal')
            plt.title('ADC Signal')
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.show()

        # def resample_signal(input_signal, target_fs):
        #     """
        #     Resample the input signal to the target sampling rate.
        #     """
        #     current_fs = 1000.0 / (input_signal[1, 0] - input_signal[0, 0])
        #     resampled_signal = signal.resample(input_signal[:, 1], int(len(input_signal) * target_fs / current_fs))
        #     resampled_time = np.linspace(input_signal[0, 0], input_signal[-1, 0], len(resampled_signal))
        #     return np.column_stack((resampled_time, resampled_signal))
            
        # def resample_signal(input_signal, target_fs):
        #     """
        #     Resample the input signal to the target sampling rate.
        #     """
        #     current_fs = 1000.0 / (input_signal[1, 0] - input_signal[0, 0])
        #     resampled_length = int(len(input_signal) * target_fs / current_fs)
            
        #     resampled_signal = signal.resample(input_signal[:, 1], resampled_length)
        #     resampled_time = np.linspace(input_signal[0, 0], input_signal[-1, 0], len(resampled_signal))
            
        #     return np.column_stack((resampled_time, resampled_signal))
            
        # def resample_signal(input_signal, target_fs):
        #     """
        #     Resample the input signal to the target sampling rate.
        #     """
        #     # Extract time and signal values
        #     input_time = input_signal[:, 0]
        #     input_values = input_signal[:, 1]

        #     # Calculate the current sampling frequency
        #     current_fs = 1.0 / np.mean(np.diff(input_time))

        #     # Resample signal
        #     resampled_length = int(len(input_values) * target_fs / current_fs)
        #     resampled_signal = signal.resample(input_values, resampled_length)

        #     print("after resample len")
            
        #     # Resample time
        #     resampled_time = np.linspace(input_time[0], input_time[-1], len(resampled_signal))

        #     print("after resample time")

        #     # Stack time and resampled signal
        #     resampled_data = np.column_stack((resampled_time, resampled_signal))

        #     print("after resample data")

        #     return resampled_data

        def adc_to_voltage_signal(signal_array):
            adc_values = signal_array[:,1]
            V_min = -20*(10**(-6))  # Minimum voltage in volts
            V_max = 20*(10**(-6))   # Maximum voltage in volts
            adc_signal =  V_min + ((V_max - V_min) / 1023) * adc_values
            return_signal = np.column_stack((signal_array[:, 0], adc_signal))
            return return_signal

        def resample_signal(input_signal, target_fs):
            """
            Resample the input signal to the target sampling rate.
            """
            current_fs = 1000.0 / (input_signal[1, 0] - input_signal[0, 0])
            resampled_signal = signal.resample(input_signal[:, 1], int(len(input_signal) * target_fs / current_fs))
            resampled_time = np.linspace(input_signal[0, 0], input_signal[-1, 0], len(resampled_signal))
            return np.column_stack((resampled_time, resampled_signal))    
        
        def apply_bandpass_filter(input_signal, lowcut, highcut, fs):
            """
            Apply a bandpass filter to the input signal.

            Parameters:
            - input_signal: NumPy array with shape (n, 2) where the first column represents time
                            and the second column represents the signal values.
            - lowcut: Lower cutoff frequency of the bandpass filter.
            - highcut: Upper cutoff frequency of the bandpass filter.
            - fs: Sampling frequency of the input signal.

            Returns:
            - filtered_signal: NumPy array with shape (n, 2) containing the filtered signal.
            """
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal_values = signal.filtfilt(b, a, input_signal[:, 1])
            return np.column_stack((input_signal[:, 0], filtered_signal_values))
        
        def apply_notch_filter(input_signal, notch_freq, q, fs):
            """
            Apply a notch filter to the input signal.

            Parameters:
            - signal: NumPy array with shape (n, 2) where the first column represents time
                    and the second column represents the signal values.
            - notch_freq: Frequency of the notch filter in Hertz.
            - q: Quality factor of the notch filter.

            Returns:
            - filtered_signal: NumPy array with shape (n, 2) containing the filtered signal.
            """
            nyquist = 0.5 * fs
            freq = notch_freq / nyquist
            b, a = signal.iirnotch(freq, q)
            filtered_signal_values = signal.filtfilt(b, a, input_signal[:, 1])
            return np.column_stack((input_signal[:, 0], filtered_signal_values))
        
        def rectify(signal_data):
            """
            Rectify the signal to get the magnitude value only
            """
            rectified_signal = np.abs(signal_data)
            return rectified_signal
        
        def smooth(signal_data, window_size):
            """
            Rectify the signal to get the magnitude value only
            """
            signal_series_time_data = pd.Series(signal_data[:,0])
            signal_series_signal_data = pd.Series(signal_data[:,1])
            smoothed_signal = signal_series_signal_data.rolling(window=window_size).mean()
            smoothed_signal_series = pd.DataFrame({'time':signal_series_time_data, 'signal':smoothed_signal})
            print(smoothed_signal_series)
            smoothed_signal_array = smoothed_signal_series.values
            return smoothed_signal_array
        
        def create_continuous_wavelet_transform(signal, scales, wavelet='cmor', fs=1.0):
            """
            Create a continuous wavelet transform of the input signal.

            Parameters:
            - signal: NumPy array with shape (n, 2) where the first column represents time
                    and the second column represents the signal values.
            - scales: Array of scales at which the wavelet transform is computed.
            - wavelet: String representing the wavelet family to be used. Default is 'cmor'.
            - fs: Sampling frequency of the input signal. Default is 1.0.

            Returns:
            - time: Time values corresponding to the input signal.
            - frequencies: Frequencies corresponding to the wavelet transform.
            - cwt_matrix: Matrix containing the continuous wavelet transform coefficients.
            """
            cwt_matrix, frequencies = pywt.cwt(signal[:, 1], scales, wavelet, sampling_period=1.0/fs)
            time = signal[:, 0]
            return time, frequencies, np.abs(cwt_matrix)
        
        def plot_spectrogram(time, frequencies, spectrogram,
                        random_string_save
                        ):
            """
            Plot the 2D spectrogram.
            """
            plt.figure(figsize=(10, 6))
            print("Fig")
            plt.imshow(spectrogram, aspect='auto', extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
                        cmap='jet', vmin=0, vmax=1*(10**(-8)))
            # plt.imshow(spectrogram, aspect='auto', extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
            #             cmap='jet')
            # print("imshow")
            plt.axis('off')

            output_file_path = f"{dataset_folder_path}y/{random_string_save}.jpg"

            print("file pathh")
            plt.savefig(output_file_path, format='jpg', bbox_inches='tight', pad_inches=0.0)
            print("Savefig")
            # plt.colorbar(label='Magnitude')
            # plt.title(title)
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Frequency (Hz)')
            # plt.show()

        # def get_raw_signal(data, channel):
        #     if 1 <= channel <= 8:
        #         signal_array = data[['time', f'channel{channel}']].values
        #         output_signal = signal.resample(signal_array, len(signal_array))
        #         return output_signal
        #     else:
        #         print('''  _____                          ______
        #                 |  __ \                        |  ____|
        #                 | |__) |__ _ _ __   __ _  ___  | |__   _ __ _ __ ___  _ __
        #                 |  _  // _` | '_ \ / _` |/ _ \ |  __| | '__| '__/ _ \| '__|
        #                 | | \ \ (_| | | | | (_| |  __/ | |____| |  | | | (_) | |
        #                 |_|  \_\__,_|_| |_|\__, |\___| |______|_|  |_|  \___/|_|
        #                                     __/ |
        #                                     |___/                                   ''')
            
        def get_raw_signal(data):
            signal_array = data[['time', 'channel']].values
            output_signal = signal.resample(signal_array, len(signal_array))
            return output_signal
        
        def spectrogram_generator_alpha(input_signal, random_string_):
            bandpass_signal_50_150 = apply_bandpass_filter(input_signal, 50, 150, 1000)
            print("bandpass")
            notch_filtered_signal = apply_notch_filter(bandpass_signal_50_150, 60, 70, 1000)
            print("notch")
            resampled_signal = resample_signal(notch_filtered_signal, 600)
            print("resample")
            rectified_signal = rectify(resampled_signal)
            print("rectify")
            # plot_signal(rectified_signal[:,0], rectified_signal[:,1])
            time,freq,cwt_matrix = create_continuous_wavelet_transform(rectified_signal, wavelete_scales, 'cmor100-100', 600)
            print("cwt")
            plot_spectrogram(time,freq,cwt_matrix, random_string_)
            print("plot")
            return time,freq,cwt_matrix
        
        def generate_random_string(length=12):
            characters = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(characters) for _ in range(length))
            return random_string
        
        #mf scpectrogram generation

        print("SIGNAL ARRAY",signal_array)
        
        time_stamps =  np.arange(len(signal_array))

        print("AFTER TIME STAMP")

        df = pd.DataFrame({'channel':signal_array, 'time':time_stamps}, columns = ['time', 'channel'])
        print(df.head)

        print("AFTER DF")

        raw_signal = get_raw_signal(df)

        voltage_signal = adc_to_voltage_signal(raw_signal)

        print("AFTER RAW SIGNAL")

        random_string = generate_random_string()

        print("AFTER RANDOM STRING")

        plot_string = f'{action_class}_{random_string}'

        generated_spectrogram  = spectrogram_generator_alpha(voltage_signal, plot_string)

        print("AFTER SPECTROGRAM")

        print(len(generated_spectrogram[2]), action_class)

        return plot_string


    except Exception as e:
        print(f"error: {e}")
        return False