import socket
import threading
from bitalino import BITalino
import numpy as np
import time
import sys
from IPython.core import ultratb
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import pyqtgraph as pg

class DataUpdater(QObject):
    data_updated = pyqtSignal(object)

    def __init__(self, window):
        super(DataUpdater, self).__init__()
        self.window = window
        self.big_data_array = np.array([])

    def update_data(self, data):
        self.big_data_array = np.append(self.big_data_array, data)
        self.big_data_array = self.big_data_array[-5000:]  # Limit data array length
        self.data_updated.emit(self.big_data_array)

class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle('EMG Data')
        self.setGeometry(100, 100, 800, 600)

        # Create the central widget and the layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a pyqtgraph PlotWidget for the time domain
        self.plot_widget_time = pg.PlotWidget()
        layout.addWidget(self.plot_widget_time)

        # Create a pyqtgraph PlotWidget for the frequency domain
        self.plot_widget_freq = pg.PlotWidget()
        layout.addWidget(self.plot_widget_freq)

        # Create a simple time domain plot
        self.plot_time()

        # Create a simple frequency domain plot
        self.plot_freq()

        # Set up a timer to update the plots every second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1000)  # Update every 1000 milliseconds (1 second)

    def plot_time(self):
        # Plot some example data in the time domain
        self.x_time = np.arange(1, 5001)
        self.y_time = np.zeros(5000)

        # Plot the data on the time domain PlotWidget without symbols, only line
        self.plot_curve_time = self.plot_widget_time.plot(self.x_time, self.y_time, name='EMG Data', pen=pg.mkPen(color=(0, 0, 255)))  # Set color to blue

        # Set a fixed range for the y-axis in the time domain
        self.plot_widget_time.setYRange(0, 1024)

    def plot_freq(self):
        # Plot some example data in the frequency domain
        self.x_freq = np.fft.fftfreq(5000, d=0.1)  # Assuming a sampling rate of 1000 Hz
        self.y_freq = np.zeros(5000)

        # Plot the data on the frequency domain PlotWidget without symbols, only line
        self.plot_curve_freq = self.plot_widget_freq.plot(self.x_freq, self.y_freq, name='Frequency Domain', pen=pg.mkPen(color=(255, 0, 0)))  # Set color to red

        # Set a fixed range for the y-axis in the frequency domain
        self.plot_widget_freq.setYRange(0, 100)

    def update_plots(self):
        pass  # Update the time and frequency domain plots in this method

    def update_plots_with_data(self, data):
        # Update the time domain plot
        self.plot_curve_time.setData(self.x_time, data)

        # Update the frequency domain plot
        spectrum = np.fft.fft(data)
        self.plot_curve_freq.setData(self.x_freq, np.abs(spectrum))

def udp_client(device, window, data_updater):
    # Create a UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 5555)  # Replace with the public IP of the server

    try:
        while True:
            data = device.read(nSamples)
            EMG = data[:, -1]
            client_socket.sendto(EMG.tobytes(), server_address)
            time.sleep(0.1)
            data_updater.update_data(EMG)
    finally:
        device.trigger(digitalOutput_on)
        time.sleep(1)
        device.trigger(digitalOutput_off)
        device.stop()
        device.close()
        print("STOP")
        # Close the socket
        client_socket.close()

if __name__ == '__main__':
    macAddress = "/dev/rfcomm0"
    batteryThreshold = 10
    acqChannels = [0]
    samplingRate = 1000
    nSamples = 1000
    digitalOutput_on = [1, 1]
    digitalOutput_off = [0, 0]

    # Connect to BITalino
    device = BITalino(macAddress)
    device.battery(batteryThreshold)
    print(device.version())

    # Start Acquisition
    device.start(samplingRate, acqChannels)
    print("START")

    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()

    data_updater = DataUpdater(main_window)
    data_updater.data_updated.connect(main_window.update_plots_with_data)

    # Start UDP client in a separate thread
    udp_thread = threading.Thread(target=udp_client, args=(device, main_window, data_updater))
    udp_thread.start()

    app.exec_()
    sys.exit()
