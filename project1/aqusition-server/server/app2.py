#### THIS IS ACTUALLY A CLIENT BTW

import socket

from bitalino import BITalino
import numpy as np
import time

import sys
from IPython.core import ultratb

from appplot import MyMainWindow,plotter 
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

def udp_client():

    big_data_array = np.array([])

    window = MyMainWindow()
    window.show()
    # Create a UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    server_address = ('127.0.0.1', 5555)  # Replace with the public IP of the server

    try:
        while True:

            data = device.read(nSamples)
                            
            EMG = data[:, -1]
            print(EMG)
            print(len(EMG))
            print(type(EMG[0]))             

            print(len(EMG.tobytes()))
            # Send the message to the server
            client_socket.sendto(EMG.tobytes(), server_address)

            big_data_array = np.append(big_data_array, EMG)
            big_data_array.flatten()
            if len(big_data_array) > 5000:
                big_data_array = big_data_array[-5000:]
            plotter(big_data_array)

    finally:
        #off the device
        device.trigger(digitalOutput_on)
        time.sleep(1)
        device.trigger(digitalOutput_off)
        device.stop()
        device.close()
        print ("STOP")
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

    print ("START")

    app = QApplication(sys.argv)
    udp_client()
    app.exec_()

    # sys.exit(app.exec_())
    