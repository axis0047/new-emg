#### THIS IS ACTUALLY A CLIENT BTW

import socket

import bitalino
import numpy as np
import time

import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

macAddress = "/dev/rfcomm0"

def udp_client():
    # Create a UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    server_address = ('192.168.1.37', 5555)  # Replace with the public IP of the server

    try:
        while True:

            data = device.read(nframes)
                            
            if np.mean(data[:, 1]) < 1: break

            EMG = data[:, -1]
            print(EMG)

            envelope = np.mean(abs(np.diff(EMG)))

            if envelope > threshold:
                device.trigger([0, 1])
            else:
                device.trigger([0, 0])
                
            # Send the message to the server
            client_socket.sendto(EMG.tobytes(), server_address)

    finally:
        print ("STOP")
        device.trigger([0, 0])
        device.stop()
        device.close()
        

    # Close the socket
    client_socket.close()

if __name__ == '__main__':
    device = bitalino.BITalino(macAddress)
    time.sleep(1)

    srate = 1000
    nframes = 100
    threshold = 5

    device.start(srate, [0])
    print ("START")
    
    udp_client()