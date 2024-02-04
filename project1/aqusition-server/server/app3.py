import socket

from bitalino import BITalino
import numpy as np
import time

import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

def udp_client():
    # Create a UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    server_address = ('127.0.0.1', 5555)  # Replace with the public IP of the server

    try:
        while True:

            # data = device.read(nSamples)
                            
            EMG =  np.random.randint(0, 1024, 500)
            print(EMG.tobytes())
                
            # Send the message to the server
            client_socket.sendto(EMG.tobytes(), server_address)
            time.sleep(0.5)

    finally:
        
        print ("STOP")
        # Close the socket
        client_socket.close()

    
if __name__ == '__main__':

    print ("START")
    
    udp_client()