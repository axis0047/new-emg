import socket
from scipy import signal
import numpy as np

import time

import requests

import math

import random

from sendbluetooth import send_bluetooth_message 

def udp_server():
    # Replace 'XX:XX:XX:XX:XX:XX' with the Bluetooth address of your target device
    target_device_address = '00:00:13:10:44:B0'
    # Create a UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific address and port
    server_address = ('127.0.0.1', 5555)  # Listen on all available interfaces
    server_socket.bind(server_address)

    print('UDP server is listening on {}:{}'.format(*server_address))

    try:
        while True:
            # Receive data and address from client
            data, server_address = server_socket.recvfrom(1024)
            print(data)
            
            print("recived")

            # Reconstruct NumPy array from bytes
            reconstructed_array = np.frombuffer(data, dtype=np.float32)
            # print(reconstructed_array)
            
            # Process the reconstructed array
            #fake processing process
            array_sum = np.sum(reconstructed_array)
            
            decision_digit = math.floor(array_sum % 4)
            print(array_sum)
            
            '''Remove this
            after testing |'''
            
            time.sleep(0.01) 
            
            if(decision_digit < 4):
                # send_bluetooth_message(target_device_address,str(decision_digit))
                send_bluetooth_message(target_device_address,str(random.randint(0,3)))
                
    finally:
        print("Server stopped.")
        server_socket.close()

if __name__ == '__main__':
    
    ws_server_url = 'http://127.0.0.1:4000'
    headers = {'Content-Type': 'application/json'}

    client_id = 2
    
    udp_server()