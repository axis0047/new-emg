import socket
from scipy import signal
import numpy as np
import time
import requests
import math
import random
import pygame
import threading
from sendbluetooth import send_bluetooth_message 
from craete_spectros import spectrogram_generator_cwt

# Initialize pygame
pygame.init()
sound = pygame.mixer.Sound('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/beep.mp3')

def udp_server():

    toggle_variable = True

    def timer_function():
        while True:
            # Sleep for 1 second (adjust as needed)
            sound.play()
            time.sleep(2.7)
            time.sleep(0.3)

            # Toggle the variable value between True and False
            # global toggle_variable
            toggle_variable = not toggle_variable

            # Print the current value (you can replace this with your own logic)
            print(f"Variable value: {toggle_variable}")


    # Replace 'XX:XX:XX:XX:XX:XX' with the Bluetooth address of your target device
    target_device_address = '00:00:13:10:44:B0'
    # Create a UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific address and port
    server_address = ('127.0.0.1', 5555)  # Listen on all available interfaces
    server_socket.bind(server_address)

    print('UDP server is listening on {}:{}'.format(*server_address))

    try:
        collection_state = False

        while True:
            
            count_classes = 0

            # Receive data and address from client
            data, server_address = server_socket.recvfrom(1024)
            print(data)
            
            print("recived")

            # Reconstruct NumPy array from bytes
            reconstructed_array = np.frombuffer(data, dtype=np.int32)

            if collection_state == True:

                toggle_variable = False

                # Create a thread for the timer function
                timer_thread = threading.Thread(target=timer_function)
                # Start the timer thread
                timer_thread.start()

                big_buffer = np.array([])
                temp_array = np.concatenate((big_buffer,reconstructed_array), axis=None)
                big_buffer = temp_array

                count_steps = 0

                if toggle_variable == True:

                    if np.len(big_buffer) > 3*1000:
                        print("Counting normal")

                        signal_segment = big_buffer[-3000:]

                        spectrogram_generated = spectrogram_generator_cwt(signal_segment, count_classes)


                    if count_classes == 0:
                        print('class 0 -  hand rest')
                        count_steps = count_steps + 1
                        if count_steps == 9:
                            count_classes = count_classes + 1
                    elif count_classes == 1:
                        print('class 1 - ')
                        count_steps = count_steps + 1
                        if count_steps == 9:
                            count_classes = count_classes + 1
                    elif count_classes == 2:
                        print('class 2')
                        count_steps = count_steps + 1
                        if count_steps == 9:
                            count_classes = count_classes + 1
                    elif count_classes == 3:
                        print('class 3')
                        count_steps = count_steps + 1
                        if count_steps == 9:
                            count_classes = count_classes + 1
                            if count_classes == 4:
                                collection_state = False
                else:
                    print()    
                
                    
            else:
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
        # Clean up pygame
        pygame.quit()

if __name__ == '__main__':
    
    ws_server_url = 'http://127.0.0.1:4000'
    headers = {'Content-Type': 'application/json'}

    client_id = 2
    
    udp_server()