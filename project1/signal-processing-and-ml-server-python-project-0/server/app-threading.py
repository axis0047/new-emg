import socket
from scipy import signal
import numpy as np
import pandas as pd
import time
import requests
import math
import random
import threading
from collections import deque
from keras.models import save_model, load_model

from counts import count_files_in_folder, check_file_exists
from sendbluetooth import send_bluetooth_message, bluetooth_init,  bluetooth_close
from craete_spectros import spectrogram_generator_cwt, spectrogram_generator_cwt_predict
from dataaqq import record_data, record_data_by_len
from train_model import train
from image_preprocessing import preprocess_image
from process_handling import find_pid_using_port, kill_process, is_port_in_use


def udp_server():

    # Replace 'XX:XX:XX:XX:XX:XX' with the Bluetooth address of your target device
    # target_device_address = '00:00:13:10:44:B0'
    target_device_address = '00:21:13:00:6A:0C'


    # Create a UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific address and port
    port = 5555

    if is_port_in_use(port):
        pid = find_pid_using_port(port)
        if pid:
            kill_process(pid)
            
    server_address = ('127.0.0.1', port)  # Listen on all available interfaces
    server_socket.bind(server_address)

    print('UDP server is listening on {}:{}'.format(*server_address))

    def generate_data_stream(server_socket):
        while True:
            data, server_address = server_socket.recvfrom(10000)
            reconstructed_array = np.frombuffer(data, dtype=np.int64)
            yield reconstructed_array

    def sliding_window(big_array, model, bluetooth_socket, lock):
        while True:
            time.sleep(0.05)
            lock.acquire()
            window_data = list(big_array)[:3000]
            lock.release()

            print("WINDOW DATA", window_data)

            data_to_predict = f'/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/y/{spectrogram_generator_cwt_predict(window_data, "y")}.jpg'

            print("DATA TO PREDICT", data_to_predict)

            image_to_predict = preprocess_image(data_to_predict)

            prediction = model.predict(np.array([image_to_predict]))

            result = np.argmax(prediction)

            print("PREDICTION", result)

            send_bluetooth_message(bluetooth_socket, str(result))
            time.sleep(0.3)

    def data_reception(server_socket, big_array, lock):
        while True:
            data_stream = generate_data_stream(server_socket)
            for data in data_stream:
                lock.acquire()
                big_array.extend(data)
                if len(big_array) > 4000:
                    big_array = big_array[-4000:]
                lock.release()


    test_name = input("Enter a Test Name: ")

    if(check_file_exists(f'/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/models/{test_name}.h5')):
        print("Model Found. Start predictions")

        model_filename = f'/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/models/{test_name}.h5'
        loaded_model = load_model(model_filename)

        print(loaded_model.summary())

        bluetooth_socket = bluetooth_init(target_device_address, 1)

        
        while True: 

            big_array = deque(maxlen=4000)
            lock = threading.Lock()

            # Create and start the thread for data reception
            reception_thread = threading.Thread(target=data_reception, args=(server_socket, big_array, lock))
            reception_thread.start()

            # Create and start the thread for sliding window
            window_thread = threading.Thread(target=sliding_window, args=(big_array, loaded_model, bluetooth_socket, lock))
            window_thread.start()

            try:
                # Keep the main thread running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                # Terminate threads gracefully on Ctrl+C
                reception_thread.join()
                window_thread.join()

    else:
        print("Start Training")

    time.sleep(3)   

    try:
        
        count = 0

        while True:

            # Receive data and address from client
            print("Aquiring mode")

            data_cap = record_data_by_len(server_socket, 3000)

            print("\n\n\nRecording\n\n\n")

            count = count + 1

            data_raw = np.array(data_cap).flatten()

            if count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/0') == 0:
                print('''
                       _____                  ______               
                        |_   _|                |___  /               
                        | |_   _ _ __   ___     / /  ___ _ __ ___  
                        | | | | | '_ \ / _ \   / /  / _ \ '__/ _ \ 
                        | | |_| | |_) |  __/ ./ /__|  __/ | | (_) |
                        \_/\__, | .__/ \___| \_____/\___|_|  \___/ 
                            __/ | |                                
                            |___/|_|                                
                      ''')
                time.sleep(3)

            elif count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/0') == 30 and count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/1') == 0 and count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/2') == 0:
                print('''
                         _____                  _____            
                        |_   _|                |  _  |           
                        | |_   _ _ __   ___  | | | |_ __   ___ 
                        | | | | | '_ \ / _ \ | | | | '_ \ / _ \
                        | | |_| | |_) |  __/ \ \_/ / | | |  __/
                        \_/\__, | .__/ \___|  \___/|_| |_|\___|
                            __/ | |                            
                            |___/|_|                            
                    ''')
                time.sleep(3)

            elif count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/1') == 30 and count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/2') == 0:
                print('''
                         _____                  _____             
                        |_   _|                |_   _|            
                        | |_   _ _ __   ___    | |_      _____  
                        | | | | | '_ \ / _ \   | \ \ /\ / / _ \ 
                        | | |_| | |_) |  __/   | |\ V  V / (_) |
                        \_/\__, | .__/ \___|   \_/ \_/\_/ \___/ 
                            __/ | |                             
                            |___/|_|                             
                    ''')
                time.sleep(3)

            elif count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/2') == 30:
                print('''

                     _____                  _____ _                   
                    |_   _|                |_   _| |                  
                    | |_   _ _ __   ___    | | | |__  _ __ ___  ___ 
                    | | | | | '_ \ / _ \   | | | '_ \| '__/ _ \/ _ \
                    | | |_| | |_) |  __/   | | | | | | | |  __/  __/
                    \_/\__, | .__/ \___|   \_/ |_| |_|_|  \___|\___|
                        __/ | |                                     
                        |___/|_|                                     
                    ''')
                time.sleep(3)

            if count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/0') < 30:
                spectrogram_generator_cwt(data_raw, 0)
            elif count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/0') >= 30 and count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/1') < 30:
                spectrogram_generator_cwt(data_raw, 1)
            elif count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/2') < 30 and count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/1') >= 30:
                spectrogram_generator_cwt(data_raw, 2)
            elif count_files_in_folder('/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/spectrograms/3') < 30:
                spectrogram_generator_cwt(data_raw, 3)
            else:
                break
        
        time.sleep(5)

        model = train()

        save_model(model, f'/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/models/{test_name}.h5')

        time.sleep(3)

        bluetooth_socket = bluetooth_init(target_device_address, 1)

        while True: 

            big_array = deque(maxlen=4000)
            lock = threading.Lock()

            # Create and start the thread for data reception
            reception_thread = threading.Thread(target=data_reception, args=(server_socket, big_array, lock))
            reception_thread.start()

            # Create and start the thread for sliding window
            window_thread = threading.Thread(target=sliding_window, args=(big_array, loaded_model, bluetooth_socket, lock))
            window_thread.start()

            try:
                # Keep the main thread running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                # Terminate threads gracefully on Ctrl+C
                reception_thread.join()
                window_thread.join()
                
    except KeyboardInterrupt:
        # Handle Ctrl+C to stop capturing
        pass
                        
    finally:
        print("Server stopped.")
        bluetooth_close(bluetooth_socket)
        server_socket.close()


if __name__ == '__main__':
    
    ws_server_url = 'http://127.0.0.1:4000'
    headers = {'Content-Type': 'application/json'}
    
    udp_server()


   