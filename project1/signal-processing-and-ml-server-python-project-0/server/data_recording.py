import socket
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

def udp_server():

    target_device_address = '00:00:13:10:44:B0'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 5555)
    server_socket.bind(server_address)

    print('UDP server is listening on {}:{}'.format(*server_address))

    try:
        row_counter = 0
        file_counter = 1
        max_rows_per_file = 3000

        while True:
            data, server_address = server_socket.recvfrom(10000)
            reconstructed_array = np.frombuffer(data, dtype=np.int64)

            if row_counter == 0:
                # Create a new file for every 300 rows
                folder_name = '/home/axis/Documents/project1/signal-processing-and-ml-server-python-project-0/server/recorded_data'
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                file_name = os.path.join(folder_name, f'{datetime.now():%Y-%m-%d_%H-%M-%S}.csv')
                file_counter += 1
                row_counter = 1000
            else:
                row_counter += 1000

            # Create a DataFrame with the reconstructed array
            df = pd.DataFrame({'index': range(1*(row_counter/1000), len(reconstructed_array) + 1), 'data_value': reconstructed_array})

            # Append the DataFrame to the CSV file
            df.to_csv(file_name, mode='a', index=False, header=(not os.path.exists(file_name)))

            if row_counter >= max_rows_per_file:
                # Reset row counter and start a new file
                row_counter = 0

    except KeyboardInterrupt:
        pass
    finally:
        print("Server stopped.")
        server_socket.close()

if __name__ == '__main__':
    ws_server_url = 'http://127.0.0.1:4000'
    headers = {'Content-Type': 'application/json'}
    udp_server()
