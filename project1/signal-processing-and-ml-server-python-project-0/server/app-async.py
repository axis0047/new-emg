import socket
import asyncio
import threading
from scipy import signal
import numpy as np
import time
import requests
import math

async def process_udp_data(data):
    # Reconstruct NumPy array from bytes
    reconstructed_array = np.frombuffer(data, dtype=np.float32)

    # Process the reconstructed array
    array_sum = np.sum(reconstructed_array)
    decision_digit = math.floor(array_sum % 16)

    if decision_digit < 8:
        data = {
            'client_id': client_id,
            'decision_digit': decision_digit
        }

        count = 0

        while count < 10:
            try:
                response = requests.post(ws_server_url, headers=headers, json=data)

                if response.status_code == 200:
                    # Request was successful, you can work with the response data
                    response_data = response.json()
                    print(response_data)
                    break
                else:
                    # Handle the error
                    print(f"Error: {response.status_code}")
                    print(response.text)  # Print the error response content
                    count += 1
                    print(count)
            except requests.exceptions.RequestException as e:
                print("Request failed:", e)
                count += 1
                print(count)

async def udp_server_async():
    # Create a UDP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific address and port
    server_address = ('127.0.0.1', 5555)
    server_socket.bind(server_address)

    print('UDP server is listening on {}:{}'.format(*server_address))

    try:
        while True:
            # Receive data and address from client
            data, _ = server_socket.recvfrom(1024)

            print("Received")

            # Process data asynchronously
            asyncio.ensure_future(process_udp_data(data))
    finally:
        print("Server stopped.")
        server_socket.close()

def start_udp_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start the UDP server in a separate thread
    udp_server_thread = threading.Thread(target=loop.run_until_complete, args=(udp_server_async(),))
    udp_server_thread.start()

if __name__ == '__main__':
    ws_server_url = 'http://127.0.0.1:4000'
    headers = {'Content-Type': 'application/json'}
    client_id = 2

    start_udp_server()
    # Your main program logic can continue here or spawn additional threads if needed
    # ...