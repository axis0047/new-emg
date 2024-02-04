import time
import numpy as np

# def record_data(server_socket,capture_duration = 2.7):


#     # Record start time
#     start_time = time.time()
#     print("TIME",start_time)

#     try:

#         # Store captured data
#         captured_data = []

#         while (time.time() - start_time) < capture_duration:
#             # Read data from the serial port
#             data, server_address = server_socket.recvfrom(10000)
#             # print(data)
            
#             print("recived")
#             print("lendata", len(data))
#             # Reconstruct NumPy array from bytes
#             reconstructed_array = np.frombuffer(data, dtype=np.int64)

#             print(len(reconstructed_array))

#             captured_data.append(reconstructed_array)

#             print(len(captured_data))
        
#         return captured_data

#     except KeyboardInterrupt:
#         # Handle Ctrl+C to stop capturing
#         pass
#     finally:
#         # Close the serial port
#         print("Captured ..")


def record_data(server_socket, capture_duration=2.7):
    start_time = time.time()

    try:
        captured_data = []

        while (time.time() - start_time) < capture_duration:
            remaining_time = start_time + capture_duration - time.time()
            if remaining_time > 0:
                time.sleep(remaining_time)

            data, server_address = server_socket.recvfrom(10000)
            reconstructed_array = np.frombuffer(data, dtype=np.int64)

            captured_data.append(reconstructed_array)

        print("TIMEEEEEE", time.time())
        return captured_data
    except KeyboardInterrupt:
        pass
    finally:
        print("Captured ..")

# def record_data_by_len(server_socket, capture_len=3000):

#     try:
#         captured_data = []

#         if len(captured_data)<capture_len:
            

#             data, server_address = server_socket.recvfrom(10000)
#             reconstructed_array = np.frombuffer(data, dtype=np.int64)

#             captured_data.append(reconstructed_array)

#         if len(captured_data) == capture_len:
#             print("Captured ..")
#             print("TIMEEEEEE", time.time())
#             return captured_data
#     except KeyboardInterrupt:
#         pass


def record_data_by_len(server_socket, capture_len=3000):

    try:
        captured_data = []
        while len(captured_data*1000) < capture_len:
            print("ITTRRRR", len(captured_data))
            data, server_address = server_socket.recvfrom(10000)
            reconstructed_array = np.frombuffer(data, dtype=np.int64)
            captured_data.append(reconstructed_array)

        flattened_data = np.concatenate(captured_data).ravel()  # Flatten the captured data

        print("Captured ..")
        print("TIMEEEEEE", time.time())
        return flattened_data
    
    except KeyboardInterrupt:
        print("Keyboard Interrupt. Stopping data capture.")
        pass