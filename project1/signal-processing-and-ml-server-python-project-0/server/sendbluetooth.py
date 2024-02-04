import bluetooth
import random
import time

def bluetooth_init(target_address, port=1):
     # Create a Bluetooth socket
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        # Set the target device address and port
        sock.connect((target_address, port))
        return sock


def send_bluetooth_message(sock, message):
    try:
        # Send the message
        sock.send(message)
        print(f"Message '{message}' sent successfully to {sock.getpeername()[0]}")
    except Exception as e:
        print(f"Error: {e}")

def bluetooth_close(sock):
        # Close the socket
        sock.close()

