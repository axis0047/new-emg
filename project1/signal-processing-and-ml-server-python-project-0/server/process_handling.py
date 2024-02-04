import os
import signal
import subprocess
import socket


def find_pid_using_port(port):
    try:
        # Run the lsof command to find the PID using the specified port
        result = subprocess.run(['lsof', '-t', '-i', f'tcp:{port}'], capture_output=True, text=True)
        pid = int(result.stdout.strip())
        return pid
    except Exception as e:
        print(f"Error finding PID: {e}")
        return None

def kill_process(pid):
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Process with PID {pid} killed successfully.")
    except OSError as e:
        print(f"Error: {e}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
        except OSError:
            return True
        return False

