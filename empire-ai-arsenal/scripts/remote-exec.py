#!/usr/bin/env python3
"""
Remote execution helper for VPS deployment.
Usage: python remote-exec.py <command>
"""
import paramiko
import sys
import time

VPS_IP = "89.116.29.33"
VPS_USER = "root"
VPS_PASS = "Ashlynn71c88g"

def ssh_exec(command, timeout=300, stream=True):
    """Execute command on VPS and return output."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_IP, username=VPS_USER, password=VPS_PASS, timeout=15)

    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)

    if stream:
        output = ""
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                chunk = stdout.channel.recv(4096).decode()
                print(chunk, end="", flush=True)
                output += chunk
            time.sleep(0.1)
        # Get remaining output
        remaining = stdout.read().decode()
        print(remaining, end="", flush=True)
        output += remaining
        errors = stderr.read().decode()
        if errors:
            print(f"\nSTDERR: {errors}", flush=True)
    else:
        output = stdout.read().decode()
        errors = stderr.read().decode()

    exit_code = stdout.channel.recv_exit_status()
    ssh.close()
    return output, errors, exit_code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remote-exec.py '<command>'")
        sys.exit(1)
    cmd = " ".join(sys.argv[1:])
    output, errors, code = ssh_exec(cmd)
    sys.exit(code)
