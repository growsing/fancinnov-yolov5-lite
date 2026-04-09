import serial
import time

ser = serial.Serial('/dev/ttyAMA0', 460800, timeout=1.0)
print("Recording raw data for 10 seconds...")

data_buffer = b''
start = time.time()

while time.time() - start < 10:
    if ser.in_waiting:
        data_buffer += ser.read(ser.in_waiting)

ser.close()

print(f"\nTotal bytes: {len(data_buffer)}")
print("Hex output:")
print(data_buffer.hex())
print("\nASCII output (printable only):")
print(data_buffer.decode('ascii', errors='ignore'))