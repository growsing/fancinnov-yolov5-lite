import serial
import time

print("Testing serial port...")

try:
    ser = serial.Serial('/dev/ttyAMA0', 460800, timeout=1.0)
    print(f"✓ Serial port opened successfully")
    print(f"  - Is open: {ser.is_open}")
    print(f"  - Baudrate: {ser.baudrate}")
    print(f"  - Bytes waiting: {ser.in_waiting}")
    
    print("\nWaiting for data (10 seconds)...")
    print("Make sure flight controller is powered on and connected")
    
    start_time = time.time()
    bytes_read = 0
    
    while time.time() - start_time < 10:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            bytes_read += len(data)
            print(f"\rReceived {bytes_read} bytes", end='')
            if bytes_read > 0 and bytes_read <= 20:
                print(f"\nFirst few bytes: {data[:20].hex()}")
        time.sleep(0.1)
    
    print(f"\n\n{'✓' if bytes_read > 0 else '✗'} Total received: {bytes_read} bytes")
    
    if bytes_read == 0:
        print("\n❌ No data received. Possible issues:")
        print("  1. Flight controller not powered on")
        print("  2. TX/RX pins not connected correctly")
        print("  3. Flight controller not sending MAVLink on this port")
        print("  4. Baud rate mismatch")
        print("\nCheck flight controller connections:")
        print("  - Pi GPIO14 (TX) -> FC RX")
        print("  - Pi GPIO15 (RX) -> FC TX")
        print("  - Common ground")
    
    ser.close()
    
except Exception as e:
    print(f"✗ Error: {e}")
