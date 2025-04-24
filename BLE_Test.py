import asyncio
from bleak import BleakScanner

TARGET_NAME = "Mi 10"  # Change this to your phone's advertised name

# Basic RSSI-to-distance estimation
def estimate_distance(rssi, tx_power=-59, n=2.0):
    return 10 ** ((tx_power - rssi) / (10 * n))

async def scan():
    print("Scanning for BLE advertisements...\nPress Ctrl+C to stop.")
    while True:
        devices = await BleakScanner.discover()
        for d in devices:
            if d.name == TARGET_NAME:
                rssi = d.rssi
                distance = estimate_distance(rssi)
                print(f"📡 Found {d.name} | RSSI: {rssi} dBm | Estimated distance: {distance:.2f} m")
        await asyncio.sleep(0.2)

asyncio.run(scan())
