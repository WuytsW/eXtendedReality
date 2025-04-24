import asyncio
from bleak import BleakScanner

TARGET_NAME = "Mi 10"
TX_POWER = -59   # Calibrated RSSI at 1 meter (adjust if needed)
N = 2.0          # Path loss exponent (2 = free space, 2.7–4 = indoors)

def estimate_distance(rssi, tx_power=TX_POWER, n=N):
    return 10 ** ((tx_power - rssi) / (10 * n))

def callback(device, adv_data):
    if device.name == TARGET_NAME:
        distance = estimate_distance(device.rssi)
        print(f"📡 {device.name} | RSSI: {device.rssi} dBm | Estimated distance: {distance:.2f} m")

async def main():
    scanner = BleakScanner()
    scanner.register_detection_callback(callback)
    await scanner.start()
    print(f"🔍 Scanning for '{TARGET_NAME}'... Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await scanner.stop()
        print("🛑 Scan stopped.")

if __name__ == "__main__":
    asyncio.run(main())
