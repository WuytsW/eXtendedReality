import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

TARGET_NAME = "Mi 10"  # Replace with your device's advertised name

async def main():
    print(f"Searching for device named '{TARGET_NAME}'...")
    devices = await BleakScanner.discover(timeout=5)
    target_device = next((d for d in devices if d.name == TARGET_NAME), None)

    if not target_device:
        print(f"Device '{TARGET_NAME}' not found.")
        return

    
    while True:
        print(f"\n🔍 Attempting to connect to {TARGET_NAME} at {target_device.address}...")
        async with BleakClient(target_device.address) as client:
            if client.is_connected:
                print(f"✅ Connected to {TARGET_NAME} at {target_device.address}")
                try:
                    while True:
                        rssi = await client.get_rssi()
                        print(f"📶 RSSI: {rssi} dBm")
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("🛑 Stopping RSSI updates...")
                    break
            else:
                print("❌ Failed to connect.")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
