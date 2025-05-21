import paho.mqtt.client as mqtt
import time
import json
import random

#MQTT broker settings,
broker = "broker.hivemq.com"  # Replace with your actual broker IP
port = 1883
topic = "catmouse/coordinates"

#Create client,
client = mqtt.Client()

#Connect to broker,
client.connect(broker, port, 60)

#Publishing loop,
try:
    cat = {"x": 0, "y": 0}
    mouse = {"x": 10, "y": 10}

    while True:
        # Simulate mouse movement
        step_size = 0.1  # Smaller step size for smoother movement
        mouse["x"] += round(random.uniform(-step_size, step_size), 2)
        mouse["y"] += round(random.uniform(-step_size, step_size), 2)

        # Create and publish the message
        message = {
            "message": "coordinates",
            "timestamp": time.time()*1000, # Current time in milliseconds
            "cat": {"x": round(cat["x"], 1), "y": round(cat["y"], 1)},
            "mouse": {"x": round(mouse["x"], 1), "y": round(mouse["y"], 1)}
        }

        json_message = json.dumps(message)
        client.publish(topic, json_message)
        print(f"Published: {json_message}")

        time.sleep(1 / 30)  # Simulate 30 frames per second


except KeyboardInterrupt:
    print("Stopped by user.")
    client.disconnect()