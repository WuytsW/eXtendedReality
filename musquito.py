import json
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected with result code " + str(rc))
    client.subscribe("XRCatAndMouse/1111")

def on_message(client, userdata, msg, properties=None):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        print(f"Received: {payload}")
    except json.JSONDecodeError:
        print(f"Received invalid JSON: {msg.payload.decode('utf-8')}")

client = mqtt.Client(protocol=mqtt.MQTTv311)  # Explicitly set protocol version
client.on_connect = on_connect
client.on_message = on_message

client.connect("broker.hivemq.com", 1883, 60)
client.loop_forever()