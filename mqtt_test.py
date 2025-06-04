'''
Filename: mqtt_test.py

Description:
Demonstrates and tests MQTT-based publish/subscribe messaging using MQTTPubSubNode and the paho-mqtt client
in the cyber simulation core. Sets up threaded nodes that exchange messages over topics, checks broker connectivity,
and logs message flow. Useful for validating MQTT infrastructure, topic handling, and node communication.


Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
# Make sure we load the logger first so we can do the logging setup for everything.
import logging
from helpers.logging_config import setup_logging
setup_logging()
main_logger = logging.getLogger(__name__)
main_logger.debug(f"Logger for {__name__} initialized: {main_logger}")

import threading
import time
from pubsubnode.mqttpubsubnode import MQTTPubSubNode
import random
import socket
import paho.mqtt.client as mqtt



class NodeThread(threading.Thread):
    def __init__(self, name, sub_topic, pub_topic):
        super().__init__()
        main_logger.debug(f"Initializing NodeThread: {name}, sub_topic: {sub_topic}, pub_topic: {pub_topic}")
        self.name = name
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.node = MQTTPubSubNode(name)
        self.node.register_handler(self.sub_topic, self.handle_message)
        self.counter = 0
        self.running = True

    def handle_message(self, topic, message):
        main_logger.info(f"[{self.name}] Received on {topic}: {message}")



    def run(self):
        main_logger.info(f"[{self.name}] Starting node with pub_topic: {self.pub_topic}, sub_topic: {self.sub_topic}")
        try:
            self.node.start()

            while self.counter < 5 and self.running:
                msg = f"Message {self.counter+1} from {self.name}"
                main_logger.info(f"[{self.name}] Publishing to {self.pub_topic}: {msg}")
                self.node.publish(self.pub_topic, msg)
                self.counter += 1
                time.sleep(random.randint(1,10))  # Random sleep to simulate variable timing

            self.stop()
        except Exception as e:
            main_logger.error(f"[{self.name}] Exception in node thread: {e}")
            

    def stop(self):
        self.running = False
        self.node.stop()
        main_logger.info(f"[{self.name}] Stopping node.")

if __name__ == "__main__":

    main_logger.info("Starting MQTT test...")
    try:
        # Check if MQTT broker is running
        with socket.create_connection(("localhost", 1883), timeout=5) as sock:
            main_logger.info("MQTT broker socket is reachable.")
            main_logger.debug(f"Socket info: {sock.getsockname()}")
            sock.close()
            main_logger.debug("Socket closed after check.")
    except Exception as e:     
        main_logger.error (f"MQTT broker is not reachable: {e}")
        quit()

    main_logger.info("MQTT broker is reachable, proceeding with simple test...")
    try:
        client = mqtt.Client(client_id="test_client")

        def on_connect(client, userdata, flags, rc):
            main_logger.info (f"Client connected with result code {rc}\nFlags: {flags}\nUserdata: {userdata}\n{client}")

        def on_disconnect(client, userdata, rc):
            main_logger.info (f"Client disconnected with result code {rc}Userdata: {userdata}\n{client}")
        
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        result = client.connect("localhost", 1883, keepalive=60)
        if result != 0:
            main_logger.error(f"Failed to connect to MQTT broker: {result}")
            quit()
        main_logger.info("MQTT client connected successfully, starting loop...")
        sock = client.socket()
        main_logger.debug(f"MQTT client socket info: {sock.getsockname() if sock else 'N/A'}")

        client.loop_start()
        main_logger.info ("loop starting now sleeping")
        time.sleep(10)  # Allow time for connection
        main_logger.info ("Disconnecting test client")
        result = client.disconnect() 
        if result != 0:
            main_logger.error(f"Failed to disconnect MQTT client: {result}")
        else:
            main_logger.info("MQTT client disconnected successfully.")
        client.loop_stop()
    except Exception as e:
        main_logger.error(f"Error during MQTT test: {e}")
        quit()
    main_logger.info("MQTT test completed successfully.")
    # Setup: node1 subscribes to topic1, publishes to topic2
    #         node2 subscribes to topic2, publishes to topic1
    node1 = NodeThread("node1", "topic1", "topic2")
    node2 = NodeThread("node2", "topic2", "topic1")

    node1.start()
    node2.start()

    node1.join()
    node2.join()

    main_logger.info("Test complete.")