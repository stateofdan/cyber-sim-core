'''
Filename: mqttpubsubnode.py

Description:
Implements the MQTTPubSubNode class, a PubSubNode subclass that uses MQTT for message-based
publish/subscribe communication in the cyber simulation core. Handles connection management,
topic subscriptions, message publishing, and integration with the paho-mqtt client, providing
logging and error handling for robust distributed messaging.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
import paho.mqtt.client as mqtt
from pubsubnode.pubsubnode import PubSubNode

import logging

class MQTTPubSubNode(PubSubNode):
    def __init__(self, node_id, broker="localhost", port=1883):
        super().__init__(node_id)
        self.broker = broker
        self.port = port
        self.client = mqtt.Client(client_id=node_id, clean_session=False)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.__client_running = False
        self.client.enable_logger(self._logger)  # Enable logging for the MQTT client
        
        self._logger.info(f"[pubsub-{self.node_id}] MQTTPubSubNode initialized with broker={broker}, port={port}")

    def _on_connect(self, client, userdata, flags, rc):
        self._logger.info(f"[pubsub-{self.node_id}] Connected to MQTT broker with rc={rc}")
        socket = client.socket()
        self._logger.debug(f"[pubsub-{self.node_id}] Local socket address: {type(socket) if socket else 'N/A'}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self._logger.critical(f"[pubsub-{self.node_id}] Disconnected from MQTT broker unexpectedly, rc={rc}")
        else:
            self._logger.info(f"[pubsub-{self.node_id}] Disconnected from MQTT broker gracefully")

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        topic = msg.topic
        self._logger.info(f"[pubsub-{self.node_id}] Received message on topic '{topic}': {payload}")
        handler = self.topic_handlers.get(topic)
        if handler:
            self._logger.debug(f"[pubsub-{self.node_id}] Found handler for topic '{topic}', invoking it.")
            # Schedule the async handler
            handler(topic, payload)
        else:
            self._logger.warning(f"[pubsub-{self.node_id}] No handler found for topic '{topic}'.")

    def publish(self, topic, message):
        ret_info = self.client.publish(topic, message)
        self._logger.debug (f"[pubsub-{self.node_id}] Published message to topic '{topic}': {message}, result: {ret_info.rc}")
        if ret_info.rc != mqtt.MQTT_ERR_SUCCESS:
            self._logger.warning(f"Failed to publish message to topic '{topic}': {ret_info.rc}")

    def subscribe(self, topic):
        self._logger.info(f"[pubsub-{self.node_id}] Subscribing to topic '{topic}'")
        if not self.client.is_connected() and not self.__client_running:
            self._logger.error(f"[pubsub-{self.node_id}] Cannot subscribe, MQTT client is not connected or running.")
            raise RuntimeError("MQTT client is not connected or running.")
        self.client.subscribe(topic)

    def start(self) -> None:
        self._logger.info(f"[pubsub-{self.node_id}] Connecting to MQTT broker at {self.broker}:{self.port}")
        ret_val = self.client.connect(self.broker, self.port, keepalive=5)
        self._logger.debug (f"[pubsub-{self.node_id}] Connection result: {ret_val}")
        if ret_val != 0:
            raise ConnectionError(f"Failed to connect to MQTT broker: {ret_val}")
        self._logger.info(f"[pubsub-{self.node_id}] Starting MQTT client loop")
        self.__client_running = True
        self.client.loop_start()
        if self.topic_handlers:
            self._logger.info(f"[pubsub-{self.node_id}] topic handlers registered prior to start: subscribing to all topics {'\n'.join(self.topic_handlers.keys())}")
            for topic in self.topic_handlers.keys():
                self.subscribe(topic)

    def stop(self) -> None:
        self._logger.info(f"[pubsub-{self.node_id}] Stopping MQTT client loop")
        self.client.loop_stop()
        self.client.disconnect()
        self.__client_running = False