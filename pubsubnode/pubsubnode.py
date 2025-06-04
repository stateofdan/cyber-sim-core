'''
Filename: pubsubnode.py

Description:
Defines the abstract PubSubNode class, providing a base interface for publish/subscribe
nodes in the cyber simulation core. Specifies required methods for publishing, subscribing,
handler registration, and lifecycle management. Intended to be subclassed for specific
messaging backends (e.g., MQTT).

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
from abc import ABC, abstractmethod
from typing import Callable, Dict

import logging


HandlerType = Callable[[str, str], None] # (topic, payload)

class PubSubNode:
    def __init__(self, node_id:str):
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._logger.debug(f"Debugging enabled for {self.__class__.__module__}.{self.__class__.__name__}")

        self.node_id = node_id
        self.topic_handlers: Dict[str, HandlerType] = {}
   
    @abstractmethod
    def publish(self, topic: str, mesage: str):
        raise NotImplemented("PubSubNode->publish not implemented")
    
    @abstractmethod
    def subscribe(self, topic: str):
        raise NotImplemented("PubSubNode->subscribe not implemented")
    
    @abstractmethod
    def register_handler(self, topic: str, handler: HandlerType):
        self._logger.debug(f"Registering handler for topic: {topic}->{handler}")
        self.topic_handlers[topic] = handler

    @abstractmethod
    def start(self):
        raise NotImplemented("PubSubNode->start not implemented")
    
    @abstractmethod
    def stop(self):
        raise NotImplemented("PubSubNode->stop not implemented")
    
    

    