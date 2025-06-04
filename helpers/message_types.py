'''
Filename: message_types.py

Description:
Defines a flexible message type system for the cyber simulation core. 
Provides a BaseMessage class with serialization/deserialization logic and 
a registry for message type subclasses. Implements specific message types 
(ControlMessage, DirectMessage, SocialMessage) for use in simulation 
communication, supporting extensible message structures and type-safe handling.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
from dataclasses import dataclass, field, asdict
from typing import Literal, Dict, Any, Type, ClassVar
from uuid import uuid4
from datetime import datetime, timezone
import json

@dataclass
class BaseMessage:
    msg_type: str  # 'control', 'direct', 'social'
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender: str = "unknown" # This is the source so media/<sender>/press or direct/<sender>/

    _registry: ClassVar[Dict[str, Type['BaseMessage']]] = {}

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def register_subclass(cls, message_type: str):
        def decorator(subclass):
            cls._registry[message_type] = subclass
            return subclass
        return decorator

    @classmethod
    def from_json(cls, data: str) -> 'BaseMessage':
        raw = json.loads(data)
        msg_type = raw.get("msg_type")
        if msg_type not in cls._registry:
            raise ValueError(f"Unknown msg_type: {msg_type}")
        subclass = cls._registry[msg_type]
        return subclass(**raw)

@BaseMessage.register_subclass("control")    
@dataclass
class ControlMessage(BaseMessage):
    msg_type: Literal['control'] = 'control'
    command: str = ""  # e.g., 'start', 'shutdown', 'update_config'
    args: Dict[str, Any] = field(default_factory=dict)  # extensible argument map

@BaseMessage.register_subclass("direct")
@dataclass
class DirectMessage(BaseMessage):
    msg_type: Literal['direct'] = 'direct'
    text: str = ""
    conversation_id: str = ""  # Optional thread ID for context

@BaseMessage.register_subclass("social")
@dataclass
class SocialMessage(BaseMessage):
    msg_type: Literal['social'] = 'social'
    post_type: Literal['post', 'reply', 'like', 'update'] = 'post'
    content: str = ""
    thread_id: str = ""
    in_reply_to: str = ""  # message ID being replied to or liked