'''
Filename: agent.py

Description:
Implements the LanguageAgent class, a threaded agent for the cyber simulation core.
The agent subscribes to pub/sub topics, processes control and direct messages,
and interacts with an LLM client to make decisions and generate responses.
Supports extensible message handling, agent-specific control, and state persistence.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
import threading
from pubsubnode.pubsubnode import PubSubNode
import queue

from typing import Callable, Dict, List
from textwrap import dedent
import json

from llm_system.decision_format import DecisionResponseFormat, create_choice_decision_format_class, TextDecisionFormat
from llm_system.llm_wrapper import LLMWrapper

import logging
import re

import helpers.message_types as mt
from jinja2 import Template

from .workflow_engine import WorkflowEngine

class LanguageAgent(threading.Thread):
    def __init__(self, pubsub_node:PubSubNode, lm_client: LLMWrapper, config:dict, state_store_on_quit: bool=True):
        super().__init__()
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._logger.debug(f"Debugging enabled for {self.__class__.__module__}.{self.__class__.__name__}")

        self.__agent_id = config.get("agent_id", "default_agent")
        system_prompts = config.get("system_prompts", [])
        self.__state_store_on_quit = state_store_on_quit

        self._workflow_config_str = config.get("workflow", None)
        if self._workflow_config_str is None:
            self._logger.critical(f'Workflow not found in LanguageAgent config')
            raise ValueError(f'Workflow not found in LanguageAgent config')
        self._workflow_engine = WorkflowEngine.from_json(self._workflow_config_str, 
                                                         self._workflow_decide, 
                                                         self._workflow_generate,
                                                         self._logger)

        self.__pubsubnode = pubsub_node
        self.__lm_client = lm_client

        for prompt in system_prompts:
            self.__lm_client.add_system_prompt(prompt)      


        self.__handlers: Dict[str, List[Callable[[str], None]]] = {}
        self.__msg_queue = queue.Queue()
        self.__running = True

        # Connect to the main control topic channel
        #self.__pubsubnode.register_handler("control", self.handle_message)
        self.register_handler("control", self.on_all_control_callback)

        # connect to the agent-specific control topic channel
        #self.__pubsubnode.register_handler(f"control/{self.__agent_id}", self.handle_message)
        self.register_handler(f"control/{self.__agent_id}", self.on_direct_control_callback)

        # Connect to the agent-specific input topic channel - used to send direct messages between agents for example alerting
        #self.__pubsubnode.register_handler(f"agents/{self.__agent_id}/in_dm", self.handle_message)
        self.register_handler(f"direct_msg/{self.__agent_id}", self.on_direct_message)
        self._logger.info(f"[{self.__agent_id}] LanguageAgent initialized with ID: {self.__agent_id}")

    
    def register_handler(self, topic: str, handler: Callable[[str], None]):
        self._logger.info (f"[{self.__agent_id}] Registering handler for topic: {topic}->{handler}")
        self.__handlers.setdefault(topic, []).append(handler)
        self.__pubsubnode.register_handler(topic, self.handle_message)

    def handle_message(self, topic: str, message: str):
        self._logger.debug(f"[{self.__agent_id}] Queued from {topic}: {message}")
        self.__msg_queue.put((topic, message))


    def _handle_ctrl_msg(self, topic:str, message:str):
        ctrl_msg = None
        try:
            ctrl_msg = mt.BaseMessage.from_json(message)
            self._logger.info(f'Received msg:\n{message[:40]}...\nDecoded as {type(ctrl_msg).__name__}')
            if not isinstance(ctrl_msg, mt.ControlMessage):
                raise TypeError("Expected ControlMessage, got " + type(ctrl_msg).__name__)
        except Exception as e:
            self._logger.error(f'recieved message:\n{message}\n Could not be deserialised as control message.\n{e}')
            return

        if ctrl_msg.command == "quit":
            self._logger.info(f"[{self.__agent_id}] Received quit command.")
            if self.__state_store_on_quit:
                self.__lm_client.save_LLM_state_to_file(f"{self.__agent_id}.json")
            self.__running = False
        else:
            self._logger.warning (f"[{self.__agent_id}] Received all control message on {topic}: {ctrl_msg.command}")

    def on_all_control_callback(self, topic:str, message:str):
        self._handle_ctrl_msg(topic, message)
    def on_direct_control_callback(self, topic:str, message:str):
        self._handle_ctrl_msg(topic, message)


    def _workflow_decide(self, prompt:str, options:List[str], context_stack:List[str]):
        prompt_template = Template(prompt)
        substitute_dict = {"sender": context_stack[0],
                        "message": context_stack[1],
                        "context": context_stack}
        prompt_render = prompt_template.render(**substitute_dict)

        self._logger.debug(f"[DECIDE] {prompt_render}\nOptions: {options}")
        decisions = tuple(set(options))
        response_format = create_choice_decision_format_class(decisions).create_request_format()
        reply = self.__lm_client.decide(prompt_render, response_format=response_format)
        reply_dict = json.loads(reply)
        choice = reply_dict['decision']
        confidence = reply_dict['confidence']
        self._logger.info(f'[DECIDE] success:{choice}, confidence:{confidence}')
        self._logger.debug(f"\n-----[DECIDE]-----\nDecision response:\n{reply}\n----------\n")

        context_stack.append(reply_dict['explanation'])
        self._logger.info(f'[DECIDE] Successfully Completed')
    
        return choice, confidence

    def _workflow_generate(self, prompt:str, channels: List[str], context_stack: List[str]):
        prompt_template = Template(prompt)
        substitute_dict = {"sender": context_stack[0],
                            "message": context_stack[1],
                            "context": context_stack}
        prompt_render = prompt_template.render(**substitute_dict)

        channels_rendered = []
        for channel in channels:
            channels_rendered.append(Template(channel).render(sender=context_stack[0]))

        self._logger.debug(f"[GENERATE] {prompt_render}")
        dm_reply = self.__lm_client.send(prompt_render, 0)
        self._logger.info(f"[GENERATE] Success reply length {len(dm_reply)}")
        self._logger.debug(f"\n-----[GENERATE]-----\reply response:\n{dm_reply}\n----------\n")
        def extract_think_and_reply(text: str) -> dict:
            match = re.match(r'<think>(.*?)</think>(.*)', text, flags=re.DOTALL)
            if match:
                return {
                    "think": match.group(1).strip(),
                    "reply": match.group(2).strip()
                }
            else:
                return {
                    "think": None,
                    "reply": text.strip()
                }
        dict_dm_reply = extract_think_and_reply(dm_reply)
        
        self._logger.debug(f'Reasoning:\n{dict_dm_reply['think']}')
        self._logger.debug(f'Response:\n{dict_dm_reply['reply']}')
        context_stack.append(dict_dm_reply['think'])
        context_stack.append(dict_dm_reply['reply'])
        
        response_dm = mt.DirectMessage(sender=self.__agent_id, text=dict_dm_reply['reply'])
        self._logger.debug(f'\n\n----- [GENERATE] Sending -----\nsending message {dict_dm_reply['reply']}\n')
        for channel in channels_rendered:
            self._logger.debug(f'To Channel: {channel}\n')
            self.__pubsubnode.publish(f"direct_msg/{context_stack[0]}", response_dm.to_json())
        self._logger.info(f'[GENERATE] Successfully Completed')
        

    def on_direct_message(self, topic:str, message:str): 
        dm_msg = None
        try:
            dm_msg = mt.BaseMessage.from_json(message)
            self._logger.info(f'Received msg:\n{message[:20]}...\nDecoded as ')
            if not isinstance(dm_msg, mt.DirectMessage):
                raise TypeError("Expected DirectMessage, got " + type(dm_msg).__name__)
        except Exception as e:
            self._logger.error(f'recieved message:\n{message}\n Could not be deserialised as direct message.\n{e}')
            raise # this just raises the exceptione again by default

        dm = dm_msg.text
        context_stack = [dm_msg.sender,dm]
        self._workflow_engine.run(context_stack)
        self._logger.debug(f'Workflow context_stack:\n{json.dumps(context_stack, indent=2)}')

    def run(self):
        self._logger.info(f"[{self.__agent_id}] Starting LanguageAgent thread.")
        self.__pubsubnode.start()

        while self.__running:
            self._logger.debug(f"[{self.__agent_id}] Agent thread is running, checking message queue...")
            # Main loop for the agent
            try:
                topic, message = self.__msg_queue.get(timeout=10)  # Wait for a message
                self._logger.info(f"[{self.__agent_id}] Processing message from {topic}: {message}")

                if topic == "control":
                    self.on_all_control_callback(topic, message)
                elif topic.startswith("control/"):
                    self.on_direct_control_callback(topic, message)
                elif topic.startswith("direct_msg/"):
                    self.on_direct_message(topic, message)

            except queue.Empty:
                self._logger.debug(f"[{self.__agent_id}] No messages in queue, continuing...")
                continue
            except Exception as e:
                self._logger.critical(f"[{self.__agent_id}] Exception in agent thread: {e}")
                self.__running = False
        
        self.__pubsubnode.stop()
        self._logger.info (f"[{self.__agent_id}] Agent Stopped.")

    def stop(self):
        self._logger.info(f"[{self.__agent_id}] Stopping LanguageAgent thread.")  
        self.__running = False