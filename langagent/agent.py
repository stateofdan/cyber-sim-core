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

from typing import Callable, Dict, List, Tuple
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
        self._logger.debug(f"[__INIT__] Debugging enabled for {self.__class__.__module__}.{self.__class__.__name__}")

        self.__pubsubnode = pubsub_node
        self.__lm_client = lm_client
        self.__state_store_on_quit = state_store_on_quit

        if pubsub_node is None:
            self._logger.critical(f'[__INIT__] pubsub_node object is None')
            raise ValueError(f'[__INIT__] pubsub_node object is None')
        if not isinstance(pubsub_node, PubSubNode):
            self._logger.critical(f'[__INIT__] pubsub_node is not a PubSubNode. Got {type(pubsub_node)}')
            raise ValueError(f'[__INIT__] pubsub_node is not a PubSubNode. Got {type(pubsub_node)}')
        
        if lm_client is None:
            self._logger.critical(f'[__INIT__] lm_client object is None')
            raise ValueError(f'[__INIT__] lm_client object is None')
        if not isinstance(lm_client, LLMWrapper):
            self._logger.critical(f'[__INIT__] lm_client is not an LLMWrapper. Got {type(lm_client)}')
            raise ValueError(f'[__INIT__] lm_client is not a PubSubNode. Got {type(lm_client)}')  
        # End of storing passed in Variables

        self.__agent_id = config.get("agent_id", "default_agent")
        if self.__agent_id == "default_agent":
            self._logger.warning("[__INIT__] No agent_id in config using 'default_agent.")
        self._logger.debug(f'[__INIT__] Agent id set to {self.__agent_id}')

        system_prompts = config.get("system_prompts", [])
        if len(system_prompts) == 0:
            self._logger.warning("[__INIT__] No System Prompts specified for the agent.")
        for prompt in system_prompts:
            self.__lm_client.add_system_prompt(prompt) 
        self._logger.debug(f'[__INIT__] loaded prompt: {"\n".join(self.__lm_client.iter_system_prompts())}')
    
        workflows = config.get('workflows', None)

        self._workflows = {}

        if workflows is None:
            self._logger.critical(f'[__INIT__] No workflows specified in config')
            raise ValueError(f'[__INIT__] No workflows specified in config')
        
        for topic, workflow_cfg_str in workflows.items():
            channel = Template(topic).render(agent_id=self.__agent_id)
            self._workflows[channel] = WorkflowEngine.from_json(workflow_cfg_str, 
                                                                self._workflow_decide, 
                                                                self._workflow_generate,
                                                                self._logger)
        self._logger.info(f'[__INIT__] Loaded {len(self._workflows)} workflows out of {len(workflows)}')

        # End of config loading and checking.

        self.__msg_queue = queue.Queue()
        self.__handlers: Dict[str, List[Callable[[str], None]]] = {}

        # Connect to the main control topic channel
        self.register_handler("control", self.on_all_control_callback)

        # connect to the agent-specific control topic channel
        self.register_handler(f"control/{self.__agent_id}", self.on_direct_control_callback)

        for topic in self._workflows.keys():
            #handler = self.on_message_do_workflow
            #if topic == f"direct_msg/{self.__agent_id}":
            #    handler = self.on_direct_message
            self.register_handler(topic, self.on_message_do_workflow)
            self._logger.debug(f'[__INIT__] Registered handler for topic {topic}: {self.on_message_do_workflow.__name__}')
        self._logger.info(f'[__INIT__] Registered {len(self._workflows)} topics')
 
        self._logger.info(f"[__INIT__] LanguageAgent initialized with ID: {self.__agent_id}\n{'+'*30}\n")
           
    def register_handler(self, topic: str, handler: Callable[[str], None]):
        self.__handlers.setdefault(topic, []).append(handler)
        self.__pubsubnode.register_handler(topic, self.handle_message)
        self._logger.info (f'[REG_HNDL] Registered handler for {topic} for agent {self.__agent_id}')
        self._logger.debug (f"[REG_HNDL] Registered {topic} connect to {handler.__name__}")


    def handle_message(self, topic: str, message: str):
        self.__msg_queue.put((topic, message))
        self._logger.info(f"[HANDLE] Queued message for {self.__agent_id}")
        self._logger.debug(f'[HANDLE] Received message from {topic}:\n'
                            f'{'-'*20}\n'
                            f'{message if len(message) <=60 else f'{message[:25]}\n...\n{message[-25:]}\n'}'
                            f'{'-'*20}\n')
        
    def _handle_ctrl_msg(self, topic:str, message:str):
        source = "ALL_CRTL" if topic == "control/" else "DIRECT_CTRL"

        ctrl_msg = None
        try:
            ctrl_msg = mt.BaseMessage.from_json(message)
            if not isinstance(ctrl_msg, mt.ControlMessage):
                raise TypeError("Expected ControlMessage, got " + type(ctrl_msg).__name__)
            self._logger.info(f'[{source}] Message deserialised correctly')
            self._logger.debug(f'[{source}] Received message on {topic}:'
                               f'{'-'*20}\n'
                               f'{message if len(message) <=60 else f'{message[:25]}\n...\n{message[-25:]}\n'}'
                               f'{'-'*20}\n')
        except Exception as e:
            self._logger.critical(f'[{source}] Deserialisation Failed:\n'
                                   f'Exception:\n{e}\n'
                                   f'{'-'*20}\n{message}\n{'-'*20}')
            return

        if ctrl_msg.command == "quit":
            self._logger.info(f"[{source}] Received quit command.")
            if self.__state_store_on_quit:
                self.__lm_client.save_LLM_state_to_file(f"{self.__agent_id}.json")
            self.__running = False
        else:
            self._logger.warning (f"[{source}] Received unknowncontrol message on {topic}: {ctrl_msg.command}")

    def on_all_control_callback(self, topic:str, message:str):
        self._logger.info(f'[ALL_CRTL] Begin Message Processing')
        self._handle_ctrl_msg(topic, message)
        self._logger.info(f'[ALL_CTRL] Finished Message Processing\n{'='*20}\n')

    def on_direct_control_callback(self, topic:str, message:str):
        self._logger.info(f'[DIRECT_CRTL] Begin Message Processing')
        self._handle_ctrl_msg(topic, message)
        self._logger.info(f'[DIRECT_CTRL] Finished Message Processing\n{'='*20}\n')


    def _workflow_decide(self, prompt:str, sys_prompts:List[int], options:List[str], context_stack:List[str]):
        self._logger.info(f'[DECIDE] workflow decision started')
        prompt_template = Template(prompt)
        substitute_dict = {"sender": context_stack[0],
                        "message": context_stack[1],
                        "context": context_stack}
        prompt_render = prompt_template.render(**substitute_dict)

        self._logger.debug(f"[DECIDE] Prompt Render\n{'-'*20}\n{prompt_render}\n{'-'*20}")
        decisions = tuple(set(options))
        if not (len(decisions) == len(options)):
            self._logger.warning(f'[DECIDE] Not all Options are unique\n{'-'*20}\n{'\n'.join(options)}{'+'*20}\n{'\n'.join(decisions)}{'-'*20}')
        response_format = create_choice_decision_format_class(decisions).create_request_format()
        self._logger.info(f'[DECIDE] Response Format scehema created for {len(decisions)} created.')
        self._logger.debug(f'[DECIDE] Response Format Schema:\n{'-'*20}\n{json.dumps(response_format, indent=2)}\n{'-'*20}')
        reply = self.__lm_client.decide(prompt_render, response_format=response_format, system_prompt_idx=sys_prompts)
        self._logger.info(f'[DECIDE] Success reply length: {len(reply)}')
        self._logger.debug(f'[DECIDE] Reply response:\n{'-'*20}\n{reply}\n{'-'*20}')
        reply_dict = json.loads(reply)
        reasoning = reply_dict['explanation']
        choice = reply_dict['decision']
        confidence = reply_dict['confidence']
        self._logger.info(f'[DECIDE] reconstructed respons success - choice:{choice}, confidence:{confidence}, reasoning length: {len(reasoning)}')
        self._logger.debug(f"[DECIDE] Decision explanation - {reasoning[:10]} ... {reasoning[-10:]}")

        context_stack.append(reasoning)
        self._logger.info(f'[DECIDE] Successfully Completed\n{'='*20}')
    
        return choice, confidence

    def _workflow_generate(self, prompt:str, sys_prompts:List[int], channels: List[str], context_stack: List[str]):
        self._logger.info(f"[GENERATE] Workflow generations started")
        prompt_template = Template(prompt)
        substitute_dict = {"sender": context_stack[0],
                            "message": context_stack[1],
                            "context": context_stack}
        prompt_render = prompt_template.render(**substitute_dict)

        channels_rendered = []
        for channel in channels:
            channels_rendered.append(Template(channel).render(sender=context_stack[0]))

        self._logger.debug(f"[GENERATE] Prompt Render\n{'-'*20}\n{prompt_render}\n{'-'*20}")
        dm_reply = self.__lm_client.send(prompt_render, sys_prompts)
        self._logger.info(f"[GENERATE] Success reply length: {len(dm_reply)}")
        self._logger.debug(f"[GENERATE] Reply response:\n{'-'*20}\n{dm_reply}\n{'-'*20}")
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
        
        self._logger.debug(f'[GENERATE] Reasoning: {f"{dict_dm_reply['think'][:10]} ... {dict_dm_reply['think'][-10:]}" if dict_dm_reply['think'] else "No reason given"}')
        self._logger.debug(f'[GENERATE] Response: {dict_dm_reply['reply'][:10]} ... {dict_dm_reply['reply'][-10:]}')
        context_stack.append(dict_dm_reply['think'])
        context_stack.append(dict_dm_reply['reply'])
        
        response_dm = mt.DirectMessage(sender=self.__agent_id, text=dict_dm_reply['reply'])
        self._logger.info(f'[GENERATE] Sending Message length: {len(dict_dm_reply['reply'])}, to {len(channels)} channels"')
        self._logger.debug(f'[GENERATE] Message being sent:\n{'-'*20}\n{dict_dm_reply['reply']}\n{'-'*20}')
        for i, channel in enumerate(channels_rendered):
            self.__pubsubnode.publish(f"direct_msg/{context_stack[0]}", response_dm.to_json())
            self._logger.debug(f'[GENERATE] To Channel: {channel}{f"\n{'-'*20}" if i == len(channels)-1 else ""}')
        self._logger.info(f'[GENERATE] Successfully Completed\n{'='*20}\n')
        

    def _mqtt_topic_match(self, sub_topic: str, tgt_topic: str) -> bool:
        sub_tokens = sub_topic.split('/')
        tgt_tokens = tgt_topic.split('/')

        i = 0
        while i < len(sub_tokens):
            if i >= len(tgt_tokens):
                # if the sub_tpic has more tockns but target topic has ended,
                # only match if the last token is '#' (matches zero or more levels)
                return sub_tokens[i] == '#' and i == len(sub_tokens) - 1
            
            if sub_tokens[i] == '#':
                # '#' must be last token and matches all remaining topic tokens
                return i == len(sub_tokens) - 1
            
            if sub_tokens[i] == '+':
                # '+' matches exactly one topic level
                i+= 1
                continue
            
            if sub_tokens[i] != tgt_tokens[i]:
                return False
                
            i += 1
        return i == len(tgt_tokens)
    

    def _get_best_workflow(self, topic: str)-> Tuple[str, WorkflowEngine]:
        matched = []
        for sub_topic, workflow in self._workflows.items():
            if self._mqtt_topic_match(sub_topic, topic):
                matched.append((sub_topic, workflow))
        if not matched:
            return None, None
        
        best_sub_topic, best_workflow = max(matched, key=lambda x: len(x[0]))
        return best_sub_topic, best_workflow


    def on_message_do_workflow(self, topic:str, message:str):
        self._logger.info(f'[DIRECT] Begin Processing')

        sub_topic, workflow = self._get_best_workflow(topic) # not sure this would match topic/# type entries.
        if workflow is None:
            self._logger.warning(f'[DIRECT] No workflow for topic: {topic}\n\n{'+'*20}\nFAILED PROCESSING\n{'+'*20}\n')
            return
        self._logger.info(f'[DIRECT] Found workflow for {topic}')
        self._logger.debug(f'[DIRECT] Topic {topic} matched with {sub_topic}')
    

        dm_msg = None
        try:
            dm_msg = mt.BaseMessage.from_json(message)
            if not isinstance(dm_msg, mt.DirectMessage):
                raise TypeError("Expected DirectMessage, got " + type(dm_msg).__name__)
            self._logger.info(f'[DIRECT] message decoded correctly')
            self._logger.debug(f'[DIRECT] Received message on {topic}:\n'
                               f'{'-'*20}\n'
                               f'{message if len(message) <=60 else f'{message[:25]}\n...\n{message[-25:]}\n'}'
                               f'{'-'*20}\n')
        except Exception as e:
            self._logger.critical(f'[DIRECT] Deserialisation Failed:\n'
                                   f'Exception:\n{e}\n'
                                   f'{'-'*20}\n{message}\n{'-'*20}')
            return

        dm = dm_msg.text
        context_stack = [dm_msg.sender,dm]
        #self._workflow_engine.run(context_stack)
        workflow.run(context_stack)
        self._logger.info(f'[DIRECT] Workflow completed successfully')
        self._logger.debug(f'[DIRECT] Workflow completed context_stack:\n'
                           f'{'-'*20}\n{json.dumps(context_stack, indent=2)}\n{'-'*20}')
        self._logger.info(f'[DIRECT] Finished Processing\n{'='*20}\n')

    def run(self):
        self._logger.info(f"[RUN] Starting LanguageAgent thread for {self.__agent_id}")
        self.__pubsubnode.start()
        self.__running = True

        while self.__running:
            self._logger.debug(f"[RUN] {self.__agent_id}] checking message queue")
            # Main loop for the agent
            try:
                topic, message = self.__msg_queue.get(timeout=10)  # Wait for a message
                self._logger.info(f'[RUN] Agent {self.__agent_id} received Message')
                self._logger.debug(f'[RUN] Processing message from {topic}:')

                if topic == "control":
                    self.on_all_control_callback(topic, message)
                elif topic.startswith("control/"):
                    self.on_direct_control_callback(topic, message)
                #elif topic.startswith("direct_msg/"):
                else:
                    self.on_message_do_workflow(topic, message)
                self._logger.info(f'[RUN] Agent {self.__agent_id} Message processed correctly')
            except queue.Empty:
                self._logger.debug(f"[RUN] Agent {self.__agent_id} No messages in queue, continuing...")
                continue
            except Exception as e:
                self._logger.critical(f'[RUN] Agent {self.__agent_id} Exception in agent thread:\n'
                                      f'{'-'*20}\n{e}\n{'-'*20}\n'
                                      f'{'x'*10} STOPPING {'x'*10}\n')
                self.__running = False
        
        self.__pubsubnode.stop()
        self._logger.info (f"[RUN] Agent {self.__agent_id}] Stopped.\n{'='*20}\n")

    def stop(self):  
        self.__running = False
        self._logger.info(f"[STOP] Agent {self.__agent_id}] force stop.")