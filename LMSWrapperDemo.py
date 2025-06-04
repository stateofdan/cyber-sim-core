'''
Filename: LMSWrapperDemo.py

Description:
Demonstrates usage of the LMStudioManager class for interacting with the LM Studio API in the cyber simulation core.
Shows how to configure system prompts, check connection status, and make decision requests using different response formats.
Useful for testing LLM integration, decision workflows, and prompt engineering in a simulated environment.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
import logging
from helpers.logging_config import setup_logging
setup_logging()
main_logger = logging.getLogger(__name__)
main_logger.debug(f"Logger for {__name__} initialized: {main_logger}")

from llm_system.decision_format import DecisionResponseFormat, create_choice_decision_format_class, TextDecisionFormat
from llm_system.lm_studio_wrapper import LMStudioManager
from textwrap import dedent
import json

if __name__ == "__main__":
    
    #print(f"TextDecisionFormat JSON Schema:\n{TextDecisionFormat.request_format_json_str(indent=2)}\n")
    #edf = create_choice_decision_format_class(
    #    "ExampleDecisionFormat",
    #    options=["option1", "option2", "option3"]
    #)
    #print(f"ExampleDecisionFormat JSON Schema:\n{edf.request_format_json_str(indent=2)}\n")
    
    # Example usage
    lm_manager = LMStudioManager(base_url="http://localhost:1234/v1/", model="qwen-7b")
    sys_prompt = '''
        You are an autonomous decision-making assistant.
        You will be able to make clear and concise responses based on the information provided.
        Your reponses should be based on the context and information given in the questions.'''
    lm_manager.add_system_prompt(dedent(sys_prompt))
    if lm_manager.check_connection():
        print("Connected to LM Studio successfully.")
        decision_question = '''
            A cybersecurity analyst has discovered a potentially malicious file on a company laptop. 
            The file was not flagged by antivirus software, but exhibits unusual behavior: 
            it attempts to communicate with an external IP address and modifies registry settings. 
            However, it is also part of a legitimate open-source toolkit used by the development team. 
            Should the analyst escalate the incident for full investigation? 
            Explain your reasoning and provide a confidence level in your decision.'''
        reply = lm_manager.decide(dedent(decision_question),response_format=TextDecisionFormat.create_request_format(), system_prompt=0)
        print(f"Decision response:\n{json.dumps(reply, indent=2)}\n")

        decision_question = '''
        You are advising a product team on the release timing for a major new software feature. 
        Internal testing is complete, but there are still some non-blocking bugs reported by beta users.
        Marketing is eager to launch next week to align with a campaign, but engineering wants two more weeks for polish and QA. 
        What release strategy should be followed?
        '''
        decisions = ("release_now", "delay_release", "limited_rollout")
        response_format = create_choice_decision_format_class(decisions).create_request_format()
        reply = lm_manager.decide(dedent(decision_question), response_format=response_format)
        print(f"Decision response:\n{json.dumps(reply, indent=2)}\n")

        #response = lm_manager.send("Hello, how are you?")
        #print(f"Response: {response}")
        #response = lm_manager.send("Im very well, thank you! Please tell me a joke.")
        #print(f"Response: {response}")
        #print("Current conversation state:")
        #print(lm_manager.get_state_json())
        #print("\n\n----------------\n\n")
        
    else:
        print("Failed to connect to LM Studio.")