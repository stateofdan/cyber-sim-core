'''
Filename: chat_demo.py

Description:
Demonstrates the setup and interaction of LanguageAgent instances using MQTT-based pub/sub messaging
and an LLM backend in the cyber simulation core. Configures agents with system prompts, sends control
and direct messages, and logs agent responses. Useful for testing agent communication, message handling,
and LLM integration in a simulated environment.

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

import helpers.message_types as message_types

import threading
import time
from pubsubnode.mqttpubsubnode import MQTTPubSubNode
from langagent.agent import LanguageAgent
import json

from llm_system.lm_studio_wrapper import LMStudioManager

# langagent/test_agent.py

persona_prompt1 = '''
You are a highly competent Cybersecurity Manager responsible for protecting an organization’s digital assets, infrastructure, and data. 
Your role includes assessing risks, responding to incidents, advising on policy, and making strategic decisions that balance security, usability, and business impact.

Behave as a professional leader: provide clear, actionable, and risk-aware guidance. Prioritize threat intelligence, regulatory compliance, and secure design principles. 
When evaluating a situation, consider likelihood, impact, threat actors, vulnerabilities, and controls.

Speak in a concise, authoritative tone suitable for executives, technical teams, and incident responders. If uncertain, state assumptions and recommend further investigation or escalation.
'''

dm_msg1 = '''
Hi boss I have discovered a potentially malicious file on a company laptop. 
The file was not flagged by antivirus software, but exhibits unusual behavior: 
it attempts to communicate with an external IP address and modifies registry settings. 
However, it is also part of a legitimate open-source toolkit used by the development team. 
Should I escalate the this for full investigation? 
'''

social_json_data = []

json_data = [
  {
    "id": 1,
    "type": "decide",
    "system_prompts": [0],
    "prompt": ("You have recieved the following message from: {{ sender }} which says:\n"
                "\n\n{{ message }}\n\n"
                "You need to decide whether to respond.\n"
                "Explain your reasoning and provide a confidence level in your decision.").strip(),
    "options": {
      "yes": { "target": 2, "threshold": 0.01 },
      "no": { "target": None, "threshold": 0.01 }
    }
  },
  {
    "id": 2,
    "type": "generate",
    "system_prompts": [0],
    "channels": ["direct_msg/{{ sender }}"],
    "prompt": ("You have recieved the following message from: {{ sender }} which says:\n"
                "\n\n{{ message }}\n\n"
                "You have decided to respond as you have reasoned: {{ context[-1] }}.\n"
                "Create an appropriate message to send back in response.\n"
                "only generate the message do not provide any other content.\n"
                "do not provide any blank or template like fields in this message.\n"
                "The message should be in the same tone as the orginal.").strip(),
    "next": None
  },
  {
    "id": 3,
    "type": "decide",
    "system_prompts": [0],
    "prompt": ("You have recieved the following message from: {{ sender }} which says:\n"
                "\n\n{{ message }}\n\n"
                "You have responded with message:\n\n{{ context[-1] }}\n\n"
                "and your reasoning for the response is:\n\n{{ context[-3] }}\n\n"
                "You need to decide whether you are going to create a social media post about it.\n"
                "Explain your reasoning and provide a confidence level in your decision.").strip(),
    "options": {
      "yes": { "target": 4, "threshold": 0.01 },
      "no": { "target": None, "threshold": 0.3 }
    }
  },
  {
    "id": 4,
    "type": "generate",
    "system_prompts": [0],
    "channels": ["media/social"],
    "prompt": "Generating a social media post .",
    "next": None
  }
]

json_str = json.dumps(json_data)


def test_langague_agent_channels():
    main_logger.info("Starting LanguageAgent channel test...")
    target_agent_id = "target_agent"
    source_agent_id = "source_agent"

    target_pubsubnode = MQTTPubSubNode(target_agent_id)
    source_pubsubnode = MQTTPubSubNode(source_agent_id)
    def handler(topic:str, message:str) ->None:
        msg = (f'\n\n------- God is Listening -------\n\n'
               f'Topic: {topic}\nmessage:\n{json.dumps(json.loads(message), indent=2)}'
                '--------------------------------\n\n'
        )
        print (msg)

    source_pubsubnode.register_handler("direct_msg/god", handler)
 
    channels = ['direct_msg/{{ agent_id }}',
                'social_media/#',       # You dont need to subscribe to your own Social media as you publish to it. '# will subscibe to eveyones.
                ]

    workflows = {'direct_msg/{{ agent_id }}': json_str,
                'social_media/#': json_str,}


    config = {"agent_id": target_agent_id,
              "system_prompts":[persona_prompt1],
              "channels": channels,
              "workflows": workflows,
              }
    

    main_logger.debug(json.dumps(config, indent=2))


    lm_client = LMStudioManager(base_url="http://localhost:1234/v1/", model="qwen-7b")
    agent = LanguageAgent(target_pubsubnode, lm_client, config)
    agent_thread = threading.Thread(target=agent.run)
    agent_thread.start()


    source_pubsubnode.start()

    # Allow agent to start up
    time.sleep(10)
    print ("\n\n\n\n=====================================================================")

    # Send messages to each channel
    ctrl_msg = message_types.ControlMessage(sender="god", command="hello control", args=None)
    source_pubsubnode.publish("control", ctrl_msg.to_json())
    time.sleep(0.2)
    ctrl_msg.command = "hello direct control"
    source_pubsubnode.publish(f"control/{target_agent_id}", ctrl_msg.to_json())
    time.sleep(0.2)
    dm_msg = message_types.DirectMessage(sender="god", text=dm_msg1,)
    source_pubsubnode.publish(f"direct_msg/{target_agent_id}", dm_msg.to_json())
    time.sleep(160)

    # Send quit to control channel
    ctrl_msg.command = "quit"
    source_pubsubnode.publish("control", ctrl_msg.to_json())
    time.sleep(0.2)

    # Wait for agent to stop
    agent_thread.join(timeout=120)
    if agent_thread.is_alive():
        main_logger.info ("Agent thread did not stop as expected")
        agent.stop()
        agent_thread.join(timeout=10)
        if agent_thread.is_alive():
            main_logger.warning("Agent thread is still running after stop command.")
        else:
            main_logger.info("Agent thread stopped successfully.")
    else:
        main_logger.info("Agent thread stopped successfully first time.")
    
    source_pubsubnode.stop()


if __name__ == "__main__":
    test_langague_agent_channels()
    main_logger.info("Test complete.")