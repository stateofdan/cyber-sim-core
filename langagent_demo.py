'''
Filename: langagent_demo.py

Description:
Demonstrates a minimal test setup for LanguageAgent instances using MQTT-based pub/sub messaging
and a dummy LLM client in the cyber simulation core. Configures agents, sends control and direct
messages, and verifies agent lifecycle and message handling. Useful for basic testing of agent
infrastructure and communication channels without requiring a real LLM backend.

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
from langagent.agent import LanguageAgent

# langagent/test_agent.py

class DummyLMClient:
    def add_system_prompt(input:str):
        pass
    
    def save_LLM_state_to_file(input:str):
        pass

def test_langague_agent_channels():
    main_logger.info("Starting LanguageAgent channel test...")
    target_agent_id = "target_agent"
    source_agent_id = "source_agent"
    target_pubsubnode = MQTTPubSubNode(target_agent_id)
    source_pubsubnode = MQTTPubSubNode(source_agent_id)
    lm_client = DummyLMClient()
    config = {"agent_id": target_agent_id}

    agent = LanguageAgent(target_pubsubnode, lm_client, config)
    agent_thread = threading.Thread(target=agent.run)
    agent_thread.start()


    source_pubsubnode.start()

    # Allow agent to start up
    time.sleep(10)

    # Send messages to each channel
    source_pubsubnode.publish("control", "hello control")
    time.sleep(0.2)
    source_pubsubnode.publish(f"control/{target_agent_id}", "hello direct control")
    time.sleep(0.2)
    source_pubsubnode.publish(f"agents/{target_agent_id}/in_dm", "hello direct message")
    time.sleep(0.2)

    # Send quit to control channel
    source_pubsubnode.publish("control", "quit")
    time.sleep(0.2)
    source_pubsubnode.stop()

    # Wait for agent to stop
    agent_thread.join(timeout=10)
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


if __name__ == "__main__":
    test_langague_agent_channels()
    main_logger.info("Test complete.")