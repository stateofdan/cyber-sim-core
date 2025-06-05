'''
Filename: lm_studio_wrapper.py

Description:
Implements the LMStudioManager class, a concrete subclass of LLMWrapper for interacting with 
the LM Studio API in the cyber simulation core. Handles model selection, connection checks, 
message sending, decision-making, summarization, and conversation management using LM Studio's 
OpenAI-compatible API.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
import openai
import time
import json
from typing import List, Dict, Any, Optional, Literal, Type
import copy  
from textwrap import dedent

from .llm_wrapper import LLMWrapper

class LMStudioManager(LLMWrapper):
    """
    LMStudioManager is a wrapper class for interacting with LM Studio-compatible LLM APIs using the OpenAI client interface.
    Inherits from:
        LLMWrapper
    Args:
        base_url (str): The base URL for the LM Studio API endpoint.
        api_key (str, optional): API key for authentication. Defaults to "LM_STUDIO".
        model (str, optional): Model name to use. Defaults to "default".
        model_strict (bool, optional): Whether to enforce strict model selection. Defaults to False.
        max_tokens (int, optional): Maximum number of tokens for responses. Defaults to 3000.
        summarise_fraction (float, optional): Fraction of conversation to summarize when needed. Defaults to 0.5.
        llm_kwargs (Dict, optional): Additional keyword arguments for the LLM client.
    Attributes:
        _client: The OpenAI client instance for communicating with LM Studio.
        _models (List[str]): List of available model IDs.
        _selected_model (str): The model selected for use.
        _encoding: Encoding used for the selected model.
        _conversation (List[Dict]): Conversation history.
        _decisions (List[Dict]): Decision history.
        _logger: Logger instance for logging events and errors.
    Methods:
        is_connected() -> bool:
            Checks if the LM Studio API is reachable.
        _summarize(messages: List[Dict[str, str]]) -> str:
            Summarizes a given list of conversation messages using the LLM.
        send(user_message: str, system_prompt: int = 0) -> str:
            Sends a user message to the LLM and returns the assistant's reply.
        decide(question: str, response_format: dict, system_prompt: int = 0, retries: int = 3) -> str:
            Sends a decision-making prompt to the LLM and expects a structured JSON response.
        check_connection(retries: int = 3, delay: float = 2.0) -> bool:
            Attempts to connect to the LM Studio API, retrying if necessary.
    Raises:
        RuntimeError: If unable to connect to LM Studio or if message sending/summarization fails.
        ValueError: If a valid structured response is not received after the specified number of retries.
    """
    def __init__(self, 
                 base_url: str, 
                 api_key: str = "LM_STUDIO",
                 model: str = "default",
                 model_strict: bool = False,
                 max_tokens: int = 3000,
                 summarise_fraction: float = 0.5,
                 llm_kwargs: Dict = None):
        
        super().__init__(base_url, api_key, model, model_strict, max_tokens, summarise_fraction, llm_kwargs)

        if model == "default":
            self._logger.info("No model specified, using default model 'deepseek-r1-distill-qwen-7b'")
            model = "deepseek-r1-distill-qwen-7b"

        self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        try:
            models = self._client.models.list().data
            self._logger.info(f'Sucessfully connected to LM studio at {self._base_url}')
            self._models = [m.id for m in models]
            self._logger.debug(f"Available models:\n {''.join(f'\t{item}\n' for item in self._models)}")
        except Exception as e:
            self._logger.critical(f"Failed to connect to, and check models with, LMStudio at {self._base_url}: {e}")
            raise RuntimeError(f"Failed to connect to, and check models with, LMStudio at {self._base_url}: {e}") from e


        self._selected_model = self._resolve_model()
        self._logger.info(f"Selected model: {self._selected_model}")
        self._encoding = self._resolve_encoding()
        self._logger.info(f"Encoding for model '{self._selected_model}' initialized successfully.")

    def is_connected(self)->bool:
        """
        Checks if the connection to the service is available by attempting a single connection check with no delay.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        return self.check_connection(retries=1, delay=0)

    def _summarize(self, messages: List[Dict[str, str]]) -> str:
        """
        Summarizes a conversation represented as a list of message dictionaries.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, where each dictionary contains
                at least the keys 'role' and 'content', representing the conversation to be summarized.

        Returns:
            str: A brief summary of the conversation.

        Raises:
            RuntimeError: If the summarization process fails due to an exception from the client.
        """
        try:
            self._logger.debug(f'Summarising the following message content:\n{json.dumps(messages, indent=2)}')
            summary = self._client.chat.completions.create(
                model=self._selected_model,
                messages=[{"role": "system", "content": "Summarize the following conversation briefly:"}] +
                         messages,
                temperature=0.3,
                max_tokens=300
            )
            self._logger.info(f'Successfully summarised {len(messages)} messages.')
            return summary.choices[0].message.content.strip()
        except Exception as e:
            self._logger.critical(f"Failed to summarize: {e}")
            raise RuntimeError(f"Failed to summarize: {e}") from e
   
    def send(self, user_message: str, system_prompt_idx: int | list[int] = 0) -> str:
        """
        Sends a user message to the language model, optionally including a system prompt, and returns the assistant's reply.
        Args:
            user_message (str): The message from the user to send to the language model.
            system_prompt (int or list of int, optional): The index or indices for the system prompt(s) to use. Defaults to 0.
        Returns:
            str: The reply generated by the language model.
        Raises:
            RuntimeError: If sending the message or receiving a reply fails.
        Side Effects:
            - Logs debug and info messages about the prompt and reply.
            - Updates the conversation history with the new user and assistant messages.
            - May summarize the conversation if necessary.
        """
        '''
        self._logger.debug(f"Decision system_prompt index: {system_prompt}")
        sys_p_text = ""
        if system_prompt is not None:
            if isinstance(system_prompt, int):
                sys_p_text = f'{self.get_system_prompt(system_prompt)}\n\n'
            elif isinstance(system_prompt, list):
                sys_p_text = "\n\n".join(self.get_system_prompt(idx) for idx in system_prompt) + "\n\n"

        sys_prompt = (
            f"{sys_p_text if system_prompt is not None else ''}"
        )
        self._logger.debug(f'\n-------\nsystem_prompt:\n {sys_prompt}\n-------\n\n')
        '''
        sys_prompt = self._system_prompt_constructor("", system_prompt_idx)
        new_message = {"role": "user", "content": dedent(user_message).strip()}
        self._logger.debug(f"New message prompt:\n{user_message}\n")
        conversation = [{"role": "system", "content": sys_prompt}] + copy.deepcopy(self._conversation) + [new_message]
        self._maybe_summarize()

        try:
            self._logger.info(f'Sending generation request to model: {self._selected_model}, with {len(conversation)} items.')
            response = self._client.chat.completions.create(
                model=self._selected_model,
                messages=conversation,
                max_tokens=1024,
                **self._llm_kwargs
            )
            reply = response.choices[0].message.content.strip()
            self._logger.info(f"Received reply from LM Studio: {reply[:40]}")
            self._logger.debug(f"Received reply from LM Studio:\n{json.dumps({k: v for k, v in response.to_dict().items() if k!= 'choices'}, indent=2)}")
            self._conversation.append({"role": "user", "content": new_message})
            self._conversation.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            self._logger.critical(f"Failed to send message: {e}")
            raise RuntimeError(f"Failed to send message: {e}") from e
        
    def decide(self, question: str, response_format: dict, system_prompt_idx: int | list[int] = 0, retries: int = 3) -> str:
        """
        Generates a decision response from the language model based on a given question and expected JSON schema.
        Args:
            question (str): The decision-making question to be answered by the model.
            response_format (dict): A dictionary specifying the expected response format, including a 'json_schema' key.
            system_prompt (int, optional): An identifier for the system prompt to use. Defaults to 0.
            retries (int, optional): Number of attempts to get a valid response from the model. Defaults to 3.
        Returns:
            str: The model's response as a string, formatted according to the provided JSON schema.
        Raises:
            ValueError: If a valid structured response is not obtained after the specified number of retries.
        """

        '''
        self._logger.debug(f"Decision system_prompt index: {system_prompt}")
        sys_p_text = ""
        if system_prompt is not None:
            sys_p_text = f'{self.get_system_prompt(system_prompt)}\n\n'

        sys_prompt = (
            f"{sys_p_text if system_prompt is not None else ''}"
            f"You are to answer the following decision-making task. "
            f"Respond ONLY in valid JSON schema matching this structure:\n"
            f"{json.dumps(response_format['json_schema'], indent=1)}\n\n"
        )

        self._logger.debug(f'\n-------\nsystem_prompt:\n {sys_prompt}\n-------\n\n')
        '''
        decision_sys_prompt = "\n".join([
            f"You are to answer the following decision-making task. "
            f"Respond ONLY in valid JSON schema matching this structure:\n"
            f"{json.dumps(response_format['json_schema'], indent=1)}\n\n"
        ]).strip()
        sys_prompt = self._system_prompt_constructor(decision_sys_prompt, system_prompt_idx)
        new_question = {"role": "user", "content": dedent(question).strip()}
        conversation = [{"role": "system", "content": sys_prompt}] + copy.deepcopy(self._conversation) + [new_question]
        self._maybe_summarize("decision")

        self._logger.debug(f"Decision prompt:\n{question}\nExpected response format:\n{json.dumps(response_format, indent=2)}")
                
        for attempt in range(1, retries + 1):
            self._logger.debug(f"Decision attempt {attempt}: sending decision prompt.")
            try:
                response = self._client.chat.completions.create(
                    model=self._selected_model,
                    messages=conversation,
                    response_format=response_format,
                    max_tokens=1024,
                    **self._llm_kwargs
                )
                decision_reasoning = response.choices[0].message
                if decision_reasoning.refusal:
                    self._logger.warning(f"attempt {attempt}: decision reasoning refusal: {decision_reasoning.refusal}")
                    continue

                reply = response.choices[0].message.content.strip()
                self._logger.info(f"Raw decision reply:\n{reply}")
                self._logger.debug(f"Received response from LM Studio:\n{json.dumps(response.to_dict(), indent=2)}")
                self._decisions.append(new_question)
                self._decisions.append({"role": "assistant", "content": reply})
                return reply
            except Exception as e:
                self._logger.warning(f"Attempt {attempt}: Failed : {e}")
        self._logger.critical(f"Failed to get valid structured response after {retries} attempts.")
        raise ValueError(f"Failed to get valid structured response after {retries} attempts.")

    def check_connection(self, retries: int = 3, delay: float = 2.0) -> bool:
        """
        Checks the connection to the client by attempting to list available models.

        Args:
            retries (int, optional): Number of times to retry the connection in case of failure. Defaults to 3.
            delay (float, optional): Delay in seconds between retries. Defaults to 2.0.

        Returns:
            bool: True if the connection is successful, False otherwise.
        """
        for attempt in range(retries):
            try:
                _ = self._client.models.list()
                return True
            except Exception:
                time.sleep(delay)
        return False
    


