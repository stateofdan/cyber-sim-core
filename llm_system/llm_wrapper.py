'''
Filename: llm_wrapper.py

Description:
Defines the abstract LLMWrapper class, providing a standard interface and utility methods
for interacting with large language models (LLMs) in the cyber simulation core. 
Handles model selection, system prompts, conversation management, token counting, 
summarization, and state persistence. Intended to be subclassed for specific LLM implementations.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
from abc import ABC, abstractmethod
import tiktoken
from typing import List, Dict, Any, Optional, Literal, Type
from difflib import get_close_matches
import logging
import json

class LLMWrapper(ABC):
    """
    Abstract base class for Large Language Model (LLM) wrappers.
    This class provides a structured interface and utility methods for interacting with LLM APIs, 
    managing system prompts, conversation history, model selection, and token counting. 
    It is designed to be subclassed for specific LLM providers.
    Attributes:
        _base_url (str): The base URL for the LLM API.
        _api_key (str): The API key for authentication.
        _model (str): The preferred model name or identifier.
        _selected_model (Optional[str]): The resolved model name after selection.
        _model_strict (bool): If True, require exact model name match.
        _max_tokens (int): Maximum allowed tokens for conversation or decision history.
        _summarise_fraction (float): Fraction of history to summarize when exceeding max tokens.
        _llm_kwargs (dict): Additional keyword arguments for LLM API calls.
        _system_prompts (List[str]): List of system prompts to prepend to conversations.
        _models (List[str]): List of available model names.
        _conversation (List[Dict[str, str]]): Conversation history.
        _decisions (List[Dict[str, str]]): Decision history.
        _encoding (Optional[tiktoken.Encoding]): Token encoding for the selected model.
    Methods:
        add_system_prompt(prompt): Add a system prompt.
        get_system_prompt(index): Retrieve a system prompt by index.
        remove_system_prompt(index): Remove a system prompt by index.
        clear_system_prompts(): Remove all system prompts.
        system_prompts_count(): Return the number of system prompts.
        iter_system_prompts(): Iterate over system prompts.
        system_prompts_as_list(): Return a copy of the system prompts list.
        base_url: Property for the LLM API base URL.
        api_key: Property for the LLM API key.
        model: Property for the preferred model.
        selected_model: Property for the resolved model.
        models: Property for the list of available models.
        model_strict: Property for strict model matching.
        max_tokens: Property for the maximum allowed tokens.
        summarise_fraction: Property for the summarization fraction.
        llm_kwargs: Property for LLM API keyword arguments.
        is_connected: Property indicating if the LLM API is reachable (must be implemented by subclass).
        send(user_message, system_prompt): Send a message to the LLM and get a response (abstract).
        decide(question, response_format, system_prompt, retries): Make a decision using the LLM (abstract).
        _resolve_model(substring_matching): Resolve the model name using strict or fuzzy matching.
        _resolve_encoding(): Resolve the token encoding for the selected model.
        _count_tokens(messages): Count tokens in a list of messages.
        _summarize(messages): Summarize a list of messages using the LLM (abstract).
        _maybe_summarize(target): Summarize conversation or decision history if token limit exceeded.
        save_LLM_state_to_file(filename): Save system prompts, conversation, and decisions to a file.
        load_LLM_state_from_file(filename): Load system prompts, conversation, and decisions from a file.
    Subclasses must implement:
        - is_connected
        - send
        - decide
        - _summarize
    Raises:
        ValueError: If initialization parameters are invalid.
        NotImplementedError: For abstract methods that must be implemented by subclasses.
        RuntimeError: If model or encoding resolution fails.
    """
    def __init__(self, 
                 base_url: str, 
                 api_key: str,
                 model: str = "default",
                 model_strict: bool = False,
                 max_tokens: int = 3000,
                 summarise_fraction: float = 0.5,
                 llm_kwargs: dict = None):
        
        
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._logger.debug(f"Debugging enabled for {self.__class__.__module__}.{self.__class__.__name__}")

        if not (0 < summarise_fraction < 1):
            self._logger.critical(f'summarise_fraction must be between 0 and 1, got {summarise_fraction}')
            raise ValueError(f'summarise_fraction must be between 0 and 1, got {summarise_fraction}')
        if max_tokens <= 0:
            self._logger.critical(f'max_tokens must be a positive integer, got {max_tokens}')
            raise ValueError(f'max_tokens must be a positive integer, got {max_tokens}')
        if not base_url:
            self._logger.critical("base_url must be provided and cannot be empty")
            raise ValueError("base_url must be provided and cannot be empty")
        if not api_key:
            self._logger.critical("api_key must be provided and cannot be empty")
            raise ValueError("api_key must be provided and cannot be empty")
        
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._selected_model = None
        self._model_strict = model_strict
        self._max_tokens = max_tokens
        self._summarise_fraction = summarise_fraction
        self._llm_kwargs = llm_kwargs
        if not self._llm_kwargs:
            self._logger.info(f"LLM kwargs not provided: using defaults")
            self._llm_kwargs = {
                "temperature": 0.7,
            } 
        self._system_prompts: List[str] = []
        self._models = []
        
        self._conversation: List[Dict[str, str]] = []
        self._decisions: List[Dict[str, str]] = []
        self._encoding = None


    def add_system_prompt(self, prompt: str) -> None:
        """Add a system prompt to the list."""
        self._system_prompts.append(prompt)

    def get_system_prompt(self, index: int) -> str:
        """Get a system prompt by index."""
        try:
            return self._system_prompts[index]
        except IndexError:
            return None

    def remove_system_prompt(self, index: int) -> None:
        """Remove a system prompt by index."""
        del self._system_prompts[index]

    def clear_system_prompts(self) -> None:
        """Clear all system prompts."""
        self._system_prompts.clear()

    def system_prompts_count(self) -> int:
        """Return the number of system prompts."""
        return len(self._system_prompts)

    def iter_system_prompts(self):
        """Iterate over system prompts."""
        return iter(self._system_prompts)

    def system_prompts_as_list(self) -> list:
        """Return a copy of the system prompts list."""
        return self._system_prompts.copy()

    @property
    def base_url(self):
        return self._base_url

    @property
    def api_key(self):
        return self._api_key
    
    @property
    def model(self):
        return self._model
    
    @property
    def selected_model(self):
        """Return the currently selected model."""
        if self._selected_model is None:
            self._logger.warning("No model has been selected yet.")
            return None
        return self._selected_model
    
    @property
    def models(self) -> List[str]:
        """Return the list of available models."""
        if not self._models:
            self._logger.warning("No models have been loaded yet.")
            return []
        return self._models.copy()
    
    @property
    def model_strict(self):
        return self._model_strict
    @property
    def max_tokens(self):
        return self._max_tokens
    @property
    def summarise_fraction(self):
        return self._summarise_fraction
    @property
    def llm_kwargs(self) -> Dict[str, Any]:
        """Return the LLM parameters as a dictionary."""
        return self._llm_kwargs 
    
    @property
    def is_connected(self) -> bool:
        raise NotImplementedError("LLMWrapper->is_connected not implemented")
    
    @abstractmethod
    def send(self, user_message: str, system_prompt: int = None) -> str:
        """Send a user message to the LLM and return the response."""
        raise NotImplementedError("LLMWrapper->send not implemented")
    
    @abstractmethod
    def decide(self, question: str, response_format: dict, system_prompt: int = None, retries: int = 3) -> str:
        """Make a decision based on the provided question and response format."""
        raise NotImplementedError("LLMWrapper->decide not implemented")
    
    def _resolve_model(self, substring_matching: bool = True) -> str:
        if not self._models:
            self._logger.critical(f"No models available to resolve against.")
            raise RuntimeError(f"No models available to resolve against.")

        if self._model_strict:
            self._logger.debug(f"Strict model matching is enabled, checking for exact match for '{self._model}'")
            if self._model not in self._models:
                self._logger.critical(f"Model '{self._model}' not found in LMStudio models: {self._models}")
                raise ValueError(f"Model '{self._model}' not found on LMStudio.")
            self._logger.debug(f"Using strict model matching, selected model: {self._model}")
            return self._model
        
        self._logger.debug(f"Model fuzzy matching enabled, searching for close matches for '{self._model}'")
        cutoff = 0.6
        if substring_matching:
            self._logger.debug(f"Substring matching is enabled, searching for models containing '{self._model}'")
            search_model_ids = self._models
            substring_matches = [m for m in self._models if self._model in m]
            self._logger.debug(f"Substring matches for '{self._model}':\n{''.join(f'\t{item}\n' for item in substring_matches) if substring_matches else 'Not Found'}")
            if substring_matches:
                if len(substring_matches) == 1:
                    self._logger.debug(f"Only one substring match found, using it: {substring_matches[0]}")
                    return substring_matches[0]
                self._logger.debug(f"Using fuzzy matching for substring matches for model resolution")
                search_model_ids = substring_matches
                cutoff = len(self._model)/ max(len(m) for m in substring_matches)

        match = get_close_matches(self._model, search_model_ids, n=1, cutoff=cutoff)
        if not match:
            self._logger.critical(f"Model '{self._model}' not found in LMStudio models: {search_model_ids}")
            raise ValueError(f"No similar model found for '{self._model}'")
        self._logger.debug(f"Using close match model, selected model: {match[0]}")
        return match[0]
    
    def _resolve_encoding(self) -> tiktoken.Encoding:
        """Resolve the encoding for the selected model."""
        if not self._selected_model:
            self._logger.critical("No model has been selected yet, cannot resolve encoding.")
            raise RuntimeError("No model has been selected yet, cannot resolve encoding.")
        
        try:
            encoding = tiktoken.encoding_for_model(self._selected_model)
            self._logger.debug(f"Encoding for model '{self._selected_model}' resolved successfully.")
            return encoding
        except Exception as e:
            self._logger.error(f"Failed to initialize encoding for model '{self._selected_model}': {e}")
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                self._logger.info("Using default encoding 'cl100k_base' for the model.")
                return encoding
            except Exception as e:
                self._logger.critical(f"Failed to initialize default encoding: {e}")
                raise RuntimeError("Failed to initialize encoding for the model and default encoding.") from e

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += len(self._encoding.encode(msg["content"]))
        return total

    @abstractmethod
    def _summarize(self, messages: List[Dict[str, str]]) -> str:
        """Summarize the provided messages using the LLM."""
        raise NotImplementedError("LLMWrapper->_summarize not implemented")

    def _maybe_summarize(self, target: str = "conversation"):
        if target not in ("conversation", "decision"):
            raise ValueError("target must be either 'conversation' or 'decision'")

        if target == "conversation":
            target_convo = self._conversation
        else:
            target_convo = self._decisions

        self._logger.debug(f"Checking if summarization is needed for {target} of length {len(target_convo)}")
        tokens = self._count_tokens(target_convo)
        self._logger.debug(f"Current token count: {tokens}, Max tokens allowed: {self._max_tokens}, Summarize fraction: {self._summarise_fraction}")
        if tokens < self._max_tokens:
            self._logger.debug("Token count is below max_tokens, no summarization needed.")
            return

        self._logger.debug("Token count exceeds max_tokens, summarizing.")
        cutoff = int(len(target_convo) * self._summarise_fraction)
        to_summarize = target_convo[:cutoff]
        summary = self._summarize(to_summarize)
        summarized = [{"role": "system", "content": f"Summary of earlier {target}: {summary}"}] + target_convo[cutoff:]

        if target == "conversation":
            self._conversation = summarized
        else:
            self._decisions = summarized

    def save_LLM_state_to_file(self, filename: str) -> None:
        data = {
            "system_prompts": self._system_prompts,
            "conversations": self._conversation,
            "decisions":self._decisions
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_LLM_state_from_file(self, filename: str)-> None:
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        self._system_prompts = data.get("system_prompts", [])
        self._conversation = data.get("conversation", [])
        self._decisions = data.get("decisions", [])
