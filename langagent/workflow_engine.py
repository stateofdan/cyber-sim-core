'''
Filename: workflow_engine.py

Description:
This module implements a flexible workflow engine for orchestrating tasks using
a block-based approach. 
It defines data models for extensible workflow blocks (such as decision and 
generation), their options, and provides a `WorkflowEngine` class that executes
a sequence of blocks based on user-defined decision and generation handler 
functions. The engine supports dynamic branching, confidence thresholds, and 
logging for workflow execution, and can be instantiated from a JSON definition.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''

import json
from pydantic import BaseModel, Field, TypeAdapter
from typing import Optional, Dict, Literal, Union, List, Annotated, Type, Callable, Tuple
import logging

class BaseBlock(BaseModel):
    """
    BaseBlock represents a fundamental unit in the workflow engine.

    Attributes:
        id (int): Unique identifier for the block.
        system_prompts (List[int]): List of system prompt identifiers to be used with the block.
        prompt (str): The main prompt or instruction for the block in an inja2 template format.
        type (Literal["decide", "generate"]): Specifies the block's function, either as a decision point or a content generator.
    """
    id: int 
    system_prompts: List[int]
    prompt: str
    type: Literal["decide", "generate"]

class DecisionOption(BaseModel):
    """
    Represents a decision option with an associated target and threshold.

    Attributes:
        target (Optional[int]): The identifier of the target entity for this decision option. Can be None if not applicable.
        threshold (float): A strict floating-point value representing the threshold for this decision option.
            Must be greater than 0.0 and less than or equal to 1.0.
    """
    target: Optional[int]
    threshold: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]

class DecideBlock(BaseBlock):
    """
    A workflow block that represents a decision point.

    Attributes:
        type (Literal["decide"]): Specifies the block type as "decide".
        options (Dict[str, DecisionOption]): A dictionary mapping option names to their corresponding DecisionOption objects, representing the possible choices at this decision point.
    """
    type: Literal["decide"]
    options: Dict[str, DecisionOption]

class GenerateBlock(BaseBlock):
    """
    Represents a block responsible for generating content or actions within a workflow.

    Attributes:
        type (Literal["generate"]): Specifies the type of the block as "generate".
        channels (List[str]): A list of channel names where the generated output will be sent.
        next (Union[int, None]): The index of the next block to execute, or None if this is the last block.
    """
    type: Literal["generate"]
    channels: List[str]
    next: Union[int, None]
    
Block = Union[DecideBlock, GenerateBlock]
"""A workflow block, which can be either a DecideBlock or GenerateBlock."""

DecideHandler = Callable[[str, List[str], List[int], List[str]], Tuple[str, float]]
"""A function handler to provide decision block functionality."""

GenerateHandler = Callable[[str, List[str], List[int], List[str]], None]
"""A function handler to provide generate block functionality."""

class WorkflowEngine:
    """
    WorkflowEngine manages and executes a sequence of workflow blocks, which can be either decision or generation steps.
    It processes a workflow defined by a list of blocks, using provided handler functions for decision and generation logic.
    Logging is supported for debugging and tracing workflow execution.
    Args:
        blocks (List[Block]): List of workflow blocks (DecideBlock or GenerateBlock) to execute.
        decide_fn (DecideHandler): Function to handle decision logic in DecideBlock.
        generate_fn (GenerateHandler): Function to handle generation logic in GenerateBlock.
        logger (logging.Logger, optional): Logger instance for debug and info messages. Defaults to None.
    """

    def __init__(self, blocks: List[Block], decide_fn: DecideHandler, generate_fn: GenerateHandler, logger: logging.Logger=None):
        """
        Initializes the WorkflowEngine with workflow blocks, decision and generation handlers, and an optional logger.
        Args:
            blocks (List[Block]): List of workflow blocks to execute.
            decide_fn (DecideHandler): Handler function for decision blocks.
            generate_fn (GenerateHandler): Handler function for generation blocks.
            logger (logging.Logger, optional): Logger for workflow execution messages. Defaults to None.
        """

        self.blocks = {block.id: block for block in blocks}
        self.decide_fn = decide_fn
        self.generate_fn = generate_fn
        self.logger = logger
        if self.logger is not None:
            self.logger.debug(f"logger loaded for WorkflowEngine: {self.logger.name}")

    def run(self, context_stack: List[str], start_id=1):
        """
        Executes the workflow starting from the specified block ID, processing each block in sequence.
        Decision blocks use the decide_fn to determine the next block, while generation blocks use the generate_fn.
        Args:
            context_stack (List[str]): Stack containing context information. 
                Expected structure:
                - context_stack[0]: Sender (str)
                - context_stack[1]: Message (str)
            start_id (int, optional): ID of the block to start execution from. Defaults to 1.
        Raises:
            ValueError: If a block with the specified ID does not exist or if context_stack is invalid.
        """
        if len(context_stack) < 2:
            raise ValueError("context_stack must contain at least two elements: sender and message.")
        #context_stack[0] should always be sender
        #context_stack[1] should always be the message
        current_id = start_id
        while current_id is not None:
            block = self.blocks.get(current_id)
            if not block:
                raise ValueError(f"No block with id {current_id}")

            if isinstance(block, DecideBlock):
                
                choice, confidence = self.decide_fn(block.prompt, block.system_prompts, list(block.options.keys()), context_stack)
                
                option = block.options.get(choice)
                if self.logger is not None:
                    self.logger.info(f'[WORKFLOW] Decide block success {choice}:{confidence}')
                if option and confidence >= option.threshold:
                    current_id = option.target
                else:
                    if self.logger is not None:
                        self.logger.debug(f"[WORKFLOW] Confidence {confidence:.2f} below threshold {option.threshold:.2f} → stopping")
                    current_id = None

            elif isinstance(block, GenerateBlock):
                
                self.generate_fn(block.prompt, block.system_prompts, block.channels, context_stack)
                current_id = block.next

            else:
                self.logger.critical(f'[WORKFLOW] Unknown type of block encountered id{current_id} type: {type(block)}')
                raise ValueError(f'[WORKFLOW] Unknown type of block encountered id{current_id} type: {type(block)}')
            
            if self.logger is not None:
                if current_id is None:
                    self.logger.debug(f"[WORKFLOW] Next Target is None → stopping")
                else:
                    self.logger.debug(f"[WORKFLOW] Next Target is {current_id}")

    @classmethod
    def from_json(cls: Type["WorkflowEngine"], json_str:str, decide_fn, generate_fn, logger: logging.Logger=None) -> "WorkflowEngine":
        """
        Creates a WorkflowEngine instance from a JSON string describing the workflow blocks.
        Args:
            json_str (str): JSON string representing the workflow blocks.
            decide_fn (DecideHandler): Handler function for decision blocks.
            generate_fn (GenerateHandler): Handler function for generation blocks.
            logger (logging.Logger, optional): Logger for workflow execution messages. Defaults to None.
        Returns:
            WorkflowEngine: An instance of WorkflowEngine initialized with the parsed blocks.
        Raises:
            ValueError: If an unknown block type is encountered in the JSON.
        """
        
        data = json.loads(json_str)
        blocks: List[Block] = TypeAdapter(List[Block]).validate_python(data)
        return cls(blocks, decide_fn, generate_fn, logger)

