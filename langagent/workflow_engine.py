import json
from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, Union, List, Annotated, Type, Callable, Tuple
import logging



class BaseBlock(BaseModel):
    id: int
    prompt: str
    type: Literal["decide", "generate"]

class DecisionOption(BaseModel):
    target: Optional[int]
    threshold: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]

class DecideBlock(BaseBlock):
    type: Literal["decide"]
    options: Dict[str, DecisionOption]

class GenerateBlock(BaseBlock):
    type: Literal["generate"]
    channels: List[str]
    next: Union[int, None]
    
Block = Union[DecideBlock, GenerateBlock]

DecideHandler = Callable[[str, List[str], List[str]], Tuple[str, float]]
GenerateHandler = Callable[[str, List[str], List[str]], None]

class WorkflowEngine:
    def __init__(self, blocks: List[Block], decide_fn: DecideHandler, generate_fn: GenerateHandler, logger: logging.Logger=None):
        self.blocks = {block.id: block for block in blocks}
        self.decide_fn = decide_fn
        self.generate_fn = generate_fn
        self.logger = logger
        if self.logger is not None:
            self.logger.debug(f"logger loaded for WorkflowEngine: {self.logger.name}")

    def run(self, context_stack: List[str], start_id=1):
        #context_stack[0] should always be sender
        #context_stack[1] should always be the message
        current_id = start_id
        while current_id is not None:
            block = self.blocks.get(current_id)
            if not block:
                raise ValueError(f"No block with id {current_id}")

            if isinstance(block, DecideBlock):
                
                choice, confidence = self.decide_fn(block.prompt, list(block.options.keys()), context_stack)
                
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
                
                self.generate_fn(block.prompt, block.channels, context_stack)
                current_id = block.next
            
            if self.logger is not None:
                if current_id is None:
                    self.logger.debug(f"[WORKFLOW] Next Target is None → stopping")
                else:
                    self.logger.debug(f"[WORKFLOW] Next Target is {current_id}")

    @classmethod
    def from_json(cls: Type["WorkflowEngine"], json_str:str, decide_fn, generate_fn, logger: logging.Logger=None) -> "WorkflowEngine":
        data = json.loads(json_str)
        blocks: List[Block] = []
        for item in data:
            if item["type"] == "decide":
                blocks.append(DecideBlock(**item))
            elif item["type"] == "generate":
                blocks.append(GenerateBlock(**item))
            else:
                raise ValueError(f"Unknown block type: {item['type']}")
        return cls(blocks, decide_fn, generate_fn, logger)

