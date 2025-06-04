'''
Filename: decision_format.py

Description:
Defines structured response formats for LLM-based decision making in 
cyber-sim-core.

This module provides Pydantic models for decision responses, including
free-text and choice-based formats, with schema serialization utilities.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
from pydantic import BaseModel, Field, field_validator, create_model
from typing import List, Dict, Any, Optional, Literal, Type
import json


# https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat&format=without-parse
# https://cookbook.openai.com/examples/structured_outputs_intro
# https://cookbook.openai.com/examples/structured_outputs_multi_agent
# Base class
class DecisionResponseFormat(BaseModel):
    explanation: str = Field(..., description="Explanation of how the decision was reached.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level between 0 and 1.")

    @classmethod
    def json_schema_str(cls, **kwargs) -> str:
        schema_dict = cls.model_json_schema()
        schema_dict['additionalProperties'] = False
        return json.dumps(schema_dict, **kwargs)
    
    @classmethod
    def request_format_json_str(cls, **kwargs) -> str:
        """Return the JSON request format string for this decision response format."""
        schema_dict = cls.model_json_schema()
        schema_dict['additionalProperties'] = False
        request_format = {
            "type": "json_schema",
            "json_schema":{
                "name": cls.__name__,
                "schema": schema_dict,
                "strict": True,
            }
        }
        return json.dumps(request_format, **kwargs)
    
    @classmethod
    def create_request_format(cls) -> Dict[str, Any]:
        """Return the JSON request format string for this decision response format."""
        schema_dict = cls.model_json_schema()
        schema_dict['additionalProperties'] = False
        request_format = {
            "type": "json_schema",
            "json_schema":{
                "name": cls.__name__,
                "schema": schema_dict,
                "strict": True,
            }
        }
        return request_format
    

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
        return v


# Free text subclass
class TextDecisionFormat(DecisionResponseFormat):
    decision: str = Field(..., description="An open-ended decision provided as free text.")


# Dynamic enum-based subclass factory
def create_choice_decision_format_class(options: List[str]) -> Type[DecisionResponseFormat]:
    """Factory to create a dynamic subclass with enum-like choices for decision field."""
    literal_type = Literal[tuple(options)]
    fields = {
        "decision": (literal_type, Field(..., description="Decision chosen from a list of options"))
    }
    name = f"ChoiceDecisionFormat_{'_'.join(options)}"
    dynamic_model = create_model(
        name,
        __base__=DecisionResponseFormat,
        **fields
    )
    return dynamic_model