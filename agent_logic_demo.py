from langagent.workflow_engine import WorkflowEngine
import json
from jinja2 import Template
from typing import Optional, Dict, Literal, Union, List, Annotated, Type, Callable, Tuple
import random


json_data = [
  {
    "id": 1,
    "type": "decide",
    "prompt": ("You have recieved the following message from: {{ sender }} which says:\n"
                "\n\n{{ message }}\n\n"
                "You need to decide whether to respond.\n"
                "Explain your reasoning and provide a confidence level in your decision.").strip(),
    "options": {
      "yes": { "target": 2, "threshold": 0.2 },
      "no": { "target": None, "threshold": 0.3 }
    }
  },
  {
    "id": 2,
    "type": "generate",
    "channels": ["direct_msg/{{ sender }}"],
    "prompt": ("You have recieved the following message from: {{ sender }} which says:\n"
                "\n\n{{ message }}\n\n"
                "You have decided to respond as you have reasoned: {{ context[-1] }}.\n"
                "Create an appropriate message to send back in response.\n"
                "only generate the message do not provide any other content.\n"
                "do not provide any blank or template like fields in this message.\n"
                "The message should be in the same tone as the orginal.").strip(),
    "next": 3
  },
  {
    "id": 3,
    "type": "decide",
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
    "channels": ["media/social"],
    "prompt": "Generating a social media post .",
    "next": None
  }
]

json_str = json.dumps(json_data)

new_context_idx = 0
contexts = [
    "Decision Reason 1",
    "Response Reason 1",
    "Response Text 1",
    "Decision Reason 2",
    "Resposne Reason 2",
    "Response Text 2",
    "An encrypted message was intercepted with unknown origin.",
    "Satellite imagery showed increased activity in the restricted zone.",
    None,
    "Weather conditions may delay drone deployment by 12 hours."
]
def decide_fn(prompt:str, options:List[str], context_stack:List[str]):
    print(f"sender: {context_stack[0]}\nMessage: {context_stack[1]}")
    prompt_template = Template(prompt)
    substitute_dict = {"sender": context_stack[0],
                       "message": context_stack[1],
                       "context": context_stack}
    prompt_render = prompt_template.render(**substitute_dict)

    print(f"[DECIDE] {prompt_render}\nOptions: {options}")

    choice = random.choice(options)
    confidence = random.uniform(0.0, 1.0)
    
    print(f"  → Chose '{choice}' with confidence {confidence:.2f}")
    global new_context_idx
    context_stack.append(contexts[new_context_idx])
    new_context_idx = new_context_idx + 1
    
    return choice, confidence

def generate_fn(prompt:str, channels: List[str], context_stack: List[str]):
    print(f"sender: {context_stack[0]}\nMessage: {context_stack[1]}")
    prompt_template = Template(prompt)
    substitute_dict = {"sender": context_stack[0],
                       "message": context_stack[1],
                       "context": context_stack}
    prompt_render = prompt_template.render(**substitute_dict)

    channels_rendered = []
    for channel in channels:
        channels_rendered.append(Template(channel).render(sender=context_stack[0]))
    
    print(f"[GENERATE] {prompt_render}")
    global new_context_idx
    context_stack.append(contexts[new_context_idx])
    new_context_idx = new_context_idx + 1
    context_stack.append(contexts[new_context_idx])
    msg = context_stack[-1]
    for channel in channels_rendered:
        print(f'[GENERATE] sending message to {channel}: {msg}')
    new_context_idx = new_context_idx + 1
    

if __name__ == "__main__":
    context_stack = ["agent_1", "I have found this thing and it is really weird."]

    engine = WorkflowEngine.from_json(json_str, decide_fn, generate_fn)
    engine.run(context_stack)
    print (json.dumps(context_stack, indent=2))
