'''
Filename: logging_config.py

Description:
Defines and applies a centralised logging configuration for the cyber-sim-core project.
Provides a standard logging format, console and file handlers, and logger settings
for consistent and flexible logging across the application.

Author: Daniel Prince
Version: 0.1

License: Apache v2.0 (see LICENSE file for details)
Copyright (c) 2025 [Lancaster University]
'''
import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'standard',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
    'loggers': {
        '__main__': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'pubsubnode.mqttpubsubnode.MQTTPubSubNode': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False
        },
        'langagent.agent.LanguageAgent': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'llm_system.lm_studio_wrapper.LMStudioManager': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
