from __future__ import annotations

from hok_ai.core.minutes_generator import HOK_MINUTES_SYSTEM_PROMPT, HOK_MINUTES_USER_TEMPLATE


DATASET_INSTRUCTION = f"""{HOK_MINUTES_SYSTEM_PROMPT}

{HOK_MINUTES_USER_TEMPLATE.replace('{transcript}', '[NYERS_LEIRAT]')}
"""
