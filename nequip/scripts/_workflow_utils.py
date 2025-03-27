from typing import List

_WORKFLOW_STATE: str = None

_ALLOWED_STATES: List[str] = ["train", "package", "compile", None]


def set_workflow_state(state: str):
    assert state in _ALLOWED_STATES
    global _WORKFLOW_STATE
    _WORKFLOW_STATE = state


def get_workflow_state():
    global _WORKFLOW_STATE
    return _WORKFLOW_STATE
