from enum import Enum

from app.schemas.store_policy import OperatingMode, StorePolicy


class ActionRequirement(str, Enum):
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    AUTOMATIC_EXECUTION_ELIGIBLE = "AUTOMATIC_EXECUTION_ELIGIBLE"


def get_action_requirement(policy: StorePolicy) -> ActionRequirement:
    """Translates a backend StorePolicy into a deterministic workflow action requirement.
    
    Contains zero LLM calls.
    """
    if not isinstance(policy, StorePolicy):
        raise TypeError("policy must be an instance of StorePolicy.")

    if policy.operating_mode == OperatingMode.ASSISTED:
        return ActionRequirement.APPROVAL_REQUIRED
    elif policy.operating_mode == OperatingMode.AUTONOMOUS:
        return ActionRequirement.AUTOMATIC_EXECUTION_ELIGIBLE
    else:
        raise ValueError(f"Unsupported operating mode: {policy.operating_mode}")


def get_action_reason(policy: StorePolicy | None) -> str | None:
    """Returns a deterministic action policy rationale based on store operating mode.
    
    Contains zero LLM calls.
    """
    if policy is None:
        return None
    if not isinstance(policy, StorePolicy):
        raise TypeError("policy must be an instance of StorePolicy or None.")

    if policy.operating_mode == OperatingMode.ASSISTED:
        return "Store operates in assisted mode; explicit owner approval is required before execution."
    elif policy.operating_mode == OperatingMode.AUTONOMOUS:
        return "Store operates in autonomous mode; the recommendation is eligible for downstream automatic execution subject to backend safety validation."
    else:
        raise ValueError(f"Unsupported operating mode: {policy.operating_mode}")
