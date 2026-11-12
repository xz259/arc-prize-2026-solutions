"""Explicit gameplay policy shared by counting, requests, and run manifests."""
from dataclasses import asdict, dataclass
import math
from fractions import Fraction

RESPONSE_TOKENS = 4096
# The user accepted the measured small throughput cost after correctness checks.
DEFAULT_MIN_P = 0.02


def proportional_headroom(context_limit, fraction=0.10):
    """Ceil(fraction * context), avoiding binary-float rounding at boundaries."""
    if (type(context_limit) is not int or context_limit <= 1
            or isinstance(fraction, bool) or not math.isfinite(fraction)
            or not 0 < fraction < 1):
        raise ValueError('Require integer context > 1 and finite fraction in (0, 1)')
    f = Fraction(str(fraction))
    result = (context_limit * f.numerator + f.denominator - 1) // f.denominator
    if result >= context_limit:
        raise ValueError('Proportional headroom leaves no input capacity')
    return result


@dataclass(frozen=True)
class GameplayConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = DEFAULT_MIN_P
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    enable_thinking: bool = True
    reasoning_effort: str = "xhigh"

    def __post_init__(self):
        if (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2
                or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1
                or self.top_k <= 0 or self.repetition_penalty <= 0
                or not -2 <= self.presence_penalty <= 2
                or not -2 <= self.frequency_penalty <= 2):
            raise ValueError("Invalid gameplay sampling parameters")
        if self.reasoning_effort not in {"low", "medium", "xhigh"}:
            raise ValueError("This model supports low, medium, xhigh reasoning")

    def template_kwargs(self):
        return dict(enable_thinking=self.enable_thinking,
                    reasoning_effort=self.reasoning_effort, preserve_thinking=True)

    def manifest(self):
        return dict(**asdict(self), response_tokens=RESPONSE_TOKENS,
                    reply_reserve_tokens=RESPONSE_TOKENS, assistant_turn_cap=None,
                    tool_steps=0, chat_template_kwargs=self.template_kwargs(),
                    min_p_target=self.min_p, min_p_note="Explicit request setting; initial public ablation uses 0.0, with 0.02 retained as a comparison",
                    input_counting="vLLM /tokenize; persistent server tokenizer and multimodal renderer")

    def chat_payload(self, model, messages, tools, *, max_tokens, seed=None):
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError("Output limit must be a positive integer or None for available context")
        payload = dict(model=model, messages=wire_messages(messages), stream=False,
                       **asdict(self))
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.pop("enable_thinking")
        payload.update(chat_template_kwargs=self.template_kwargs(),
                       add_generation_prompt=True, continue_final_message=False,
                       add_special_tokens=False)
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        if seed is not None and seed >= 0:
            payload["seed"] = seed
        return payload


def tokenization_payload(payload):
    """Only render-affecting fields accepted by this pinned /tokenize API.

    reasoning_effort is also in chat_template_kwargs, because TokenizeChatRequest
    has no top-level reasoning_effort field. The model template does not branch
    on tool_choice; gameplay uses auto. Both paths use add_special_tokens=False.
    """
    fields = ("model", "messages", "tools", "chat_template_kwargs",
              "add_generation_prompt", "continue_final_message", "add_special_tokens")
    return {key: payload[key] for key in fields if key in payload}


def validate_runtime_sampling(gameplay, server_argv, server_environment=None):
    # The original runtime lacks speculative min-p. The launcher records whether
    # our validated sampler patch is enabled; do not silently substitute zero.
    if (gameplay.min_p > 0 and "--speculative-config" in server_argv
            and (server_environment or {}).get("ARC3_MTP_MIN_P") != "1"):
        raise ValueError(
            "The running MTP server does not support min_p > 0 in its recorded configuration. "
            "Use --min-p 0, or install the validated patch and start with ARC3_MTP_MIN_P=1. "
            "No automatic substitution is performed.")


def wire_messages(messages):
    """Local observation/recovery identities never become chat API fields."""
    return [{k: v for k, v in message.items() if not k.startswith("_arc3_")}
            for message in messages]


@dataclass(frozen=True)
class ContextPolicy:
    context_limit: int = 32768
    response_headroom: int = 4096
    response_cap: int | None = 4096

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.context_limit, self.response_headroom))
                or not 0 < self.response_headroom < self.context_limit
                or (self.response_cap is not None and
                    (type(self.response_cap) is not int or not 0 < self.response_cap <= self.response_headroom))):
            raise ValueError("Require integer 0 < headroom < context and cap=None or 0 < cap <= headroom")

    @property
    def input_budget(self):
        return self.context_limit - self.response_headroom

    def manifest(self):
        return dict(**asdict(self), input_budget=self.input_budget)

    @classmethod
    def for_precision(cls, precision, context_limit, *, headroom=None, response_cap=None):
        # Explicit integer policy choices; context capacity is validated separately.
        reserves = {"bf16": 2048, "fp8_e4m3": 4096, "fp8_e5m2": 4096, "nvfp4": 6000}
        reserve = reserves[precision] if headroom is None else headroom
        cap = min(RESPONSE_TOKENS, reserve) if response_cap is None else response_cap
        return cls(context_limit, reserve, cap)
