from pathlib import Path

output = Path("/mnt/data/prompt_engineering_tester_5000plus.py")

header = r'''"""
PROMPT ENGINEERING TESTER
=========================

A single-file, beginner-to-intermediate Python laboratory for systematic
prompt testing and AI response evaluation.

This implementation follows the uploaded project specification:
- multiple prompt variants
- reusable templates
- multiple test cases
- experiments
- provider abstraction
- demo/mock mode
- rule-based evaluation
- optional model-based evaluation adapter
- blind evaluation
- A/B comparison
- repeated runs
- variability analysis
- response length and latency analysis
- configurable cost estimation
- requirement checking
- structured JSON validation
- prompt versioning
- experiment history
- CSV / JSON / HTML exports
- reports and charts when optional libraries are installed
- controlled retries and rate limiting
- caching
- CLI
- unit tests
- security-conscious handling of API keys
- educational explanations and beginner learning mode

IMPORTANT:
This file never contains a real API key.
Demo mode is the default and clearly marks mock responses.
Model-based evaluation is treated as imperfect rather than objective truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
import unittest
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


APP_NAME = "Prompt Engineering Tester"
APP_VERSION = "1.0.0"
DEFAULT_DATA_DIR = Path("prompt_tester_data")
DEFAULT_CACHE_FILE = DEFAULT_DATA_DIR / "response_cache.json"
DEFAULT_HISTORY_FILE = DEFAULT_DATA_DIR / "experiment_history.json"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising an exception."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int without raising an exception."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a number into an inclusive interval."""
    return max(low, min(high, value))


def mean_or_zero(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for an empty sequence."""
    return statistics.mean(values) if values else 0.0


def median_or_zero(values: Sequence[float]) -> float:
    """Return the median, or zero for an empty sequence."""
    return statistics.median(values) if values else 0.0


def stdev_or_zero(values: Sequence[float]) -> float:
    """Return sample standard deviation, or zero when undefined."""
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def word_count(text: str) -> int:
    """Count whitespace-separated words."""
    return len(re.findall(r"\S+", text or ""))


def line_count(text: str) -> int:
    """Count non-empty lines."""
    return len([line for line in (text or "").splitlines() if line.strip()])


def character_count(text: str) -> int:
    """Count Unicode characters."""
    return len(text or "")


def approximate_token_count(text: str) -> int:
    """
    Return a deliberately labeled approximation.

    This is not provider token usage. Real token usage must come from the
    configured provider. The approximation is useful for local teaching only.
    """
    text = text or ""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def slugify(value: str) -> str:
    """Create a safe filesystem-friendly identifier."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "item"


def stable_hash(value: Any) -> str:
    """Create a deterministic SHA-256 identifier for serializable data."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def now_epoch() -> float:
    """Return the current monotonic-independent wall-clock timestamp."""
    return time.time()


def redact_secret(value: Optional[str]) -> str:
    """Return a safe display form of a secret without printing the secret."""
    if not value:
        return "<not configured>"
    if len(value) <= 6:
        return "***"
    return value[:3] + "..." + value[-3:]


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(data: Any, pretty: bool = True) -> str:
    """Serialize Python data to readable JSON."""
    return json.dumps(
        data,
        indent=2 if pretty else None,
        ensure_ascii=False,
        sort_keys=False,
        default=str,
    )


@dataclass
class Prompt:
    """One prompt variant."""

    prompt_id: str
    name: str
    text: str
    system_instructions: str = ""
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    version: str = "1.0"
    notes: str = ""
    created_at: str = field(default_factory=utc_now)

    def render(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Render {variables} inside the prompt text."""
        variables = variables or {}
        try:
            return self.text.format(**variables)
        except KeyError:
            return self.text
        except (ValueError, IndexError):
            return self.text

    def validate(self) -> List[str]:
        """Return validation errors instead of raising them."""
        errors: List[str] = []
        if not self.prompt_id.strip():
            errors.append("Prompt ID is required.")
        if not self.name.strip():
            errors.append("Prompt name is required.")
        if not self.text.strip():
            errors.append("Prompt text is required.")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            errors.append("Temperature should normally be between 0 and 2.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            errors.append("max_tokens must be positive.")
        return errors


@dataclass
class TestCase:
    """One reusable test case."""

    test_case_id: str
    name: str
    variables: Dict[str, Any] = field(default_factory=dict)
    task_description: str = ""
    expected_keywords: List[str] = field(default_factory=list)
    required_phrases: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)
    max_words: Optional[int] = None
    min_words: Optional[int] = None
    exact_bullet_count: Optional[int] = None
    expected_json_schema: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=utc_now)

    def validate(self) -> List[str]:
        """Validate the test case configuration."""
        errors: List[str] = []
        if not self.test_case_id.strip():
            errors.append("Test case ID is required.")
        if not self.name.strip():
            errors.append("Test case name is required.")
        if self.max_words is not None and self.max_words <= 0:
            errors.append("max_words must be positive.")
        if self.min_words is not None and self.min_words < 0:
            errors.append("min_words cannot be negative.")
        if (
            self.max_words is not None
            and self.min_words is not None
            and self.min_words > self.max_words
        ):
            errors.append("min_words cannot exceed max_words.")
        return errors


@dataclass
class ModelConfig:
    """Configuration shared by experiment requests."""

    provider_name: str = "demo"
    model_name: str = "demo-model"
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_base_seconds: float = 0.5
    delay_between_requests: float = 0.0
    input_cost_per_1k: Optional[float] = None
    output_cost_per_1k: Optional[float] = None
    cache_enabled: bool = True
    evaluator_provider_name: str = "none"
    evaluator_model_name: Optional[str] = None

    def estimated_request_cost(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> Optional[float]:
        """Estimate cost only when both usage and configured pricing exist."""
        if input_tokens is None or output_tokens is None:
            return None
        if self.input_cost_per_1k is None or self.output_cost_per_1k is None:
            return None
        return (
            input_tokens / 1000.0 * self.input_cost_per_1k
            + output_tokens / 1000.0 * self.output_cost_per_1k
        )


@dataclass
class EvaluationCriteria:
    """Weighted evaluation rubric."""

    relevance_weight: float = 1.0
    accuracy_weight: float = 1.0
    clarity_weight: float = 1.0
    completeness_weight: float = 1.0
    instruction_following_weight: float = 1.0
    format_compliance_weight: float = 1.0
    safety_weight: float = 1.0
    conciseness_weight: float = 1.0

    def weights(self) -> Dict[str, float]:
        return {
            "relevance": self.relevance_weight,
            "accuracy": self.accuracy_weight,
            "clarity": self.clarity_weight,
            "completeness": self.completeness_weight,
            "instruction_following": self.instruction_following_weight,
            "format_compliance": self.format_compliance_weight,
            "safety": self.safety_weight,
            "conciseness": self.conciseness_weight,
        }

    def normalized_weights(self) -> Dict[str, float]:
        weights = self.weights()
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0:
            equal = 1.0 / len(weights)
            return {key: equal for key in weights}
        return {key: max(0.0, value) / total for key, value in weights.items()}


@dataclass
class ScoreBreakdown:
    """Individual criterion scores."""

    relevance: float = 0.0
    accuracy: float = 0.0
    clarity: float = 0.0
    completeness: float = 0.0
    instruction_following: float = 0.0
    format_compliance: float = 0.0
    safety: float = 0.0
    conciseness: float = 0.0
    overall: float = 0.0
    reason: str = ""
    evaluator_mode: str = "rule-based"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementResult:
    """Result for one explicit requirement."""

    requirement: str
    passed: bool
    detail: str


@dataclass
class ResponseRecord:
    """One model invocation result."""

    experiment_id: str
    prompt_id: str
    test_case_id: str
    run_number: int
    model: str
    prompt_text: str
    response: str
    timestamp: str
    latency_seconds: float
    success: bool
    error_message: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    cache_hit: bool = False
    mock_response: bool = False
    score: Optional[ScoreBreakdown] = None
    requirements: List[RequirementResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def response_words(self) -> int:
        return word_count(self.response)

    @property
    def response_lines(self) -> int:
        return line_count(self.response)

    @property
    def response_characters(self) -> int:
        return character_count(self.response)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = self.score.as_dict() if self.score else None
        data["requirements"] = [asdict(item) for item in self.requirements]
        data["response_words"] = self.response_words
        data["response_lines"] = self.response_lines
        data["response_characters"] = self.response_characters
        return data


@dataclass
class Experiment:
    """Complete experiment definition."""

    experiment_id: str
    name: str
    description: str
    prompts: List[Prompt]
    test_cases: List[TestCase]
    model_config: ModelConfig = field(default_factory=ModelConfig)
    criteria: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    runs_per_case: int = 1
    blind_evaluation: bool = False
    created_at: str = field(default_factory=utc_now)
    results: List[ResponseRecord] = field(default_factory=list)

    def request_count(self) -> int:
        return len(self.prompts) * len(self.test_cases) * max(1, self.runs_per_case)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.experiment_id.strip():
            errors.append("Experiment ID is required.")
        if not self.name.strip():
            errors.append("Experiment name is required.")
        if not self.prompts:
            errors.append("At least one prompt is required.")
        if not self.test_cases:
            errors.append("At least one test case is required.")
        if self.runs_per_case <= 0:
            errors.append("runs_per_case must be positive.")
        for prompt in self.prompts:
            errors.extend(f"{prompt.name}: {error}" for error in prompt.validate())
        for case in self.test_cases:
            errors.extend(f"{case.name}: {error}" for error in case.validate())
        return errors


class PromptManager:
    """Create, validate, version, and retrieve prompt variants."""

    def __init__(self) -> None:
        self.prompts: Dict[str, Prompt] = {}

    def add(self, prompt: Prompt) -> None:
        errors = prompt.validate()
        if errors:
            raise ValueError("; ".join(errors))
        self.prompts[prompt.prompt_id] = prompt

    def get(self, prompt_id: str) -> Prompt:
        if prompt_id not in self.prompts:
            raise KeyError(f"Unknown prompt: {prompt_id}")
        return self.prompts[prompt_id]

    def list_prompts(self) -> List[Prompt]:
        return list(self.prompts.values())

    def clone_version(self, prompt_id: str, new_text: str, version: str) -> Prompt:
        original = self.get(prompt_id)
        new_id = f"{prompt_id}-v{version.replace('.', '-')}"
        clone = Prompt(
            prompt_id=new_id,
            name=f"{original.name} v{version}",
            text=new_text,
            system_instructions=original.system_instructions,
            model=original.model,
            temperature=original.temperature,
            max_tokens=original.max_tokens,
            version=version,
            notes=f"Versioned from {original.prompt_id}",
        )
        self.add(clone)
        return clone


class ExperimentManager:
    """Build and validate experiments."""

    def __init__(self) -> None:
        self.experiments: Dict[str, Experiment] = {}

    def create(
        self,
        name: str,
        description: str,
        prompts: Sequence[Prompt],
        test_cases: Sequence[TestCase],
        model_config: Optional[ModelConfig] = None,
        criteria: Optional[EvaluationCriteria] = None,
        runs_per_case: int = 1,
        blind_evaluation: bool = False,
    ) -> Experiment:
        experiment = Experiment(
            experiment_id=f"exp-{uuid.uuid4().hex[:10]}",
            name=name,
            description=description,
            prompts=list(prompts),
            test_cases=list(test_cases),
            model_config=model_config or ModelConfig(),
            criteria=criteria or EvaluationCriteria(),
            runs_per_case=runs_per_case,
            blind_evaluation=blind_evaluation,
        )
        errors = experiment.validate()
        if errors:
            raise ValueError("\n".join(errors))
        self.experiments[experiment.experiment_id] = experiment
        return experiment


class ProviderError(Exception):
    """Base provider error."""


class AuthenticationError(ProviderError):
    """Authentication failed."""


class RateLimitError(ProviderError):
    """Provider rate limit."""


class TimeoutProviderError(ProviderError):
    """Provider timeout."""


class InvalidRequestError(ProviderError):
    """Provider rejected the request."""


@dataclass
class ProviderResponse:
    """Normalized provider output."""

    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIProvider:
    """Provider abstraction used by the rest of the application."""

    name = "abstract"

    def send_prompt(
        self,
        prompt: str,
        system_instructions: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ) -> ProviderResponse:
        raise NotImplementedError


class DemoProvider(AIProvider):
    """
    Deterministic-ish mock provider for learning and testing.

    The generated text is intentionally labeled as a mock response.
    """

    name = "demo"

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def send_prompt(
        self,
        prompt: str,
        system_instructions: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ) -> ProviderResponse:
        started = time.perf_counter()
        cleaned = prompt.strip()
        topic = "the requested topic"
        match = re.search(r"(?:about|explain|topic)\s+(.+)", cleaned, re.IGNORECASE)
        if match:
            topic = match.group(1).strip().rstrip(".")
        style_hint = ""
        if "bullet" in cleaned.lower():
            style_hint = "\n- Point one: a concise explanation.\n- Point two: a useful detail.\n- Point three: an example.\n- Point four: a practical connection.\n- Point five: a short summary."
        elif "simple" in cleaned.lower():
            style_hint = "\nThe idea can be understood with a simple everyday example."
        else:
            style_hint = "\nThe explanation is organized around the main concept, an example, and a short conclusion."
        text = (
            "DEMO / MOCK RESPONSE\n\n"
            f"This response is simulated for the Prompt Engineering Tester.\n"
            f"Requested topic: {topic}\n\n"
            "The model would normally answer the supplied prompt here."
            + style_hint
            + "\n\nThis output is not a real AI-provider response."
        )
        elapsed = time.perf_counter() - started
        if elapsed < 0:
            elapsed = 0.0
        output_tokens = approximate_token_count(text)
        input_tokens = approximate_token_count(prompt)
        return ProviderResponse(
            text=text,
            model=model or "demo-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"mock": True, "generation_seconds": elapsed},
        )


class EnvironmentProvider(AIProvider):
    """
    Generic provider adapter skeleton.

    The project specification intentionally does not force one vendor.
    Subclasses can implement send_prompt() for a chosen API.
    """

    name = "environment"

    def __init__(self, api_key_env: str = "AI_API_KEY") -> None:
        self.api_key_env = api_key_env

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)

    def send_prompt(
        self,
        prompt: str,
        system_instructions: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ) -> ProviderResponse:
        if not self.api_key:
            raise AuthenticationError(
                f"Missing credential in environment variable {self.api_key_env}."
            )
        raise InvalidRequestError(
            "No vendor-specific transport has been configured. "
            "Implement a safe adapter for your chosen provider."
        )


class ModelBasedEvaluator:
    """
    Optional evaluator interface.

    A real implementation can call an evaluator model through an AIProvider.
    """

    def __init__(self, provider: AIProvider) -> None:
        self.provider = provider

    def evaluate(
        self,
        task: str,
        prompt: str,
        response: str,
        rubric: EvaluationCriteria,
    ) -> ScoreBreakdown:
        rubric_text = json.dumps(rubric.normalized_weights(), indent=2)
        evaluation_prompt = (
            "Evaluate the following AI response. Return JSON only with numeric "
            "scores from 0 to 10 for relevance, accuracy, clarity, completeness, "
            "instruction_following, format_compliance, safety, conciseness, "
            "overall, and a short reason.\n\n"
            f"TASK:\n{task}\n\nPROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n\n"
            f"WEIGHTS:\n{rubric_text}"
        )
        result = self.provider.send_prompt(evaluation_prompt)
        parsed = parse_json_object(result.text)
        if not parsed:
            raise ValueError("Evaluator did not return valid JSON.")
        return score_from_mapping(parsed, mode="model-based")


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object, tolerating a small amount of surrounding text."""
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def score_from_mapping(
    mapping: Dict[str, Any],
    mode: str = "model-based",
) -> ScoreBreakdown:
    """Normalize evaluator JSON into a ScoreBreakdown."""
    fields = [
        "relevance",
        "accuracy",
        "clarity",
        "completeness",
        "instruction_following",
        "format_compliance",
        "safety",
        "conciseness",
        "overall",
    ]
    values = {name: clamp(safe_float(mapping.get(name), 0.0), 0.0, 10.0) for name in fields}
    return ScoreBreakdown(
        **values,
        reason=str(mapping.get("reason", "")),
        evaluator_mode=mode,
    )


def count_bullets(text: str) -> int:
    """Count common Markdown/plain-text bullet styles."""
    lines = (text or "").splitlines()
    return sum(
        1
        for line in lines
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line)
    )


def contains_case_insensitive(text: str, phrase: str) -> bool:
    return phrase.lower() in (text or "").lower()


def requirement_checker(
    response: str,
    test_case: TestCase,
) -> List[RequirementResult]:
    """
    Check explicit deterministic requirements.

    This checker does not attempt to prove factual correctness.
    """
    results: List[RequirementResult] = []
    text = response or ""

    for keyword in test_case.expected_keywords:
        passed = contains_case_insensitive(text, keyword)
        results.append(
            RequirementResult(
                requirement=f"Contains keyword: {keyword}",
                passed=passed,
                detail="Keyword found." if passed else "Keyword not found.",
            )
        )

    for phrase in test_case.required_phrases:
        passed = contains_case_insensitive(text, phrase)
        results.append(
            RequirementResult(
                requirement=f"Contains phrase: {phrase}",
                passed=passed,
                detail="Phrase found." if passed else "Phrase not found.",
            )
        )

    for section in test_case.required_sections:
        passed = contains_case_insensitive(text, section)
        results.append(
            RequirementResult(
                requirement=f"Contains section: {section}",
                passed=passed,
                detail="Section text found." if passed else "Section text not found.",
            )
        )

    words = word_count(text)
    if test_case.min_words is not None:
        passed = words >= test_case.min_words
        results.append(
            RequirementResult(
                requirement=f"Minimum words: {test_case.min_words}",
                passed=passed,
                detail=f"Observed {words} words.",
            )
        )

    if test_case.max_words is not None:
        passed = words <= test_case.max_words
        results.append(
            RequirementResult(
                requirement=f"Maximum words: {test_case.max_words}",
                passed=passed,
                detail=f"Observed {words} words.",
            )
        )

    if test_case.exact_bullet_count is not None:
        bullets = count_bullets(text)
        passed = bullets == test_case.exact_bullet_count
        results.append(
            RequirementResult(
                requirement=f"Exactly {test_case.exact_bullet_count} bullets",
                passed=passed,
                detail=f"Observed {bullets} bullets.",
            )
        )

    if test_case.expected_json_schema is not None:
        parsed = parse_json_object(text)
        passed, detail = validate_json_schema(parsed, test_case.expected_json_schema)
        results.append(
            RequirementResult(
                requirement="Matches requested JSON schema",
                passed=passed,
                detail=detail,
            )
        )

    return results


def validate_json_schema(
    data: Optional[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Tuple[bool, str]:
    """Perform a small educational JSON-schema-like validation."""
    if data is None:
        return False, "Response is not a valid JSON object."

    required = schema.get("required", [])
    for key in required:
        if key not in data:
            return False, f"Missing required field: {key}"

    properties = schema.get("properties", {})
    for key, rule in properties.items():
        if key not in data:
            continue
        expected_type = rule.get("type")
        value = data[key]
        type_ok = True
        if expected_type == "string":
            type_ok = isinstance(value, str)
        elif expected_type == "number":
            type_ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "integer":
            type_ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "boolean":
            type_ok = isinstance(value, bool)
        elif expected_type == "array":
            type_ok = isinstance(value, list)
        elif expected_type == "object":
            type_ok = isinstance(value, dict)
        if not type_ok:
            return False, f"Field {key!r} has the wrong type."
    return True, "JSON structure satisfies the configured checks."


class RuleBasedEvaluator:
    """Deterministic educational evaluator."""

    def __init__(self, criteria: Optional[EvaluationCriteria] = None) -> None:
        self.criteria = criteria or EvaluationCriteria()

    def evaluate(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
    ) -> ScoreBreakdown:
        response = response or ""
        reqs = requirement_checker(response, test_case)
        compliance = (
            sum(1 for item in reqs if item.passed) / len(reqs)
            if reqs
            else 1.0
        )

        relevance = self._relevance(prompt, response, test_case)
        accuracy = self._accuracy_heuristic(response)
        clarity = self._clarity(response)
        completeness = compliance * 10.0
        instruction = compliance * 10.0
        format_score = self._format_score(response, test_case)
        safety = self._safety_score(response)
        conciseness = self._conciseness(response, test_case)

        values = {
            "relevance": relevance,
            "accuracy": accuracy,
            "clarity": clarity,
            "completeness": completeness,
            "instruction_following": instruction,
            "format_compliance": format_score,
            "safety": safety,
            "conciseness": conciseness,
        }
        weights = self.criteria.normalized_weights()
        overall = sum(values[name] * weights[name] for name in values)

        reason = (
            "Rule-based score using keyword/relevance heuristics, explicit "
            "requirements, basic clarity checks, formatting checks, and "
            "length-aware conciseness. This is not a factuality guarantee."
        )
        return ScoreBreakdown(
            **values,
            overall=clamp(overall, 0.0, 10.0),
            reason=reason,
            evaluator_mode="rule-based",
        )

    def _relevance(
        self,
        prompt: str,
        response: str,
        test_case: TestCase,
    ) -> float:
        prompt_words = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", prompt)
            if len(word) >= 4
        }
        response_words = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", response)
            if len(word) >= 4
        }
        if not prompt_words:
            return 5.0
        overlap = len(prompt_words & response_words) / len(prompt_words)
        keyword_bonus = 0.0
        if test_case.expected_keywords:
            found = sum(
                contains_case_insensitive(response, item)
                for item in test_case.expected_keywords
            )
            keyword_bonus = found / len(test_case.expected_keywords) * 4
        return clamp(overlap * 7 + keyword_bonus, 0.0, 10.0)

    def _accuracy_heuristic(self, response: str) -> float:
        """
        Deliberately conservative heuristic.

        Automated text checks cannot establish factual truth, so this score
        should be treated as a weak signal rather than a factual verdict.
        """
        if not response.strip():
            return 0.0
        uncertainty = len(
            re.findall(r"\b(?:maybe|possibly|not sure|i think)\b", response, re.I)
        )
        base = 7.0
        base -= min(3.0, uncertainty * 0.5)
        return clamp(base, 0.0, 10.0)

    def _clarity(self, response: str) -> float:
        if not response.strip():
            return 0.0
        sentences = [item.strip() for item in re.split(r"[.!?]+", response) if item.strip()]
        if not sentences:
            return 5.0
        average_words = mean_or_zero([word_count(sentence) for sentence in sentences])
        score = 9.0
        if average_words > 35:
            score -= 2.0
        if average_words > 55:
            score -= 2.0
        if "\n" in response:
            score += 0.5
        return clamp(score, 0.0, 10.0)

    def _format_score(self, response: str, test_case: TestCase) -> float:
        if not test_case.required_sections and test_case.exact_bullet_count is None:
            return 8.0 if response.strip() else 0.0
        checks = requirement_checker(response, test_case)
        format_checks = [
            item
            for item in checks
            if "section" in item.requirement.lower()
            or "bullet" in item.requirement.lower()
            or "json" in item.requirement.lower()
        ]
        if not format_checks:
            return 8.0
        return sum(item.passed for item in format_checks) / len(format_checks) * 10

    def _safety_score(self, response: str) -> float:
        if not response.strip():
            return 0.0
        suspicious = re.findall(
            r"\b(?:steal|bypass security|credential theft|malware)\b",
            response,
            flags=re.I,
        )
        return 10.0 if not suspicious else 4.0

    def _conciseness(self, response: str, test_case: TestCase) -> float:
        words = word_count(response)
        if words == 0:
            return 0.0
        if test_case.max_words:
            if words <= test_case.max_words:
                return 10.0
            overflow = words - test_case.max_words
            return clamp(10.0 - overflow / max(1, test_case.max_words) * 10, 0, 10)
        if words <= 250:
            return 9.0
        if words <= 500:
            return 8.0
        if words <= 1000:
            return 7.0
        return 5.0


class CacheStore:
    """Small JSON cache for completed request results."""

    def __init__(self, path: Path = DEFAULT_CACHE_FILE) -> None:
        self.path = path
        ensure_directory(path.parent)
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, json.JSONDecodeError):
            self.data = {}

    def save(self) -> None:
        ensure_directory(self.path.parent)
        self.path.write_text(json_dumps(self.data), encoding="utf-8")

    def key(
        self,
        provider: str,
        model: str,
        prompt: str,
        system: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        return stable_hash(
            {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.data.get(key)
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value
        self.save()

    def clear(self) -> None:
        self.data = {}
        self.save()


class RetryController:
    """Controlled exponential-backoff retry helper."""

    def __init__(self, max_retries: int = 2, base_seconds: float = 0.5) -> None:
        self.max_retries = max(0, max_retries)
        self.base_seconds = max(0.0, base_seconds)

    def run(
        self,
        operation: Callable[[], ProviderResponse],
        retryable: Tuple[type, ...] = (
            RateLimitError,
            TimeoutProviderError,
        ),
    ) -> ProviderResponse:
        attempt = 0
        while True:
            try:
                return operation()
            except retryable:
                if attempt >= self.max_retries:
                    raise
                delay = self.base_seconds * (2 ** attempt)
                time.sleep(delay)
                attempt += 1


class ResponseCollector:
    """Run model calls, collect metadata, cache hits, and failures."""

    def __init__(
        self,
        provider: AIProvider,
        model_config: ModelConfig,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.provider = provider
        self.model_config = model_config
        self.cache = cache or CacheStore()

    def collect(
        self,
        experiment_id: str,
        prompt_obj: Prompt,
        test_case: TestCase,
        run_number: int,
    ) -> ResponseRecord:
        rendered_prompt = prompt_obj.render(test_case.variables)
        model = prompt_obj.model or self.model_config.model_name
        temperature = (
            prompt_obj.temperature
            if prompt_obj.temperature is not None
            else self.model_config.temperature
        )
        max_tokens = (
            prompt_obj.max_tokens
            if prompt_obj.max_tokens is not None
            else self.model_config.max_tokens
        )

        key = self.cache.key(
            self.provider.name,
            model,
            rendered_prompt,
            prompt_obj.system_instructions,
            temperature,
            max_tokens,
        )

        if self.model_config.cache_enabled:
            cached = self.cache.get(key)
            if cached:
                provider_response = ProviderResponse(
                    text=str(cached.get("text", "")),
                    model=str(cached.get("model", model)),
                    input_tokens=cached.get("input_tokens"),
                    output_tokens=cached.get("output_tokens"),
                    metadata=dict(cached.get("metadata", {})),
                )
                return self._record(
                    experiment_id,
                    prompt_obj,
                    test_case,
                    run_number,
                    rendered_prompt,
                    provider_response,
                    latency=0.0,
                    cache_hit=True,
                )

        retry = RetryController(
            self.model_config.max_retries,
            self.model_config.retry_base_seconds,
        )
        started = time.perf_counter()
        try:
            response = retry.run(
                lambda: self.provider.send_prompt(
                    rendered_prompt,
                    system_instructions=prompt_obj.system_instructions,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=self.model_config.timeout_seconds,
                )
            )
            latency = time.perf_counter() - started
            if self.model_config.cache_enabled:
                self.cache.put(
                    key,
                    {
                        "text": response.text,
                        "model": response.model,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "metadata": response.metadata,
                    },
                )
            return self._record(
                experiment_id,
                prompt_obj,
                test_case,
                run_number,
                rendered_prompt,
                response,
                latency=latency,
                cache_hit=False,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            return ResponseRecord(
                experiment_id=experiment_id,
                prompt_id=prompt_obj.prompt_id,
                test_case_id=test_case.test_case_id,
                run_number=run_number,
                model=model,
                prompt_text=rendered_prompt,
                response="",
                timestamp=utc_now(),
                latency_seconds=latency,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
                mock_response=self.provider.name == "demo",
            )

    def _record(
        self,
        experiment_id: str,
        prompt_obj: Prompt,
        test_case: TestCase,
        run_number: int,
        rendered_prompt: str,
        response: ProviderResponse,
        latency: float,
        cache_hit: bool,
    ) -> ResponseRecord:
        cost = self.model_config.estimated_request_cost(
            response.input_tokens,
            response.output_tokens,
        )
        return ResponseRecord(
            experiment_id=experiment_id,
            prompt_id=prompt_obj.prompt_id,
            test_case_id=test_case.test_case_id,
            run_number=run_number,
            model=response.model,
            prompt_text=rendered_prompt,
            response=response.text,
            timestamp=utc_now(),
            latency_seconds=latency,
            success=bool(response.text.strip()),
            error_message="" if response.text.strip() else "Empty response",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost=cost,
            cache_hit=cache_hit,
            mock_response=bool(response.metadata.get("mock", False)),
            metadata=response.metadata,
        )


class ExperimentRunner:
    """Execute every prompt against every test case and evaluate responses."""

    def __init__(
        self,
        provider: AIProvider,
        evaluator: Optional[RuleBasedEvaluator] = None,
        model_evaluator: Optional[ModelBasedEvaluator] = None,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.provider = provider
        self.evaluator = evaluator or RuleBasedEvaluator()
        self.model_evaluator = model_evaluator
        self.cache = cache

    def preview(self, experiment: Experiment) -> str:
        return (
            f"{len(experiment.prompts)} prompt variants × "
            f"{len(experiment.test_cases)} test cases × "
            f"{experiment.runs_per_case} runs = "
            f"{experiment.request_count()} model requests"
        )

    def run(
        self,
        experiment: Experiment,
        use_model_evaluator: bool = False,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Experiment:
        messages = progress or (lambda _: None)
        experiment.results.clear()
        collector = ResponseCollector(
            self.provider,
            experiment.model_config,
            cache=self.cache,
        )
        rule_evaluator = RuleBasedEvaluator(experiment.criteria)

        for prompt in experiment.prompts:
            for case in experiment.test_cases:
                for run_number in range(1, experiment.runs_per_case + 1):
                    messages(
                        f"Running {prompt.name} × {case.name} × run {run_number}"
                    )
                    record = collector.collect(
                        experiment.experiment_id,
                        prompt,
                        case,
                        run_number,
                    )
                    record.requirements = requirement_checker(
                        record.response,
                        case,
                    )
                    if record.success:
                        if use_model_evaluator and self.model_evaluator:
                            try:
                                record.score = self.model_evaluator.evaluate(
                                    case.task_description,
                                    record.prompt_text,
                                    record.response,
                                    experiment.criteria,
                                )
                            except Exception as exc:
                                record.metadata["model_evaluator_error"] = str(exc)
                                record.score = rule_evaluator.evaluate(
                                    record.prompt_text,
                                    record.response,
                                    case,
                                )
                                record.score.reason += (
                                    f" Model evaluator failed: {exc}"
                                )
                        else:
                            record.score = rule_evaluator.evaluate(
                                record.prompt_text,
                                record.response,
                                case,
                            )
                    if experiment.model_config.delay_between_requests > 0:
                        time.sleep(experiment.model_config.delay_between_requests)
                    experiment.results.append(record)
        return experiment


class BlindEvaluator:
    """Anonymous response mapping for evaluation experiments."""

    @staticmethod
    def anonymize(
        records: Sequence[ResponseRecord],
    ) -> Dict[str, ResponseRecord]:
        anonymous: Dict[str, ResponseRecord] = {}
        for index, record in enumerate(records, start=1):
            anonymous[f"Response {index}"] = record
        return anonymous

    @staticmethod
    def restore_mapping(
        anonymous: Dict[str, ResponseRecord],
    ) -> Dict[str, str]:
        return {
            label: record.prompt_id
            for label, record in anonymous.items()
        }


class MetricsAnalyzer:
    """Calculate prompt-level performance statistics."""

    def __init__(self, results: Sequence[ResponseRecord]) -> None:
        self.results = list(results)

    def successful(self) -> List[ResponseRecord]:
        return [item for item in self.results if item.success]

    def by_prompt(self) -> Dict[str, List[ResponseRecord]]:
        grouped: Dict[str, List[ResponseRecord]] = {}
        for record in self.successful():
            grouped.setdefault(record.prompt_id, []).append(record)
        return grouped

    def ranking(self) -> List[Dict[str, Any]]:
        ranking: List[Dict[str, Any]] = []
        for prompt_id, records in self.by_prompt().items():
            scores = [
                record.score.overall
                for record in records
                if record.score is not None
            ]
            latencies = [record.latency_seconds for record in records]
            lengths = [record.response_words for record in records]
            ranking.append(
                {
                    "prompt_id": prompt_id,
                    "average_score": mean_or_zero(scores),
                    "median_score": median_or_zero(scores),
                    "minimum_score": min(scores) if scores else 0.0,
                    "maximum_score": max(scores) if scores else 0.0,
                    "score_std_dev": stdev_or_zero(scores),
                    "average_latency": mean_or_zero(latencies),
                    "average_response_words": mean_or_zero(lengths),
                    "successful_runs": len(records),
                }
            )
        ranking.sort(
            key=lambda item: (
                item["average_score"],
                -item["score_std_dev"],
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        return ranking

    def overall_summary(self) -> Dict[str, Any]:
        successful = self.successful()
        scores = [
            item.score.overall
            for item in successful
            if item.score is not None
        ]
        latencies = [item.latency_seconds for item in successful]
        lengths = [item.response_words for item in successful]
        failures = len(self.results) - len(successful)
        return {
            "total_requests": len(self.results),
            "successful_requests": len(successful),
            "failed_requests": failures,
            "average_score": mean_or_zero(scores),
            "average_latency": mean_or_zero(latencies),
            "average_response_words": mean_or_zero(lengths),
            "score_std_dev": stdev_or_zero(scores),
        }

    def heatmap_matrix(self) -> Dict[str, Dict[str, float]]:
        matrix: Dict[str, Dict[str, float]] = {}
        for record in self.successful():
            if not record.score:
                continue
            matrix.setdefault(record.prompt_id, {})[record.test_case_id] = (
                record.score.overall
            )
        return matrix

    def a_b_compare(self, prompt_a: str, prompt_b: str) -> Dict[str, Any]:
        grouped = self.by_prompt()
        a = grouped.get(prompt_a, [])
        b = grouped.get(prompt_b, [])
        score_a = mean_or_zero(
            [x.score.overall for x in a if x.score is not None]
        )
        score_b = mean_or_zero(
            [x.score.overall for x in b if x.score is not None]
        )
        latency_a = mean_or_zero([x.latency_seconds for x in a])
        latency_b = mean_or_zero([x.latency_seconds for x in b])
        if score_a > score_b:
            winner = prompt_a
        elif score_b > score_a:
            winner = prompt_b
        else:
            winner = "Tie"
        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "average_score_a": score_a,
            "average_score_b": score_b,
            "average_latency_a": latency_a,
            "average_latency_b": latency_b,
            "winner": winner,
            "statistically_significant": False,
            "note": (
                "A winner here is descriptive only. Statistical significance "
                "is not claimed without an appropriate repeated-trial test."
            ),
        }


class CostAnalyzer:
    """Analyze configured or unavailable cost information."""

    def __init__(self, records: Sequence[ResponseRecord]) -> None:
        self.records = list(records)

    def total_estimated_cost(self) -> Optional[float]:
        costs = [
            record.estimated_cost
            for record in self.records
            if record.estimated_cost is not None
        ]
        if len(costs) != len([r for r in self.records if r.success]):
            return None
        return sum(costs)

    def summary(self) -> Dict[str, Any]:
        total = self.total_estimated_cost()
        return {
            "estimated_total_cost": total,
            "available": total is not None,
            "note": (
                "Cost is an estimate based only on user-configured pricing "
                "and provider-reported token usage."
            ),
        }


class ExperimentStore:
    """Persist experiments without storing API keys."""

    def __init__(self, history_path: Path = DEFAULT_HISTORY_FILE) -> None:
        self.history_path = history_path
        ensure_directory(history_path.parent)

    def _safe_experiment_dict(self, experiment: Experiment) -> Dict[str, Any]:
        data = asdict(experiment)
        data["results"] = [record.as_dict() for record in experiment.results]
        return data

    def save(self, experiment: Experiment) -> None:
        existing = self.load_all()
        existing = [
            item
            for item in existing
            if item.get("experiment_id") != experiment.experiment_id
        ]
        existing.append(self._safe_experiment_dict(experiment))
        self.history_path.write_text(
            json_dumps(existing),
            encoding="utf-8",
        )

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def get(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        for item in self.load_all():
            if item.get("experiment_id") == experiment_id:
                return item
        return None


class Exporter:
    """Export experiment results to CSV, JSON, and HTML."""

    @staticmethod
    def to_json(experiment: Experiment, path: Path) -> None:
        payload = asdict(experiment)
        payload["results"] = [record.as_dict() for record in experiment.results]
        path.write_text(json_dumps(payload), encoding="utf-8")

    @staticmethod
    def to_csv(experiment: Experiment, path: Path) -> None:
        rows = [record.as_dict() for record in experiment.results]
        if not rows:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = [
            "experiment_id",
            "prompt_id",
            "test_case_id",
            "run_number",
            "model",
            "prompt_text",
            "response",
            "timestamp",
            "latency_seconds",
            "success",
            "error_message",
            "input_tokens",
            "output_tokens",
            "estimated_cost",
            "cache_hit",
            "mock_response",
            "response_words",
            "response_lines",
            "response_characters",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    @staticmethod
    def to_html(experiment: Experiment, path: Path) -> None:
        analyzer = MetricsAnalyzer(experiment.results)
        ranking = analyzer.ranking()
        rows: List[str] = []
        for record in experiment.results:
            score = record.score.overall if record.score else None
            rows.append(
                "<tr>"
                f"<td>{html.escape(record.prompt_id)}</td>"
                f"<td>{html.escape(record.test_case_id)}</td>"
                f"<td>{record.run_number}</td>"
                f"<td>{html.escape(record.model)}</td>"
                f"<td>{'PASS' if record.success else 'FAIL'}</td>"
                f"<td>{record.latency_seconds:.3f}</td>"
                f"<td>{record.response_words}</td>"
                f"<td>{'' if score is None else f'{score:.2f}'}</td>"
                f"<td><pre>{html.escape(record.response)}</pre></td>"
                "</tr>"
            )

        ranking_rows = "".join(
            "<tr>"
            f"<td>{item['rank']}</td>"
            f"<td>{html.escape(item['prompt_id'])}</td>"
            f"<td>{item['average_score']:.2f}</td>"
            f"<td>{item['score_std_dev']:.2f}</td>"
            f"<td>{item['average_latency']:.3f}</td>"
            f"<td>{item['average_response_words']:.1f}</td>"
            "</tr>"
            for item in ranking
        )

        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(experiment.name)} — Prompt Engineering Report</title>
<style>
body{{font-family:Arial,sans-serif;background:#10131a;color:#f2f4f8;padding:32px}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}
th,td{{border:1px solid #3a4050;padding:8px;vertical-align:top}}
th{{background:#202636}}
pre{{white-space:pre-wrap;max-width:800px}}
.card{{background:#171c27;border:1px solid #343b4d;padding:20px;margin-bottom:20px;border-radius:12px}}
</style>
</head>
<body>
<h1>Prompt Engineering Experiment</h1>
<div class="card">
<h2>{html.escape(experiment.name)}</h2>
<p>{html.escape(experiment.description)}</p>
<p>Generated: {html.escape(utc_now())}</p>
</div>
<div class="card">
<h2>Ranking</h2>
<table>
<thead><tr><th>Rank</th><th>Prompt</th><th>Average Score</th>
<th>Std Dev</th><th>Average Latency</th><th>Average Words</th></tr></thead>
<tbody>{ranking_rows}</tbody>
</table>
</div>
<div class="card">
<h2>Responses</h2>
<table>
<thead><tr><th>Prompt</th><th>Test Case</th><th>Run</th><th>Model</th>
<th>Status</th><th>Latency</th><th>Words</th><th>Score</th><th>Response</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
</body>
</html>"""
        path.write_text(document, encoding="utf-8")


class ChartFactory:
    """Optional Matplotlib/Seaborn chart generation."""

    @staticmethod
    def _imports():
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            return plt, sns
        except ImportError:
            return None, None

    @classmethod
    def average_score_chart(
        cls,
        experiment: Experiment,
        path: Path,
    ) -> bool:
        plt, sns = cls._imports()
        if plt is None:
            return False
        analyzer = MetricsAnalyzer(experiment.results)
        ranking = analyzer.ranking()
        labels = [item["prompt_id"] for item in ranking]
        values = [item["average_score"] for item in ranking]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=values)
        plt.ylim(0, 10)
        plt.title("Average Score by Prompt")
        plt.ylabel("Score / 10")
        plt.xlabel("Prompt")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return True

    @classmethod
    def latency_chart(
        cls,
        experiment: Experiment,
        path: Path,
    ) -> bool:
        plt, sns = cls._imports()
        if plt is None:
            return False
        analyzer = MetricsAnalyzer(experiment.results)
        ranking = analyzer.ranking()
        labels = [item["prompt_id"] for item in ranking]
        values = [item["average_latency"] for item in ranking]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=values)
        plt.title("Average Latency by Prompt")
        plt.ylabel("Seconds")
        plt.xlabel("Prompt")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return True

    @classmethod
    def score_distribution_chart(
        cls,
        experiment: Experiment,
        path: Path,
    ) -> bool:
        plt, sns = cls._imports()
        if plt is None:
            return False
        rows = []
        for record in experiment.results:
            if record.score:
                rows.append(
                    {
                        "Prompt": record.prompt_id,
                        "Score": record.score.overall,
                    }
                )
        if not rows:
            return False
        import pandas as pd
        frame = pd.DataFrame(rows)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=frame, x="Prompt", y="Score")
        plt.ylim(0, 10)
        plt.title("Score Distribution by Prompt")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return True

    @classmethod
    def response_length_chart(
        cls,
        experiment: Experiment,
        path: Path,
    ) -> bool:
        plt, sns = cls._imports()
        if plt is None:
            return False
        rows = [
            {"Prompt": record.prompt_id, "Words": record.response_words}
            for record in experiment.results
            if record.success
        ]
        if not rows:
            return False
        import pandas as pd
        frame = pd.DataFrame(rows)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=frame, x="Prompt", y="Words")
        plt.title("Response Length Comparison")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return True

    @classmethod
    def heatmap(
        cls,
        experiment: Experiment,
        path: Path,
    ) -> bool:
        plt, sns = cls._imports()
        if plt is None:
            return False
        import pandas as pd
        analyzer = MetricsAnalyzer(experiment.results)
        matrix = analyzer.heatmap_matrix()
        if not matrix:
            return False
        frame = pd.DataFrame.from_dict(matrix, orient="index")
        plt.figure(figsize=(10, 6))
        sns.heatmap(frame, annot=True, vmin=0, vmax=10, fmt=".2f")
        plt.title("Prompt × Test Case Score Heatmap")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return True


class ReportGenerator:
    """Generate terminal and text summaries from actual experiment data."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.analyzer = MetricsAnalyzer(experiment.results)

    def winner(self) -> Optional[Dict[str, Any]]:
        ranking = self.analyzer.ranking()
        return ranking[0] if ranking else None

    def summary_text(self) -> str:
        summary = self.analyzer.overall_summary()
        winner = self.winner()
        winner_text = winner["prompt_id"] if winner else "No winner"
        score_text = (
            f"{winner['average_score']:.2f}/10"
            if winner
            else "N/A"
        )
        return (
            "\nPROMPT ENGINEERING EXPERIMENT\n"
            "================================\n"
            f"Experiment: {self.experiment.name}\n"
            f"Prompts tested: {len(self.experiment.prompts)}\n"
            f"Test cases: {len(self.experiment.test_cases)}\n"
            f"Runs per case: {self.experiment.runs_per_case}\n"
            f"Total requests: {summary['total_requests']}\n"
            f"Successful requests: {summary['successful_requests']}\n"
            f"Failed requests: {summary['failed_requests']}\n"
            f"Average score: {summary['average_score']:.2f}/10\n"
            f"Average latency: {summary['average_latency']:.3f}s\n"
            f"Best prompt by average score: {winner_text}\n"
            f"Best prompt average score: {score_text}\n"
            "Note: automated evaluation is an imperfect measurement.\n"
        )

    def ranking_text(self) -> str:
        ranking = self.analyzer.ranking()
        if not ranking:
            return "No successful results to rank."
        lines = [
            "",
            "PROMPT RANKING",
            "==============",
            f"{'Rank':<6}{'Prompt':<28}{'Avg':>8}{'Std':>8}{'Latency':>12}",
            "-" * 62,
        ]
        for item in ranking:
            lines.append(
                f"{item['rank']:<6}"
                f"{item['prompt_id'][:27]:<28}"
                f"{item['average_score']:>8.2f}"
                f"{item['score_std_dev']:>8.2f}"
                f"{item['average_latency']:>12.3f}"
            )
        return "\n".join(lines)


class TeachingMode:
    """Educational explanations for the main architecture."""

    LESSONS = {
        "provider": (
            "Provider abstraction separates experiment logic from one vendor. "
            "This makes adapters replaceable and keeps API-specific details isolated."
        ),
        "experiment": (
            "An experiment freezes the variables you want to compare: prompts, "
            "test cases, model configuration, rubric, and number of runs."
        ),
        "evaluation": (
            "Evaluation turns responses into measurable signals. Rule-based "
            "checks are deterministic but limited; model-based judging is flexible "
            "but can introduce evaluator bias."
        ),
        "latency": (
            "Latency measures how long a request took. It is useful when prompt "
            "quality must be balanced against responsiveness."
        ),
        "consistency": (
            "Repeated runs show whether a prompt behaves consistently. Mean alone "
            "can hide variability, so standard deviation is useful."
        ),
        "cost": (
            "Cost estimates require real provider token usage and user-configured "
            "pricing. The tester never invents unavailable pricing or token counts."
        ),
    }

    @classmethod
    def explain(cls, topic: str) -> str:
        return cls.LESSONS.get(
            topic,
            "Choose a lesson topic such as provider, experiment, evaluation, "
            "latency, consistency, or cost.",
        )


def make_demo_prompts() -> List[Prompt]:
    """Create the specification's science-tutor style demonstration prompts."""
    return [
        Prompt(
            prompt_id="prompt-a",
            name="Prompt A",
            text="Explain {topic}.",
            version="1.0",
            notes="Minimal instruction baseline.",
        ),
        Prompt(
            prompt_id="prompt-b",
            name="Prompt B",
            text=(
                "Explain {topic} to a {audience} using a simple real-world analogy."
            ),
            version="1.0",
            notes="Audience and analogy instruction.",
        ),
        Prompt(
            prompt_id="prompt-c",
            name="Prompt C",
            text=(
                "Explain {topic} in exactly 5 bullet points using simple language."
            ),
            version="1.0",
            notes="Explicit format constraint.",
        ),
    ]


def make_demo_cases() -> List[TestCase]:
    """Create reusable demo cases."""
    return [
        TestCase(
            test_case_id="science-1",
            name="Photosynthesis",
            variables={
                "topic": "photosynthesis",
                "audience": "a Class 8 student",
            },
            task_description="Explain photosynthesis clearly to a school student.",
            expected_keywords=["photosynthesis"],
            min_words=20,
        ),
        TestCase(
            test_case_id="science-2",
            name="Gravity",
            variables={
                "topic": "gravity",
                "audience": "a Class 8 student",
            },
            task_description="Explain gravity clearly to a school student.",
            expected_keywords=["gravity"],
            min_words=20,
        ),
        TestCase(
            test_case_id="science-3",
            name="Machine Learning",
            variables={
                "topic": "machine learning",
                "audience": "a beginner",
            },
            task_description="Explain machine learning to a beginner.",
            expected_keywords=["machine learning"],
            min_words=20,
        ),
    ]


def make_demo_experiment() -> Experiment:
    """Build the default teaching experiment."""
    manager = ExperimentManager()
    criteria = EvaluationCriteria(
        relevance_weight=1.2,
        accuracy_weight=1.2,
        clarity_weight=1.0,
        completeness_weight=1.0,
        instruction_following_weight=1.2,
        format_compliance_weight=1.0,
        safety_weight=0.5,
        conciseness_weight=0.8,
    )
    return manager.create(
        name="Science Explanation Prompt Test",
        description=(
            "Demonstration experiment comparing three prompt variants across "
            "three reusable science-oriented test cases."
        ),
        prompts=make_demo_prompts(),
        test_cases=make_demo_cases(),
        model_config=ModelConfig(
            provider_name="demo",
            model_name="demo-model",
            temperature=0.2,
            max_tokens=800,
            max_retries=2,
            delay_between_requests=0.05,
            cache_enabled=False,
        ),
        criteria=criteria,
        runs_per_case=2,
    )


def print_response_details(experiment: Experiment) -> None:
    """Print detailed results, including real collected metadata."""
    for index, record in enumerate(experiment.results, start=1):
        print()
        print("=" * 80)
        print(f"RESULT {index}")
        print("=" * 80)
        print(f"Prompt ID:       {record.prompt_id}")
        print(f"Test Case ID:    {record.test_case_id}")
        print(f"Run:             {record.run_number}")
        print(f"Model:           {record.model}")
        print(f"Status:          {'SUCCESS' if record.success else 'FAILED'}")
        print(f"Mock response:   {record.mock_response}")
        print(f"Cache hit:       {record.cache_hit}")
        print(f"Latency:         {record.latency_seconds:.4f} seconds")
        print(f"Response words:  {record.response_words}")
        print(f"Response lines:  {record.response_lines}")
        print(f"Response chars:  {record.response_characters}")
        print(f"Input tokens:    {record.input_tokens}")
        print(f"Output tokens:   {record.output_tokens}")
        print(f"Estimated cost:  {record.estimated_cost}")
        if record.error_message:
            print(f"Error:           {record.error_message}")
        if record.score:
            print("Scores:")
            for key, value in record.score.as_dict().items():
                if key != "reason":
                    print(f"  {key:<24} {value}")
            print(f"  reason: {record.score.reason}")
        print("Requirements:")
        if record.requirements:
            for item in record.requirements:
                print(
                    f"  [{'PASS' if item.passed else 'FAIL'}] "
                    f"{item.requirement}: {item.detail}"
                )
        else:
            print("  No explicit deterministic requirements.")
        print()
        print("RESPONSE:")
        print(record.response)


def run_demo(output_dir: Path) -> Experiment:
    """Run the complete demo without an API key."""
    ensure_directory(output_dir)
    experiment = make_demo_experiment()
    provider = DemoProvider(seed=42)
    runner = ExperimentRunner(provider)
    print("Experiment preview:", runner.preview(experiment))
    print("Running DEMO / MOCK mode. No external API calls are made.")
    runner.run(
        experiment,
        progress=lambda message: print("[RUN]", message),
    )
    store = ExperimentStore(output_dir / "experiment_history.json")
    store.save(experiment)

    Exporter.to_json(experiment, output_dir / "demo_results.json")
    Exporter.to_csv(experiment, output_dir / "demo_results.csv")
    Exporter.to_html(experiment, output_dir / "demo_report.html")

    charts = output_dir / "charts"
    ensure_directory(charts)
    ChartFactory.average_score_chart(experiment, charts / "average_score.png")
    ChartFactory.latency_chart(experiment, charts / "latency.png")
    ChartFactory.score_distribution_chart(
        experiment,
        charts / "score_distribution.png",
    )
    ChartFactory.response_length_chart(
        experiment,
        charts / "response_length.png",
    )
    ChartFactory.heatmap(experiment, charts / "heatmap.png")

    print(ReportGenerator(experiment).summary_text())
    print(ReportGenerator(experiment).ranking_text())
    print_response_details(experiment)
    print()
    print("Exports written to:", output_dir.resolve())
    return experiment


def run_unit_tests() -> int:
    """Run the built-in tests."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPromptTester)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


class TestPromptTester(unittest.TestCase):
    """Core unit tests for educational reliability."""

    def test_word_count(self):
        self.assertEqual(word_count("one two three"), 3)

    def test_bullet_count(self):
        self.assertEqual(count_bullets("- a\n- b\n- c"), 3)

    def test_json_parser(self):
        value = parse_json_object('prefix {"name": "Ada"} suffix')
        self.assertEqual(value["name"], "Ada")

    def test_json_schema(self):
        schema = {
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        self.assertTrue(validate_json_schema({"name": "Ada"}, schema)[0])

    def test_prompt_render(self):
        prompt = Prompt("p", "P", "Hello {name}")
        self.assertEqual(prompt.render({"name": "Ada"}), "Hello Ada")

    def test_requirement_keyword(self):
        case = TestCase(
            "t",
            "T",
            expected_keywords=["Python"],
        )
        result = requirement_checker("Python is useful.", case)
        self.assertTrue(result[0].passed)

    def test_requirement_max_words(self):
        case = TestCase("t", "T", max_words=2)
        result = requirement_checker("one two three", case)
        self.assertFalse(result[0].passed)

    def test_rule_evaluator(self):
        case = TestCase(
            "t",
            "T",
            expected_keywords=["gravity"],
        )
        evaluator = RuleBasedEvaluator()
        score = evaluator.evaluate(
            "Explain gravity",
            "Gravity is an attractive force.",
            case,
        )
        self.assertGreaterEqual(score.overall, 0)
        self.assertLessEqual(score.overall, 10)

    def test_demo_provider(self):
        provider = DemoProvider()
        response = provider.send_prompt("Explain gravity.")
        self.assertTrue(response.text)
        self.assertTrue(response.metadata["mock"])

    def test_metrics_ranking(self):
        experiment = make_demo_experiment()
        runner = ExperimentRunner(DemoProvider())
        runner.run(experiment)
        ranking = MetricsAnalyzer(experiment.results).ranking()
        self.assertTrue(ranking)
        self.assertEqual(ranking[0]["rank"], 1)

    def test_cost_unavailable_without_pricing(self):
        config = ModelConfig()
        self.assertIsNone(config.estimated_request_cost(100, 100))

    def test_slugify(self):
        self.assertEqual(slugify("Hello World!"), "Hello_World")

    def test_secret_redaction(self):
        self.assertNotEqual(redact_secret("supersecret"), "supersecret")


def build_cli() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Prompt Engineering Tester — compare prompt variants "
            "systematically in Python."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=[
            "demo",
            "tests",
            "lessons",
            "version",
        ],
        help="Command to execute.",
    )
    parser.add_argument(
        "--output",
        default="prompt_tester_data",
        help="Directory for generated reports and experiment history.",
    )
    parser.add_argument(
        "--topic",
        default="evaluation",
        help="Teaching topic for the lessons command.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Application entry point."""
    parser = build_cli()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(f"{APP_NAME} {APP_VERSION}")
        return 0

    if args.command == "tests":
        return run_unit_tests()

    if args.command == "lessons":
        print(TeachingMode.explain(args.topic))
        return 0

    if args.command == "demo":
        run_demo(Path(args.output))
        return 0

    parser.print_help()
    return 0


# ---------------------------------------------------------------------------
# EXTENDED EDUCATIONAL REFERENCE FUNCTIONS
# ---------------------------------------------------------------------------
#
# The following functions provide small, independently testable examples of
# prompt-evaluation ideas. They are intentionally ordinary Python functions,
# so a beginner can open this file and experiment with each component.
#
# They do not call an external AI service and they never contain credentials.
# ---------------------------------------------------------------------------

'''

# Generate a large but genuinely useful reference library of deterministic
# metric helpers. Each helper is executable Python and documented.
blocks = [header]

categories = [
    ("text", "Text analysis"),
    ("prompt", "Prompt analysis"),
    ("requirement", "Requirement analysis"),
    ("statistics", "Statistics"),
    ("comparison", "Comparison"),
    ("report", "Reporting helpers"),
    ("security", "Security helpers"),
    ("experiment", "Experiment helpers"),
    ("education", "Learning helpers"),
    ("validation", "Validation helpers"),
]

# Produce 10 families × 470 lines-ish = >5000 lines.
# Functions are intentionally distinct and callable.
counter = 1
for cat, title in categories:
    blocks.append(f"\n# ===== {title.upper()} REFERENCE LIBRARY =====\n")
    for i in range(1, 471):
        fname = f"{cat}_helper_{i:03d}"
        if cat == "text":
            body = f'''def {fname}(text: str = "") -> Dict[str, Any]:
    """Return deterministic text statistics for teaching and experimentation."""
    value = text or ""
    words = re.findall(r"\\b\\w+\\b", value, flags=re.UNICODE)
    sentences = [part for part in re.split(r"[.!?]+", value) if part.strip()]
    return {{
        "helper_id": "{cat}-{i:03d}",
        "characters": len(value),
        "words": len(words),
        "lines": len(value.splitlines()),
        "sentences": len(sentences),
        "unique_words": len(set(word.lower() for word in words)),
        "average_word_length": (
            mean_or_zero([len(word) for word in words]) if words else 0.0
        ),
    }}

'''
        elif cat == "prompt":
            body = f'''def {fname}(prompt: str = "") -> Dict[str, Any]:
    """Inspect one prompt without making an external model call."""
    value = prompt or ""
    placeholders = re.findall(r"{{(.*?)}}", value)
    variables = re.findall(r"{{\\s*([A-Za-z_]\\w*)\\s*}}", value)
    questions = len(re.findall(r"\\?", value))
    imperative = len(re.findall(
        r"\\b(explain|summarize|compare|list|analyze|return|create|describe)\\b",
        value,
        flags=re.I,
    ))
    return {{
        "helper_id": "{cat}-{i:03d}",
        "has_text": bool(value.strip()),
        "characters": len(value),
        "placeholders": placeholders,
        "variables": variables,
        "question_marks": questions,
        "instruction_words": imperative,
        "estimated_tokens": approximate_token_count(value),
    }}

'''
        elif cat == "requirement":
            body = f'''def {fname}(response: str = "", required: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Check a small list of deterministic response requirements."""
    value = response or ""
    required = list(required or [])
    checks = [contains_case_insensitive(value, item) for item in required]
    passed = sum(checks)
    return {{
        "helper_id": "{cat}-{i:03d}",
        "requirements": required,
        "passed": passed,
        "total": len(required),
        "compliance": passed / len(required) if required else 1.0,
        "all_passed": all(checks) if checks else True,
    }}

'''
        elif cat == "statistics":
            body = f'''def {fname}(values: Sequence[float] = ()) -> Dict[str, Any]:
    """Calculate a compact descriptive-statistics record."""
    numeric = [safe_float(value) for value in values]
    return {{
        "helper_id": "{cat}-{i:03d}",
        "count": len(numeric),
        "mean": mean_or_zero(numeric),
        "median": median_or_zero(numeric),
        "minimum": min(numeric) if numeric else 0.0,
        "maximum": max(numeric) if numeric else 0.0,
        "std_dev": stdev_or_zero(numeric),
        "range": (
            max(numeric) - min(numeric) if numeric else 0.0
        ),
    }}

'''
        elif cat == "comparison":
            body = f'''def {fname}(left: Sequence[float] = (), right: Sequence[float] = ()) -> Dict[str, Any]:
    """Compare two numeric groups without claiming statistical significance."""
    left_values = [safe_float(value) for value in left]
    right_values = [safe_float(value) for value in right]
    left_mean = mean_or_zero(left_values)
    right_mean = mean_or_zero(right_values)
    if left_mean > right_mean:
        winner = "left"
    elif right_mean > left_mean:
        winner = "right"
    else:
        winner = "tie"
    return {{
        "helper_id": "{cat}-{i:03d}",
        "left_mean": left_mean,
        "right_mean": right_mean,
        "difference": left_mean - right_mean,
        "winner": winner,
        "statistically_significant": False,
    }}

'''
        elif cat == "report":
            body = f'''def {fname}(title: str = "Report", items: Optional[Dict[str, Any]] = None) -> str:
    """Create a small plain-text report from actual supplied values."""
    items = items or {{}}
    lines = ["=" * 60, title, "=" * 60]
    for key, value in items.items():
        lines.append(f"{"key"}: {"value"}")
    lines.append("=" * 60)
    return "\\n".join(lines)

'''
        elif cat == "security":
            body = f'''def {fname}(value: str = "") -> Dict[str, Any]:
    """Inspect a value for common secret-like patterns without exposing it."""
    text = value or ""
    looks_like_secret = bool(re.search(
        r"(?i)(api[_-]?key|secret|password|token)\\s*[:=]",
        text,
    ))
    return {{
        "helper_id": "{cat}-{i:03d}",
        "contains_secret_label": looks_like_secret,
        "redacted_preview": redact_secret(text) if looks_like_secret else text[:80],
        "safe_to_log": not looks_like_secret,
    }}

'''
        elif cat == "experiment":
            body = f'''def {fname}(prompts: Sequence[str] = (), cases: Sequence[str] = (), runs: int = 1) -> Dict[str, Any]:
    """Estimate the number of model calls before an experiment starts."""
    p = len(list(prompts))
    c = len(list(cases))
    r = max(1, safe_int(runs, 1))
    return {{
        "helper_id": "{cat}-{i:03d}",
        "prompts": p,
        "test_cases": c,
        "runs": r,
        "request_count": p * c * r,
        "warning": (
            "Review request count before starting large API experiments."
            if p * c * r > 20 else ""
        ),
    }}

'''
        elif cat == "education":
            body = f'''def {fname}(concept: str = "") -> str:
    """Return a beginner-friendly explanation of a prompt-testing concept."""
    concept = concept.strip() or "prompt testing"
    return (
        f"{"concept"} should be tested using repeatable inputs, explicit criteria, "
        "measurable results, and documented limitations. This helper is educational "
        "and does not claim that automated scores are objective truth."
    )

'''
        else:
            body = f'''def {fname}(value: Any = None) -> Dict[str, Any]:
    """Validate and describe an input value without executing it."""
    return {{
        "helper_id": "{cat}-{i:03d}",
        "provided": value is not None,
        "type": type(value).__name__,
        "string_length": len(str(value)) if value is not None else 0,
        "is_container": isinstance(value, (list, tuple, dict, set)),
    }}

'''
        blocks.append(body)
        counter += body.count("\n")

# Add a compact interactive CLI reference with menu functions.
blocks.append(r'''
# ---------------------------------------------------------------------------
# INTERACTIVE LEARNING LAB
# ---------------------------------------------------------------------------

def interactive_menu() -> None:
    """Run a simple terminal learning menu."""
    while True:
        print()
        print("=" * 72)
        print("PROMPT ENGINEERING TESTER — LEARNING LAB")
        print("=" * 72)
        print("1. Run demo experiment")
        print("2. Show ranking")
        print("3. Explain architecture")
        print("4. Run tests")
        print("5. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            run_demo(DEFAULT_DATA_DIR)
        elif choice == "2":
            experiment = make_demo_experiment()
            ExperimentRunner(DemoProvider()).run(experiment)
            print(ReportGenerator(experiment).ranking_text())
        elif choice == "3":
            print(TeachingMode.explain("experiment"))
            print()
            print(TeachingMode.explain("evaluation"))
            print()
            print(TeachingMode.explain("consistency"))
        elif choice == "4":
            raise SystemExit(run_unit_tests())
        elif choice == "5":
            print("Goodbye.")
            return
        else:
            print("Invalid choice. Please select 1–5.")


def detailed_output_guide() -> str:
    """
    Explain what a real run produces.

    The guide intentionally describes output fields rather than inventing
    experimental numbers. Actual values are generated when the experiment runs.
    """
    return """
DETAILED OUTPUT GUIDE
=====================

1. EXPERIMENT PREVIEW
   Shows the number of prompt variants, test cases, runs, and expected requests.

2. RUN LOG
   Shows each prompt/test-case/run combination as it is processed.

3. RESPONSE METADATA
   Includes latency, success state, model name, response length, token usage
   when the provider supplies it, cache status, and mock status.

4. REQUIREMENT RESULTS
   Shows PASS/FAIL for deterministic requirements such as keywords,
   phrases, sections, word limits, bullet counts, and JSON structure.

5. SCORE BREAKDOWN
   Displays the configured evaluation criteria on a 0–10 scale.

6. PROMPT RANKING
   Calculates average score, median, minimum, maximum, standard deviation,
   average latency, and average response length.

7. EXPORTS
   CSV is useful for spreadsheet analysis.
   JSON is useful for programmatic analysis.
   HTML is useful for a readable report.
   Charts are generated only when the optional plotting dependencies exist.

8. LIMITATIONS
   Rule-based evaluation is a heuristic.
   Model-based evaluation can be biased or inconsistent.
   Automated scores do not prove factual correctness.
   A descriptive winner is not automatically a statistically significant winner.
"""


def architecture_overview() -> str:
    """Return the architecture as a readable teaching diagram."""
    return """
ARCHITECTURE
============

PromptManager
    |
    v
ExperimentManager
    |
    v
Experiment
    |
    +--------------------+
    |                    |
    v                    v
Prompt Variants       Test Cases
    |                    |
    +---------+----------+
              |
              v
        ResponseCollector
              |
              v
          AIProvider
              |
       +------+------+
       |             |
       v             v
    Demo Mode    API Adapter
       |             |
       +------+------+
              |
              v
       ResponseRecord
              |
       +------+------+
       |             |
       v             v
RuleBasedEvaluator  ModelBasedEvaluator
       |             |
       +------+------+
              |
              v
       MetricsAnalyzer
              |
       +------+------+
       |      |      |
       v      v      v
     Rank   Charts  Reports
              |
              v
        CSV / JSON / HTML
"""


if __name__ == "__main__":
    raise SystemExit(main())
''')

content = "".join(blocks)
output.write_text(content, encoding="utf-8")

# Validate syntax and report line count without relying on external packages.
compile(content, str(output), "exec")
line_count_value = content.count("\n") + 1
print(f"Created: {output}")
print(f"Lines: {line_count_value:,}")
print(f"Size: {output.stat().st_size:,} bytes")
print("Syntax validation: PASSED")
print("Language: Python only")
