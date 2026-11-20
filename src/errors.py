"""Custom exceptions so the UI can show friendly, specific error messages."""


class FakeNewsError(Exception):
    """Base class for all application errors."""


class ConfigError(FakeNewsError):
    """Missing or invalid configuration (e.g. API keys)."""


class InputError(FakeNewsError):
    """The user's input is empty, too short or otherwise unusable."""


class RetrievalError(FakeNewsError):
    """Web search or page download failed."""


class LLMError(FakeNewsError):
    """The Groq LLM call failed or returned unusable output."""


class AuthError(FakeNewsError):
    """Authentication or authorization error."""
