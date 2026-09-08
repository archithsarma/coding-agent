"""Provider-neutral model boundary errors."""


class ModelError(RuntimeError):
    """Base error for expected model boundary failures."""


class ModelProviderError(ModelError):
    """The configured provider rejected or could not complete a request."""


class ModelTimeoutError(ModelError):
    """The provider request exceeded its configured timeout."""


class ModelStructuredOutputError(ModelError):
    """The provider did not return the requested structured output."""
