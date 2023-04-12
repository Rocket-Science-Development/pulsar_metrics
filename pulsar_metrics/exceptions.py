class CustomExceptionPulsarMetric(Exception):
    """
    Custom error handler that is raised in Pulsar Metrics(with value and message fields)
    """

    def __init__(self, value: str, message: str) -> None:
        self.value = value
        self.message = message
        super().__init__(message)
