class CustomExceptionPulsarMetric(Exception):
    """
    Custom error handler that is raised in Pulsar Metrics(with value and message fields)
    """

    def __init__(self, value: str, message: str) -> None:
        """constructor of the  CustomExceptionPulsarMetric class

        Parameters
        ----------
        value : str
            The input value of string type
        message : str
            The input message of string type

        Returns
        -------
        None
           return type of None
        """
        self.value = value
        self.message = message
        super().__init__(message)
