class AnalysisBase:
    """Base class for all analysis plugins."""
    name = "Unnamed Analysis"
    description = "No description provided."
    default_params = {}  # GUI uses this to build input fields

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def analyze(self, data):
        raise NotImplementedError("Subclasses must implement this method")
