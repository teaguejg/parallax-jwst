class ParallaxError(Exception):
    """Base for errors with no clean builtin mapping."""
    pass

class TargetNotFoundError(ParallaxError):
    """Raised when a target name cannot be resolved to sky coordinates."""
    pass
