"""Target Capture: index long videos once, then search them for plates or people."""
__version__ = "0.2.0"

# Footage is sensitive: keep Ultralytics from sending usage analytics.
try:
    from ultralytics import settings as _settings

    _settings.update({"sync": False})
except Exception:  # ultralytics not installed yet, or read-only settings dir
    pass
