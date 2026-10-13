"""Target Capture: index long videos once, then search them for plates or people."""
__version__ = "0.4.1"

import os as _os
import warnings as _warnings

# Harmless messages that look like errors during indexing:
# * the video decoder complaining about a damaged frame in DVR/CCTV H.265 files ("Could not find
#   ref with POC"); that frame is skipped and the rest is read normally;
# * a library-version notice inside InsightFace (face alignment).
_os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.")

# Footage is sensitive: keep Ultralytics from sending usage analytics.
try:
    from ultralytics import settings as _settings

    _settings.update({"sync": False})
except Exception:  # ultralytics not installed yet, or read-only settings dir
    pass
