"""Target Capture: index long videos once, then search them for plates or people."""
__version__ = "0.7.8"

import os as _os
import warnings as _warnings

# Harmless messages that look like errors during indexing:
# * the video decoder complaining about a damaged frame in DVR/CCTV H.265 files ("Could not find
#   ref with POC"); that frame is skipped and the rest is read normally;
# * a library-version notice inside InsightFace (face alignment).
_os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.")

# Model downloads (car model, plate reader, Arabic reader) go over https. On a newly set up
# Windows PC Python's own certificate list can lack the root a site needs, because Windows only
# fetches root certificates when something asks for them ("CERTIFICATE_VERIFY_FAILED: unable to
# get local issuer certificate"). truststore checks certificates the way Windows itself does
# (and so also accepts an antivirus that inspects https); certifi's list is the fallback.
try:
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:
    try:
        import ssl as _ssl

        import certifi as _certifi

        _ssl._create_default_https_context = lambda: _ssl.create_default_context(cafile=_certifi.where())
    except Exception:
        pass

# Footage is sensitive: keep Ultralytics from sending usage analytics.
try:
    from ultralytics import settings as _settings

    _settings.update({"sync": False})
except Exception:  # ultralytics not installed yet, or read-only settings dir
    pass
