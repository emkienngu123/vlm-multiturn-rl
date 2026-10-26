"""Build-time fetch of the immutable MiniWoB assets recommended by BrowserGym."""

import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request


REVISION = "7fd85d71a4b60325c6585396ec4f48377d049838"
ARCHIVE_SHA256 = "89cc878fc4769d8815630f5367c5c53b0ad4ae0a98b6fa9663eba83cf99a3807"
data = urllib.request.urlopen(
    f"https://codeload.github.com/Farama-Foundation/miniwob-plusplus/tar.gz/{REVISION}", timeout=120
).read()
if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
    raise RuntimeError("MiniWoB source archive checksum mismatch")
destination = Path("/opt/miniwob")
destination.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    root = archive.getnames()[0].split('/')[0]
    archive.extractall(destination, filter="data")
(destination/root).rename(destination/"source")
if not (destination/"source/miniwob/html/miniwob/click-test.html").is_file():
    raise RuntimeError("MiniWoB task assets missing after extraction")
(destination / "revision.txt").write_text(REVISION+"\n")
(destination / "archive.sha256").write_text(ARCHIVE_SHA256+"\n")
