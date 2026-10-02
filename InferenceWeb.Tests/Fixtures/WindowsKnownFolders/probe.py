"""Check the non-creating native lookup used by Chromium in a fresh child."""
import ctypes
import os
from pathlib import Path

folder = ctypes.windll.shell32.SHGetFolderPathW
folder.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p]
folder.restype = ctypes.c_long
for csidl, variable, leaf in [(28, "LOCALAPPDATA", "Local"), (26, "APPDATA", "Roaming")]:
    value = ctypes.create_unicode_buffer(260)
    result = folder(None, csidl, None, 0, value)  # no CSIDL_FLAG_CREATE
    assert result == 0, (variable, hex(result & 0xffffffff), value.value)
    expected = Path(os.environ["USERPROFILE"]) / "AppData" / leaf
    assert os.path.normcase(value.value) == os.path.normcase(str(expected)), (variable, value.value, str(expected))
    assert os.path.normcase(value.value) == os.path.normcase(os.environ[variable])
    assert expected.is_dir()
print("native-known-folders-agree")
