#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""End-to-end chat checks against a RUNNING TensorAgent app, through the same loopback
API its page uses: the Mac app launched by run-mac.sh, or the simulator by run-sim.sh.

Usage:
  chat-e2e.py <app stdout log> [--scenarios fact,follow,newchat,long,think,tool,image,audio]
  chat-e2e.py <app stdout log> --scenarios draw,edit      (an image model, e.g. Qwen-Image 2.1)
  chat-e2e.py <app stdout log> --scenarios film,animate   (MiniMax-H3, the keyframes entry)
  chat-e2e.py <app stdout log> --scenarios reference      (MiniMax-H3 References)
              [--media <dir with image.png and sample.wav>] [--out report.json]
  chat-e2e.py --base http://127.0.0.1:5000/ ...   (a TensorSharp.Server, which needs no token)

The log carries the Debug build's "entry URL" line with the port and the launch token;
every request presents that token as the cookie the page gets. Each scenario asks for
something only a working pipeline can produce: the answer to a question, a follow-up
the KV cache should serve, a new chat the shared-prefix checkpoint should start, a
number only a program the model ran can know, the title printed on an image, and a
word spoken in a recording. Every turn also reports what a user feels: the time to the
first token, the decode rate, and how much of the prompt the cache served.

With an image model loaded the same route makes pictures instead (ImageTurns): `draw`
asks for one from words and `edit` for a change to an attached photo. Each must stream
its denoising steps, end with exactly one picture the app serves as a PNG of the size it
reported, and leave that picture in the saved conversation.

With MiniMax-H3 loaded it makes clips (VideoTurns): `film` from words, `animate` with
the attached photo as the first frame, and `reference` with the photo as the subject of
a new scene. The keyframes and references checkpoints are two catalog entries, and the
same photo is a first frame to one and a reference to the other, so `animate` runs only
with the keyframes entry loaded and `reference` only with the references entry (GET
/api/models says which). Each clip must stream its phases in order through its last
denoising step and end as exactly one MP4 the app serves with Range and HEAD, whose
video track has the frame count, rate and size the turn reported and whose sound,
muxed into the MP4 or a WAV beside it, is 32 kHz stereo as long as the clip; and it
must be in the saved conversation. Both files are read with the standard library. With
cv2 importable the middle frame is decoded as well; without it that check is reported
as skipped, never as passed. chat-e2e-selftest.py runs these checks against files it
makes itself, with no app.

Exits non-zero when any scenario fails, so it can gate a build.
"""
import argparse
import atexit
import array
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave


def entry_from_log(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    found = re.findall(r"entry URL (http://127\.0\.0\.1:\d+/)\?token=([0-9a-f]+)", text)
    if not found:
        sys.exit(f"No 'entry URL' line in {path}; is this a Debug build and has the app started?")
    return found[-1]


class App:
    def __init__(self, base, token):
        self.base = base if base.endswith("/") else base + "/"
        self.cookie = f"tensoragent_token={token}" if token else None

    def request(self, method, path, body=None, content_type="application/json", timeout=60, extra_headers=None):
        data = None
        headers = {"Cookie": self.cookie} if self.cookie else {}
        headers.update(extra_headers or {})
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = content_type
        elif method == "POST":
            data = b""
        req = urllib.request.Request(self.base + path.lstrip("/"), data=data, headers=headers, method=method)
        return urllib.request.urlopen(req, timeout=timeout)

    def json(self, method, path, body=None):
        with self.request(method, path, body) as response:
            return json.loads(response.read().decode())

    def new_session(self):
        return self.json("POST", "api/sessions?conversation=new")["sessionId"]

    def new_chat(self):
        """A new session and the conversation it is filed under."""
        created = self.json("POST", "api/sessions?conversation=new")
        return created["sessionId"], created.get("conversationId")

    def fetch(self, path, timeout=60):
        with self.request("GET", path, timeout=timeout) as response:
            return response.read()

    def raw(self, path, method="GET", headers=None, timeout=60):
        """Status, headers and body, an error status included: a probe of what a file
        route answers has to see the 200 that should have been a 206, not an exception."""
        try:
            with self.request(method, path, timeout=timeout, extra_headers=headers) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as ex:
            return ex.code, ex.headers, ex.read()

    def upload(self, path):
        boundary = "----tensoragent" + uuid.uuid4().hex
        name = os.path.basename(path)
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n").encode()
        payload += open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
        with self.request("POST", "api/upload", payload, f"multipart/form-data; boundary={boundary}") as response:
            return json.loads(response.read().decode())

    def chat(self, session, history, max_tokens, think=False, timeout=900):
        """Send one turn and read the SSE stream the way the page does."""
        body = {"sessionId": session, "messages": history, "maxTokens": max_tokens, "think": think}
        start = time.monotonic()
        first = None
        answer, thinking, tools, error, done = [], [], [], None, {}
        pictures, steps, previews, loras = [], [], 0, []
        clips, progress = [], []
        try:
            with self.request("POST", "api/chat", body, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue
                    try:
                        frame = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if "thinking" in frame and isinstance(frame["thinking"], str):
                        first = first or time.monotonic()
                        thinking.append(frame["thinking"])
                    if "token" in frame and isinstance(frame["token"], str):
                        first = first or time.monotonic()
                        answer.append(frame["token"])
                    if isinstance(frame.get("replace"), str):
                        answer = [frame["replace"]]
                    if "skill_step" in frame or "tool_calls" in frame:
                        tools.append(frame)
                    # An image model's turn: denoising steps (some with a preview), then
                    # the picture.
                    if isinstance(frame.get("image_step"), int):
                        first = first or time.monotonic()
                        steps.append((frame["image_step"], frame.get("image_steps")))
                        previews += 1 if frame.get("preview") else 0
                        # The LoRA plug-ins the host says the picture is drawn with.
                        if isinstance(frame.get("image_loras"), list):
                            loras = frame["image_loras"]
                    if isinstance(frame.get("imageUrl"), str) and frame["imageUrl"]:
                        pictures.append(frame)
                    # A video model's turn: progress through its phases, kept with the time
                    # each frame arrived (the stream is flushed frame by frame), then the clip.
                    if isinstance(frame.get("video_step"), int):
                        now = time.monotonic()
                        first = first or now
                        progress.append((now - start, frame))
                    if isinstance(frame.get("videoUrl"), str) and frame["videoUrl"]:
                        clips.append(frame)
                    if frame.get("error"):
                        error = frame["error"]
                    if frame.get("done") is True:
                        done = frame
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
            error = f"transport: {ex}"
        total = time.monotonic() - start
        ttft = (first - start) if first else None
        return {
            "answer": "".join(answer),
            "thinking": "".join(thinking),
            "tools": tools,
            "error": error,
            "ttft": ttft,
            "total": total,
            "tokens": done.get("tokenCount", 0),
            "tokPerSec": done.get("tokPerSec", 0.0),
            "promptTokens": done.get("promptTokens", 0),
            "reused": done.get("kvReusedTokens", 0),
            "reusePct": done.get("kvReusePercent", 0.0),
            "truncated": done.get("truncated", False),
            "pictures": pictures,
            "steps": steps,
            "previews": previews,
            "loras": loras,
            "aborted": bool(done.get("aborted")),
            "clips": clips,
            "progress": progress,
            "video": video_timing(progress, ttft, total) if progress or clips else None,
        }


def squash(text):
    return re.sub(r"\s+", "", text or "").lower()


def png_size(data):
    """Width and height from a PNG's IHDR, or None when the bytes are not a PNG."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


# A video turn's phases, in the order the host reports them.
VIDEO_PHASES = ("text-encode", "denoise", "vae-decode", "audio-decode", "encode")


def video_timing(progress, first_frame, total):
    """What a video turn felt like: the first progress frame, when each phase began
    (seconds since the request was sent), and the time a denoising step took, measured
    between the first frames of the first and last distinct steps."""
    timeline, phase, arrived, steps = [], None, {}, None
    for at, frame in progress:
        name = frame.get("video_phase")
        if isinstance(name, str) and name != phase:
            phase = name
            timeline.append({"phase": name, "at": round(at, 3), "step": frame.get("video_step"),
                             "elapsed": frame.get("elapsed"), "eta": frame.get("eta")})
        if name == "denoise":
            arrived.setdefault(frame["video_step"], at)
            steps = frame.get("video_steps")
    per_step = None
    if len(arrived) >= 2:
        (k0, t0), (k1, t1) = min(arrived.items()), max(arrived.items())
        per_step = round((t1 - t0) / (k1 - k0), 3)
    return {"firstFrame": first_frame, "total": total, "secondsPerStep": per_step,
            "steps": steps, "timeline": timeline}


def progress_problem(progress):
    """Why a video turn's progress frames break the contract, or "": phases only from
    VIDEO_PHASES and never going back, and denoising that reached its last step."""
    phases = [f["video_phase"] for _, f in progress if isinstance(f.get("video_phase"), str)]
    unknown = sorted(set(phases) - set(VIDEO_PHASES))
    if unknown:
        return f"unknown phase(s) {unknown}; the contract's phases are {list(VIDEO_PHASES)}"
    order = [VIDEO_PHASES.index(p) for p in phases]
    if any(b < a for a, b in zip(order, order[1:])):
        seen = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
        return f"the phases went backwards: {' -> '.join(seen)}"
    denoise = [f for _, f in progress if f.get("video_phase") == "denoise"]
    if not denoise:
        return "no denoising step was streamed"
    last, total = max(f["video_step"] for f in denoise), denoise[-1].get("video_steps")
    if not total or last != total:
        return f"denoising stopped at step {last} of {total}"
    return ""


def mp4_boxes(data, start=0, end=None):
    """(type, body start, body end) of each box from start to end. A box is a 32-bit size
    and a type; size 1 means a 64-bit size follows, size 0 that the box runs to the end."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f"the '{kind.decode('latin-1')}' box at byte {pos} runs past its parent")
        yield kind.decode("latin-1"), pos + header, pos + size
        pos += size


def mp4_box(data, start, end, *path):
    """The body (start, end) of the first box down path, or None."""
    for kind, body, stop in mp4_boxes(data, start, end):
        if kind == path[0]:
            return (body, stop) if len(path) == 1 else mp4_box(data, body, stop, *path[1:])
    return None


# The sampling frequencies an AudioSpecificConfig indexes (ISO/IEC 14496-3, 1.6.3.4).
AAC_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)


def aac_config(data, start, end):
    """(rate, channels) from the AudioSpecificConfig in an esds box (start, end), or None
    when it holds none. It sits behind two descriptors, each a tag and a length of one to
    four 7-bit bytes; channels is 0 when a program config element defines the layout."""
    def descriptor(pos):
        tag, size, pos = data[pos], 0, pos + 1
        for _ in range(4):
            size, pos = size << 7 | data[pos] & 0x7F, pos + 1
            if not data[pos - 1] & 0x80:
                break
        return tag, size, pos

    tag, _, pos = descriptor(start + 4)  # the ES descriptor, past esds' version and flags
    if tag != 3:
        return None
    flags, pos = data[pos + 2], pos + 3  # past the ES id; then the optional fields it flags
    pos += 2 if flags & 0x80 else 0
    pos += 1 + data[pos] if flags & 0x40 else 0
    pos += 2 if flags & 0x20 else 0
    tag, _, pos = descriptor(pos)  # the decoder config descriptor
    if tag != 4:
        return None
    tag, size, pos = descriptor(pos + 13)  # past its object type, stream type, buffer size and bit rates
    if tag != 5:
        return None
    config = data[pos:min(pos + size, end)]
    bits, left = int.from_bytes(config, "big"), 8 * len(config)

    def take(n):
        nonlocal left
        left -= n
        if left < 0:
            raise ValueError("the AudioSpecificConfig is shorter than its fields")
        return bits >> left & (1 << n) - 1

    if take(5) == 31:  # an escaped audio object type
        take(6)
    index = take(4)
    if index != 15 and index >= len(AAC_RATES):
        raise ValueError(f"the AudioSpecificConfig has the reserved frequency index {index}")
    rate = take(24) if index == 15 else AAC_RATES[index]
    return rate, take(4)


def mp4_tracks(data):
    """What a player finds in an MP4, read from its box tree with struct alone: per track
    the handler ('vide', 'soun'), the sample entry, the sample count from stsz and from
    stts, the size from tkhd, the frame rate, the channels and rate of a sound, and the
    duration. Raises ValueError (or struct.error, or IndexError on a truncated box) when
    the tree is not an MP4's."""
    moov = mp4_box(data, 0, len(data), "moov")
    mvhd = moov and mp4_box(data, *moov, "mvhd")
    if not mvhd:
        raise ValueError("no 'moov' box with an 'mvhd'")
    # mvhd, mdhd and tkhd start with a version byte; version 1 widens the times to 64 bits.
    movie_scale = struct.unpack_from(">I", data, mvhd[0] + (20 if data[mvhd[0]] == 1 else 12))[0]
    tracks = []
    for kind, start, end in mp4_boxes(data, *moov):
        if kind != "trak":
            continue
        tkhd = mp4_box(data, start, end, "tkhd")
        mdhd = mp4_box(data, start, end, "mdia", "mdhd")
        hdlr = mp4_box(data, start, end, "mdia", "hdlr")
        stbl = mp4_box(data, start, end, "mdia", "minf", "stbl")
        stsd, stts = stbl and mp4_box(data, *stbl, "stsd"), stbl and mp4_box(data, *stbl, "stts")
        if not (tkhd and mdhd and hdlr and stsd and stts):
            raise ValueError("a track without tkhd, mdhd, hdlr, stsd or stts")
        wide = data[mdhd[0]] == 1
        scale, duration = struct.unpack_from(">IQ" if wide else ">II", data, mdhd[0] + (20 if wide else 12))
        # Width and height are tkhd's last eight bytes, 16.16 fixed point, in either version.
        width, height = struct.unpack_from(">II", data, tkhd[1] - 8)
        codec = data[stsd[0] + 12:stsd[0] + 16].decode("latin-1")
        runs = [struct.unpack_from(">II", data, stts[0] + 8 + 8 * i)
                for i in range(struct.unpack_from(">I", data, stts[0] + 4)[0])]
        stsz = mp4_box(data, *stbl, "stsz")
        track = {
            "handler": data[hdlr[0] + 8:hdlr[0] + 12].decode("latin-1"),
            "codec": codec,
            "width": width / 65536,
            "height": height / 65536,
            "timescale": scale,
            "samples": struct.unpack_from(">I", data, stsz[0] + 8)[0] if stsz else None,
            "timedSamples": sum(n for n, _ in runs),
            # The usual sample duration, from the longest run of equal ones, so a last frame
            # a muxer gave a different duration does not move the rate.
            "fps": round(scale / max(runs)[1], 3) if runs and max(runs)[1] else None,
            "mediaSeconds": round(duration / scale, 4) if scale else None,
            "seconds": round(duration / scale, 4) if scale else None,
        }
        # What plays is the edit list, in the movie's timescale: an AAC track's media also
        # holds the encoder's priming (afinfo counts 2112 samples in afconvert's AAC), which
        # its edit list trims. Without one the media duration is what plays.
        elst = mp4_box(data, start, end, "edts", "elst")
        if elst and movie_scale:
            wide = data[elst[0]] == 1
            spans = [struct.unpack_from(">Q" if wide else ">I", data, elst[0] + 8 + (20 if wide else 12) * i)[0]
                     for i in range(struct.unpack_from(">I", data, elst[0] + 4)[0])]
            track["seconds"] = round(sum(spans) / movie_scale, 4)
        entry = stsd[0] + 16  # the first sample entry's body, past its size and type
        if track["handler"] == "soun":
            version, = struct.unpack_from(">H", data, entry + 8)
            channels, = struct.unpack_from(">H", data, entry + 16)
            rate = struct.unpack_from(">I", data, entry + 24)[0] / 65536
            if version == 2:  # QuickTime's v2 sound description moves both past the legacy fields
                rate, channels = struct.unpack_from(">dI", data, entry + 32)
            # For AAC those fields are only a template: AVAssetWriter, configured as the app
            # configures it, writes channelcount 2 for a mono track that afinfo reads as 1 ch.
            # A decoder goes by the AudioSpecificConfig in the entry's esds box, so this does.
            esds = codec == "mp4a" and mp4_box(data, entry + (28, 44, 64)[min(version, 2)],
                                               stsd[0] + 8 + struct.unpack_from(">I", data, stsd[0] + 8)[0], "esds")
            config = esds and aac_config(data, *esds)
            if config:
                rate, channels = config[0], config[1] or channels
            track.update(channels=channels, rate=rate)
        tracks.append(track)
    return tracks


def clip_problem(clip, tracks, video=None):
    """Why the MP4's tracks are not the clip the turn reported, or "". `video` is the
    loaded model's capability from GET /api/models."""
    frames, fps = clip["frames"], clip["fps"]
    pictures = [t for t in tracks if t["handler"] == "vide"]
    sounds = [t for t in tracks if t["handler"] == "soun"]
    if len(pictures) != 1:
        return f"expected one video track, the MP4 has {len(pictures)}"
    v = pictures[0]
    if v["samples"] != frames or v["timedSamples"] != frames:
        return (f"the video track has {v['samples']} samples ({v['timedSamples']} in its timing table), "
                f"the turn reported {frames} frames")
    if v["fps"] is None or abs(v["fps"] - fps) > 0.01:
        return f"the video track runs at {v['fps']} fps, the turn reported {fps}"
    if (round(v["width"]), round(v["height"])) != (clip["width"], clip["height"]):
        return f"the video track is {v['width']:g}x{v['height']:g}, the turn reported {clip['width']}x{clip['height']}"
    has_audio, separate = clip.get("hasAudio") is True, bool(clip.get("audioUrl"))
    if video and video.get("supportsAudio") and not has_audio:
        return "the loaded model makes sound (supportsAudio) but the clip has none"
    if separate and not has_audio:
        return "the turn gave an audioUrl for a clip it says has no sound"
    if sounds and not has_audio:
        return f"hasAudio is false but the MP4 has {len(sounds)} sound track(s)"
    if sounds and separate:
        return "the sound is muxed into the MP4 and is a separate WAV too; the page would play it twice"
    if has_audio and not separate:
        if len(sounds) != 1:
            return f"hasAudio without an audioUrl means the sound is muxed, but the MP4 has {len(sounds)} sound tracks"
        s = sounds[0]
        if (s["codec"], s["channels"], s["rate"]) != ("mp4a", 2, 32000):
            return f"the sound track is {s['codec']}, {s['channels']} channels at {s['rate']:g} Hz; expected mp4a stereo at 32000 Hz"
        if abs(s["seconds"] - frames / fps) > 0.15:
            return f"the sound track lasts {s['seconds']:.3f} s, the clip {frames / fps:.3f} s"
    return ""


def wav_facts(data):
    """Channels, rate, bits, length and RMS (full scale = 1) of a PCM WAV, by the wave
    module, which raises wave.Error on anything else: Python 3.9's reads only format 1,
    the one TensorSharp's WavWriter writes, and not afconvert's WAVE_FORMAT_EXTENSIBLE."""
    with wave.open(io.BytesIO(data)) as w:
        channels, rate, width, count = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
        pcm = w.readframes(count)
    rms = 0.0
    if width == 2 and len(pcm) >= 2:
        samples = array.array("h", pcm[:len(pcm) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768
    return {"channels": channels, "rate": rate, "bits": 8 * width,
            "seconds": round(count / rate, 4) if rate else 0.0, "rms": round(rms, 6)}


def wav_problem(facts, seconds):
    """Why a separate soundtrack is not the clip's, or ""."""
    if (facts["channels"], facts["rate"], facts["bits"]) != (2, 32000, 16):
        return (f"the WAV is {facts['channels']} channels at {facts['rate']} Hz, {facts['bits']}-bit; "
                "expected 16-bit stereo at 32000 Hz")
    if abs(facts["seconds"] - seconds) > 0.15:
        return f"the WAV lasts {facts['seconds']:.3f} s, the clip {seconds:.3f} s"
    if facts["rms"] <= 1e-4:
        return f"the WAV is silent (RMS {facts['rms']})"
    return ""


def frame_facts(data, frames):
    """The middle frame as cv2 decodes it: its size and how much its pixels vary, which
    is what tells a black or flat frame (a VAE or encoder failure the container cannot
    show) from a picture. None when cv2 is not importable."""
    try:
        import cv2
    except ImportError:
        return None
    middle, frame = frames // 2, None
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "clip.mp4")
        with open(path, "wb") as f:
            f.write(data)
        capture = cv2.VideoCapture(path)
        try:
            # Read up to the middle rather than seek: whether a seek in an H.264 stream with
            # reordered frames is frame-exact depends on the backend, and a clip this short
            # costs nothing to decode.
            for index in range(middle + 1):
                ok, frame = capture.read()
                if not ok:
                    return {"frame": index, "decoded": False}
        finally:
            capture.release()
    return {"frame": middle, "decoded": True, "width": int(frame.shape[1]), "height": int(frame.shape[0]),
            "std": round(float(frame.std()), 2)}


def range_problem(app, url, whole):
    """Why the file route would not feed a <video> element, or "". WebKit starts a media
    load with 'Range: bytes=0-1' and will not play from a server that answers it with the
    whole file; it then seeks with ranges of its own. HEAD is how a client learns the
    length without the body."""
    size = len(whole)
    status, headers, body = app.raw(url, headers={"Range": "bytes=0-1"})
    if status != 206 or headers.get("Content-Range") != f"bytes 0-1/{size}" or body != whole[:2]:
        return (f"Range bytes=0-1 answered {status}, Content-Range {headers.get('Content-Range')!r}, "
                f"{len(body)} bytes; expected 206, 'bytes 0-1/{size}' and the first 2 bytes")
    start = size // 2
    end = min(size - 1, start + 4095)
    status, headers, body = app.raw(url, headers={"Range": f"bytes={start}-{end}"})
    if status != 206 or headers.get("Content-Range") != f"bytes {start}-{end}/{size}" or body != whole[start:end + 1]:
        return (f"Range bytes={start}-{end} answered {status}, Content-Range {headers.get('Content-Range')!r}, "
                f"{len(body)} bytes, which is not that slice of the file")
    status, headers, body = app.raw(url, method="HEAD")
    if status != 200 or headers.get("Content-Length") != str(size) or body:
        return f"HEAD answered {status} with Content-Length {headers.get('Content-Length')!r}; expected 200 and {size}"
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="the app's stdout log with the Debug 'entry URL' line")
    parser.add_argument("--base", help="talk to this base URL instead, without a token (a TensorSharp.Server)")
    parser.add_argument("--scenarios", default="fact,follow,newchat,long,think,tool,image,audio")
    parser.add_argument("--media", default=os.environ.get("TS_TEST_MEDIA_DIR", os.path.expanduser("~/work/models/testmedia")))
    parser.add_argument("--out")
    parser.add_argument("--draw-prompt", default="A red apple on a white table, soft daylight, studio photograph",
                        help="what the draw scenario asks for")
    parser.add_argument("--edit-prompt", default="Make the background a deep blue night sky with stars, keep the text unchanged",
                        help="what the edit scenario asks to change about --edit-photo")
    parser.add_argument("--edit-photo", help="the photo the edit scenario attaches (default: image.png in --media)")
    parser.add_argument("--loras", default="",
                        help="LoRA plug-ins to turn on for draw/edit, as id or id:strength, comma-separated "
                             "(GET /api/agent/loras lists them); the previous choice is restored afterwards")
    args = parser.parse_args(argv)

    if args.base:
        base, token = args.base, None
    elif args.log:
        base, token = entry_from_log(args.log)
    else:
        parser.error("give the app's log, or --base")
    app = App(base, token)
    wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    # A misspelt scenario used to run nothing and report "All 0 chat scenarios passed".
    known = {"fact", "follow", "newchat", "long", "think", "tool", "image", "audio",
             "draw", "edit", "film", "animate", "reference"}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        parser.error(f"unknown scenario(s) {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    try:
        engine = app.json("GET", "api/agent/engine")
        model = engine.get("model") or {}
        # A launch that was just asked to load a model is still loading it, and a turn sent
        # now is refused; wait for the load (a few seconds, or longer for a big model's
        # first read off disk) rather than report that refusal as a failed scenario.
        deadline = time.monotonic() + 300
        while model.get("state") == "Loading" and time.monotonic() < deadline:
            time.sleep(1)
            engine = app.json("GET", "api/agent/engine")
            model = engine.get("model") or {}
        print(f"==> {app.base} · {engine.get('engine', '?')}")
        print(f"    model {model.get('id')} {model.get('state')} (prefix cache warm: {model.get('prefixCacheWarm')})")
    except (urllib.error.HTTPError, ValueError):
        engine = {}
        print(f"==> {app.base} (no TensorAgent engine route: a TensorSharp.Server)")

    rows, failures = [], []

    # LoRA plug-ins for the picture scenarios, chosen through the app's own route and put back
    # as they were when the run ends. What each picture must then show: the plug-ins it is drawn
    # with (an edit-only one is not applied to a picture made from words) and, with a speed
    # plug-in, its step count instead of the model's 40.
    lora_plan, previous_loras = None, None
    if args.loras:
        offered = {l["id"]: l for l in app.json("GET", "api/agent/loras").get("loras", [])}
        wanted_loras = []
        for item in (x.strip() for x in args.loras.split(",") if x.strip()):
            lora_id, _, strength = item.partition(":")
            lora = offered.get(lora_id)
            if lora is None:
                sys.exit(f"--loras: no plug-in '{lora_id}' (offered: {', '.join(sorted(offered))})")
            if lora.get("state") != "Installed":
                sys.exit(f"--loras: {lora_id} is {lora.get('state')}; download it first")
            wanted_loras.append({"id": lora_id, "strength": float(strength) if strength else lora["defaultStrength"]})
        previous_loras = app.json("GET", "api/agent/loras").get("chosen", [])
        # The run's choice drops whatever is on now, and the host takes back only plug-ins
        # whose files are there: one that is on with its files gone could not be restored.
        gone = [offered[c["id"]]["name"] for c in previous_loras
                if c["id"] in offered and offered[c["id"]].get("state") != "Installed"]
        if gone:
            sys.exit(f"--loras: {', '.join(gone)} is turned on but not downloaded, so the current choice "
                     "could not be put back after the run; download it again or turn it off first")

        def restore_loras():
            """Put the choice from before the run back; False, having said why, when the host refused."""
            try:
                app.json("POST", "api/agent/loras/choice", {"loras": previous_loras})
                return True
            except urllib.error.HTTPError as e:
                reason = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            except (OSError, ValueError) as e:
                reason = str(e)
            print(f"WARNING: the LoRA choice from before the run could not be put back ({reason}); "
                  "set it again in the app's LoRA sheet")
            return False

        # Put back however the run ends: a failed scenario, an exception or Ctrl-C included.
        atexit.register(restore_loras)
        saved = app.json("POST", "api/agent/loras/choice", {"loras": wanted_loras})
        chosen = [offered[c["id"]] for c in saved.get("chosen", []) if c["id"] in offered]
        speeds = [l for l in chosen if l["kind"] == "Speed"]

        def applied(editing):
            used = [l for l in chosen if editing or not l.get("needsPhoto")]
            # A plug-in that works only at the model's own steps keeps them: the speed one sits out.
            if any(l.get("needsModelSteps") for l in used):
                used = [l for l in used if l["kind"] != "Speed"]
            return [l for l in used if l["kind"] == "Speed"] + [l for l in used if l["kind"] != "Speed"]

        def expected(editing):
            return [l["name"] for l in applied(editing)]

        def speed_steps(editing):
            return next((l["steps"] for l in applied(editing) if l["kind"] == "Speed"), None)
        lora_plan = {"expected": expected, "steps": speed_steps}
        described = ", ".join("{}@{}".format(c["id"], c["strength"]) for c in saved.get("chosen", []))
        print(f"    LoRA plug-ins: {described}")

    def turn(name, session, history, prompt, check, max_tokens=256, think=False, extra=None):
        message = {"role": "user", "content": prompt}
        message.update(extra or {})
        history.append(message)
        result = app.chat(session, history, max_tokens, think=think)
        history.append({"role": "assistant", "content": result["answer"]})
        ok, why = check(result)
        if result["error"]:
            ok, why = False, f"error: {result['error']}"
        decode = result["tokPerSec"]
        print(f"--- {name}: {'ok ' if ok else 'FAIL'} ttft {result['ttft'] or 0:.2f}s, "
              f"{result['tokens']} tokens at {decode:.1f} tok/s, total {result['total']:.1f}s, "
              f"prompt {result['promptTokens']} (reused {result['reused']}, {result['reusePct']:.1f}%)"
              + (f", {len(result['tools'])} tool events" if result["tools"] else ""))
        print(f"    answer: {result['answer'][:160]!r}")
        if not ok:
            print(f"    why: {why}")
            failures.append(name)
        # A video turn's raw progress frames stay out: its "video" summary carries them.
        rows.append({"scenario": name, "ok": ok, "why": why,
                     **{k: v for k, v in result.items() if k not in ("tools", "steps", "progress")},
                     "toolEvents": len(result["tools"]), "imageSteps": len(result["steps"])})
        return result

    def contains(word):
        return lambda r: (word.lower() in r["answer"].lower(), f"expected '{word}' in the answer")

    history = []
    session = None
    if "fact" in wanted or "follow" in wanted:
        session = app.new_session()
        turn("fact", session, history, "What is the capital of France? Reply with one word.", contains("Paris"))
    if "follow" in wanted:
        def reuses(r):
            if "rome" not in r["answer"].lower():
                return False, "expected 'Rome' in the answer"
            if r["reusePct"] < 50:
                return False, f"a follow-up should reuse most of its prompt, reused {r['reusePct']:.1f}%"
            return True, ""
        turn("follow", session, history, "And the capital of Italy? One word.", reuses)
    if "newchat" in wanted:
        def warm_start(r):
            if "jupiter" not in r["answer"].lower():
                return False, "expected 'Jupiter' in the answer"
            if r["reused"] <= 0:
                return False, "a new chat should start from the shared-prefix checkpoint (0 tokens reused)"
            return True, ""
        turn("newchat", app.new_session(), [], "What is the largest planet in the solar system? One word.", warm_start)
    if "long" in wanted:
        # The decode rate needs an answer long enough for the first token not to dominate.
        # The story itself is the check, with room to reason first: a token count alone
        # passed Muse-Glimmer at 400 tokens, every one of them reasoning and the answer empty.
        def story(r):
            words = len(r["answer"].split())
            return (words >= 150, f"expected a story of about 250 words, got {words} words")
        turn("long", app.new_session(), [],
             "Write a 250-word story about a lighthouse keeper who finds a message in a bottle.",
             story, max_tokens=2048)
    if "think" in wanted:
        def reasoned(r):
            if not r["thinking"].strip():
                return False, "a thinking turn produced no reasoning"
            return ("no" in r["answer"].lower(), "expected 'no' (91 = 7 x 13)")
        turn("think", app.new_session(), [], "Is 91 a prime number? Answer yes or no.", reasoned, max_tokens=4096, think=True)
    if "tool" in wanted:
        digest = hashlib.sha256(b"tensoragent").hexdigest()[:12]

        def ran_code(r):
            if not r["tools"]:
                return False, "no tool was run"
            return (digest in r["answer"].lower(), f"expected the digest prefix {digest}")
        turn("tool", app.new_session(), [],
             "Run a shell command that prints the SHA-256 hex digest of the exact ASCII string "
             "tensoragent (no trailing newline), then reply with only the first 12 hex characters.",
             ran_code, max_tokens=1024)
    def attach(path):
        """Upload a file and describe it in a message the way the page's messageFor does."""
        uploaded = app.upload(path)
        stored, kind = uploaded["file"], uploaded.get("mediaType")
        extra = {"attachments": [{"file": stored, "fileName": uploaded.get("fileName"), "mediaType": kind}]}
        if kind == "image":
            extra.update({"imagePaths": [stored], "stillImagePaths": [stored]})
        elif kind == "audio":
            extra["audioPaths"] = [stored]
        return extra

    if "image" in wanted:
        turn("image", app.new_session(), [], "What is the title written on this banner? Reply with just the title.",
             lambda r: ("tensorsharp" in squash(r["answer"]), "expected the banner's title, TensorSharp"),
             extra=attach(os.path.join(args.media, "image.png")))
    if "audio" in wanted:
        # The engine suite's own wording (EngineParallelInferenceTests): "this recording"
        # alone gets E2B answering that it has no transcription tool.
        turn("audio", app.new_session(), [], "Transcribe the first sentence of this audio clip.",
             lambda r: ("fox" in r["answer"].lower(), "expected the pangram's 'fox'"),
             extra=attach(os.path.join(args.media, "sample.wav")))

    def drew(editing=False, input_size=None):
        """One picture, served as a PNG of the size the turn reported, after its steps.
        `editing` says a photo is attached, which is what decides the plug-ins; `input_size`
        is the photo's size when it could be read (a PNG), to hold the edit to its shape."""
        def check(r):
            if r["aborted"]:
                return False, "the turn ended without a picture (stopped)"
            if len(r["pictures"]) != 1:
                return False, f"expected exactly one picture, got {len(r['pictures'])}"
            if not r["steps"]:
                return False, "no denoising step was streamed"
            last, total = r["steps"][-1]
            if total and last != total:
                return False, f"the steps stopped at {last} of {total}"
            if lora_plan:
                want = lora_plan["expected"](editing)
                if r["loras"] != want:
                    return False, f"the picture was drawn with {r['loras'] or 'no plug-ins'}, expected {want or 'none'}"
                steps = lora_plan["steps"](editing)
                if steps and total != steps:
                    return False, f"{total} steps, but the speed plug-in runs {steps}"
            picture = r["pictures"][0]
            size = png_size(app.fetch(picture["imageUrl"]))
            if size is None:
                return False, f"{picture['imageUrl']} is not a PNG"
            if size != (picture.get("width"), picture.get("height")):
                return False, f"the PNG is {size[0]}x{size[1]} but the turn reported {picture.get('width')}x{picture.get('height')}"
            if input_size:
                # An edit keeps the photo's shape (to the 32-pixel grid the model works on).
                want, got = input_size[0] / input_size[1], size[0] / size[1]
                if abs(want - got) > 0.05:
                    return False, f"the edit changed the photo's shape: {input_size} -> {size}"
            elif editing:
                # Only a PNG's size is read here, and a check that did not run is not a pass.
                r.setdefault("files", {}).setdefault("skipped", []).append(
                    "the edit's shape is checked only for a PNG photo")
            return True, ""
        return check

    def saved(conversation, name):
        """The picture is in the saved conversation, which is what a reopened chat shows."""
        if not conversation:
            return
        stored = app.json("GET", f"api/agent/conversations/{conversation}")
        messages = stored.get("messages") or []
        made = [m.get("imageUrl") for m in messages if m.get("role") == "assistant" and m.get("imageUrl")]
        row = rows[-1]
        if not made and row["ok"]:
            row["ok"], row["why"] = False, "the saved conversation has no picture"
            failures.append(name)
            print(f"    why: {row['why']}")
        elif made:
            print(f"    saved: {made[-1]}")

    def image_turn(name, prompt, extra=None, editing=False, input_size=None):
        session, conversation = app.new_chat()
        result = turn(name, session, [], prompt, drew(editing, input_size), extra=extra)
        if result["pictures"]:
            p = result["pictures"][0]
            print(f"    picture: {p['imageUrl']} {p.get('width')}x{p.get('height')}, "
                  f"{len(result['steps'])} steps ({result['previews']} with a preview)"
                  + (f", LoRA {' + '.join(result['loras'])}" if result["loras"] else ""))
        saved(conversation, name)

    if "draw" in wanted:
        image_turn("draw", args.draw_prompt)
    if "edit" in wanted:
        photo = args.edit_photo or os.path.join(args.media, "image.png")
        image_turn("edit", args.edit_prompt, extra=attach(photo), editing=True,
                   input_size=png_size(open(photo, "rb").read()))

    def filmed(video, input_size=None):
        """One clip after its phases and steps, served as an MP4 (and a WAV, when the sound
        is a separate file) with the facts the turn reported. What was measured is left in
        the result as "files", so the report carries it beside the verdict."""
        def check(r):
            files = r["files"] = {}
            if r["aborted"]:
                return False, "the turn ended without a clip (stopped)"
            if len(r["clips"]) != 1:
                return False, f"expected exactly one videoUrl, got {len(r['clips'])}"
            why = progress_problem(r["progress"])
            if why:
                return False, why
            clip = r["clips"][0]
            frames, fps, url = clip.get("frames"), clip.get("fps"), clip["videoUrl"]
            numbers = (frames, fps, clip.get("width"), clip.get("height"))
            if type(frames) is not int or not all(type(n) in (int, float) and n > 0 for n in numbers):
                return False, f"the clip reports frames, fps, width and height of {numbers}"
            status, headers, data = app.raw(url)
            kind = headers.get("Content-Type") or ""
            files["mp4"] = {"status": status, "contentType": kind, "bytes": len(data)}
            if status != 200 or not kind.startswith("video/mp4") or data[4:8] != b"ftyp":
                return False, f"GET {url} answered {status} {kind or 'with no Content-Type'}, not an MP4 with an ftyp box"
            try:
                files["mp4"]["tracks"] = mp4_tracks(data)
            except (ValueError, struct.error, IndexError) as ex:
                return False, f"{url} does not parse as an MP4: {ex}"
            why = clip_problem(clip, files["mp4"]["tracks"], video)
            if why:
                return False, why
            if clip.get("audioUrl"):
                status, headers, sound = app.raw(clip["audioUrl"])
                kind = headers.get("Content-Type") or ""
                if status != 200 or not kind.startswith("audio/wav"):
                    return False, f"GET {clip['audioUrl']} answered {status} {kind or 'with no Content-Type'}, not a WAV"
                try:
                    files["wav"] = wav_facts(sound)
                except (wave.Error, EOFError, struct.error) as ex:
                    return False, f"{clip['audioUrl']} is not a PCM WAV: {ex}"
                why = wav_problem(files["wav"], frames / fps)
                if why:
                    return False, why
            frame = files["frame"] = frame_facts(data, frames)
            if frame is None:
                files["skipped"] = ["middle-frame decode: cv2 is not importable"]
            elif not frame["decoded"]:
                return False, f"cv2 could not decode frame {frame['frame']} of {url}"
            elif (frame["width"], frame["height"]) != (clip["width"], clip["height"]):
                return False, (f"the decoded frame is {frame['width']}x{frame['height']}, "
                               f"the turn reported {clip['width']}x{clip['height']}")
            elif frame["std"] <= 5:
                return False, f"the middle frame is blank (pixel std {frame['std']})"
            why = range_problem(app, url, data)
            if why:
                return False, why
            if input_size:
                # The host picks a canvas at the photo's shape, on the model's size grid.
                want, got = input_size[0] / input_size[1], clip["width"] / clip["height"]
                if abs(want - got) > 0.08:
                    return False, f"the clip is {clip['width']}x{clip['height']}, not the photo's shape {input_size[0]}x{input_size[1]}"
            return True, ""
        return check

    def saved_clip(conversation, name, result):
        """The clip is in the saved conversation, with its WAV exactly when the stream gave one."""
        if not conversation or not result["clips"]:
            return
        stored = app.json("GET", f"api/agent/conversations/{conversation}")
        made = [m for m in stored.get("messages") or [] if m.get("role") == "assistant" and m.get("videoUrl")]
        clip, row, why = result["clips"][0], rows[-1], ""
        if not made:
            why = "the saved conversation has no clip"
        elif made[-1]["videoUrl"] != clip["videoUrl"]:
            why = f"the saved clip is {made[-1]['videoUrl']}, the turn made {clip['videoUrl']}"
        elif (made[-1].get("audioUrl") or None) != (clip.get("audioUrl") or None):
            why = f"the saved audioUrl is {made[-1].get('audioUrl')!r}, the turn gave {clip.get('audioUrl')!r}"
        if why and row["ok"]:
            row["ok"], row["why"] = False, why
            failures.append(name)
            print(f"    why: {why}")
        elif made:
            print(f"    saved: {made[-1]['videoUrl']}" + (f" + {made[-1]['audioUrl']}" if made[-1].get("audioUrl") else ""))

    # The two MiniMax-H3 entries read the same photo differently, a first frame on the
    # keyframes checkpoint and a reference on the references one, so a scenario run on the
    # wrong one would pass while testing the other. GET /api/models says which is loaded.
    def video_model(v):
        return "" if v else "no video model is loaded (GET /api/models has no video capability)"

    def keyframes_model(v):
        if v and v.get("supportsImageConditioning") and not v.get("supportsReferenceConditioning"):
            return ""
        return video_model(v) or ("needs the MiniMax-H3 keyframes entry (minimax-h3-fl2va-q4k) loaded: "
                                  "the loaded video model does not take a photo as the first frame")

    def references_model(v):
        if v and v.get("supportsReferenceConditioning"):
            return ""
        return video_model(v) or ("needs the MiniMax-H3 References entry (minimax-h3-ref2va-q4k) loaded: "
                                  "the loaded video model takes no references")

    def video_turn(name, prompt, needs, photo=None, keep_shape=False):
        video = app.json("GET", "api/models").get("video")
        why = needs(video)
        if why:
            print(f"--- {name}: FAIL (not run)")
            print(f"    why: {why}")
            rows.append({"scenario": name, "ok": False, "why": why})
            failures.append(name)
            return
        session, conversation = app.new_chat()
        extra = attach(photo) if photo else None
        input_size = png_size(open(photo, "rb").read()) if photo and keep_shape else None
        result = turn(name, session, [], prompt, filmed(video, input_size), extra=extra)
        if result["clips"]:
            c = result["clips"][0]
            sound = ("a separate WAV" if c.get("audioUrl") else "muxed") if c.get("hasAudio") else "none"
            print(f"    clip: {c['videoUrl']} {c.get('width')}x{c.get('height')}, {c.get('frames')} frames "
                  f"at {c.get('fps')} fps, sound {sound}, seed {c.get('seed')}")
        timing = result["video"]
        if timing:
            rate = (f"{timing['secondsPerStep']:.2f} s a step over {timing['steps']} steps"
                    if timing["secondsPerStep"] is not None else "no step rate")
            phases = ", ".join(f"{p['phase']} at {p['at']:.1f}s" for p in timing["timeline"])
            print(f"    timing: first frame {timing['firstFrame'] or 0:.2f}s, {rate}; {phases or 'no phases'}")
        files = result.get("files") or {}
        for t in (files.get("mp4") or {}).get("tracks", []):
            shape = (f"{t['width']:g}x{t['height']:g}, {t['fps']} fps" if t["handler"] == "vide"
                     else f"{t.get('channels')} ch at {t.get('rate', 0):g} Hz")
            print(f"    track: {t['handler']} {t['codec']} {shape}, {t['samples']} samples, {t['seconds']} s")
        if files.get("wav"):
            w = files["wav"]
            print(f"    wav: {w['channels']} ch at {w['rate']} Hz, {w['bits']}-bit, {w['seconds']} s, RMS {w['rms']}")
        for note in files.get("skipped", []):
            print(f"    skipped: {note}")
        saved_clip(conversation, name, result)

    photo = os.path.join(args.media, "image.png")
    if "film" in wanted:
        video_turn("film", "A red fox trotting through falling snow, cinematic", video_model)
    if "animate" in wanted:
        video_turn("animate", "The scene comes to life: a slow camera push-in, gentle natural motion",
                   keyframes_model, photo=photo, keep_shape=True)
    if "reference" in wanted:
        # Who is in the shot has to be said: the reference supplies how she looks, and a prompt
        # that only says "the subject of the picture" got a well-made garden with no one in it.
        video_turn("reference", "The woman from the picture, long dark hair and a blue floral dress, walks "
                   "through a sunlit garden as the camera slowly circles her",
                   references_model, photo=photo)

    if previous_loras is not None:
        atexit.unregister(restore_loras)
        if not restore_loras():
            failures.append("restoring the LoRA choice")

    # A check that could not run is reported as such; it is never counted as a pass.
    skipped = [f"{row['scenario']}: {note}" for row in rows for note in (row.get("files") or {}).get("skipped", [])]
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"base": base, "engine": engine, "rows": rows, "failures": failures, "skipped": skipped},
                      f, indent=2)
    for note in skipped:
        print(f"SKIPPED (not a pass): {note}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(rows)} chat scenarios passed" + (f", {len(skipped)} check(s) skipped." if skipped else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
