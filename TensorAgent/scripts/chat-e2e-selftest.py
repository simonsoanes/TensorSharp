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
"""Runs chat-e2e.py's video checks against files made here, with no app and no model:

  /usr/bin/python3 TensorAgent/scripts/chat-e2e-selftest.py

Only the app with MiniMax-H3 loaded makes a real clip, and a check first run there finds
its own bugs minutes into a generation. So here the MP4 parser reads an H.264 clip that
AVFoundation writes (through cv2's 'avc1' writer), AAC that afconvert writes from a WAV
made with the wave module, and the two muxed into one file, which is how the app
delivers a clip with sound; the WAV checks read good and bad WAVs; the Range/HEAD probe
runs against a stub that honours Range and HEAD and against stubs that each break one
of them; and film, animate and reference run end to end against a stub of the app's
routes that streams the contract's frames, then against stubs broken one way at a time
(the wrong entry loaded, a stopped turn, a second clip, a mislabelled or blank file, a
saved chat that lost the clip), each of which the scenario must fail.

What cannot run on this machine (no cv2, no afconvert) is reported as skipped, never as
passed. Exits non-zero when a check fails.
"""
import array
import contextlib
import http.server
import importlib.util
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

_spec = importlib.util.spec_from_file_location(
    "chat_e2e", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat-e2e.py"))
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)

# The contract's example clip: 22 frames at 24 fps, 640x384, with 32 kHz stereo sound.
FRAMES, FPS, SECONDS = 22, 24, 22 / 24
counts = {"ok": 0, "FAIL": 0, "skip": 0}


def check(name, problem, detail=""):
    """One check: `problem` is "" when it passed, otherwise why it did not."""
    status = "FAIL" if problem else "ok"
    counts[status] += 1
    print(f"{status:4} {name}" + (f": {problem}" if problem else f" ({detail})" if detail else ""))


def fails(name, problem, words):
    """A check that must fail, and say so in these words."""
    check(name, "" if problem and words in problem else f"expected a failure saying {words!r}, got {problem!r}",
          problem)


def skip(name, why):
    counts["skip"] += 1
    print(f"skip {name}: {why}")


def describe(t):
    shape = (f"{t['width']:g}x{t['height']:g}, {t['fps']} fps" if t.get("handler") == "vide"
             else f"{t.get('channels')} ch at {t.get('rate', 0):g} Hz")
    return (f"{t.get('handler')} {t.get('codec')} {shape}, {t.get('samples')} samples, "
            f"{t.get('seconds')} s (media {t.get('mediaSeconds')} s)")


@contextlib.contextmanager
def quiet_stderr():
    """cv2's AVFoundation writer prints a line per frame to fd 2 from native code."""
    saved = os.dup(2)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def make_clip(path, width, height, blank=False):
    """An H.264 MP4 from AVFoundation, the framework the app encodes with, through cv2."""
    import cv2
    import numpy as np
    with quiet_stderr():
        writer = cv2.VideoWriter(path, cv2.CAP_AVFOUNDATION, cv2.VideoWriter_fourcc(*"avc1"), FPS, (width, height))
        if not writer.isOpened():
            return False
        for i in range(FRAMES):
            image = np.zeros((height, width, 3), np.uint8)
            if not blank:
                image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
                image[:, :, 1] = i * 11
                cv2.circle(image, (width * (i + 1) // (FRAMES + 1), height // 2), height // 8, (255, 255, 255), -1)
            writer.write(image)
        writer.release()
    return os.path.getsize(path) > 0


def make_wav(path, seconds, channels=2, rate=32000, amplitude=8000):
    """16-bit PCM with a tone in each channel, or silence at amplitude 0."""
    pcm = array.array("h", (int(amplitude * math.sin(2 * math.pi * (440 + 220 * c) * i / rate))
                            for i in range(int(round(seconds * rate))) for c in range(channels)))
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def mux(video, audio):
    """The clip with the AAC file's one track added: what the app delivers when the sound
    is muxed. Nothing in the standard library or on a stock Mac muxes two files, so this
    moves boxes. The video's moov (AVFoundation writes it last) gains the audio's trak, the
    audio's mdat goes after it, and that trak's track id, chunk offsets and movie-timescale
    durations are rewritten. The caller checks that afinfo reads the result."""
    vmoov, amoov = e2e.mp4_box(video, 0, len(video), "moov"), e2e.mp4_box(audio, 0, len(audio), "moov")
    amdat = e2e.mp4_box(audio, 0, len(audio), "mdat")
    if vmoov[1] != len(video):
        raise ValueError("the clip's moov is not its last box")

    def movie_scale(data, moov):
        mvhd = e2e.mp4_box(data, *moov, "mvhd")[0]
        return struct.unpack_from(">I", data, mvhd + (20 if data[mvhd] == 1 else 12))[0]

    vscale, ascale = movie_scale(video, vmoov), movie_scale(audio, amoov)
    start, end = e2e.mp4_box(audio, *amoov, "trak")
    trak, base = bytearray(audio[start - 8:end]), start - 8

    def at(*path):
        body = e2e.mp4_box(audio, start, end, *path)
        return body and body[0] - base

    tkhd, elst, stco = at("tkhd"), at("edts", "elst"), at("mdia", "minf", "stbl", "stco")
    if trak[tkhd] != 0 or (elst and trak[elst] != 0) or not stco:
        raise ValueError("expected afconvert's version-0 tkhd and elst and a 32-bit stco")

    def patch(offset, change):
        struct.pack_into(">I", trak, offset, change(struct.unpack_from(">I", trak, offset)[0]))

    patch(tkhd + 12, lambda _: 2)
    patch(tkhd + 20, lambda d: d * vscale // ascale)
    for i in range(struct.unpack_from(">I", trak, elst + 4)[0] if elst else 0):
        patch(elst + 8 + 12 * i, lambda d: d * vscale // ascale)
    prefix = video[:vmoov[0] - 8]
    moov_size = 8 + (vmoov[1] - vmoov[0]) + len(trak)
    shift = len(prefix) + moov_size + 8 - amdat[0]
    for i in range(struct.unpack_from(">I", trak, stco + 4)[0]):
        patch(stco + 8 + 4 * i, lambda o: o + shift)
    return (prefix + struct.pack(">I4s", moov_size, b"moov") + video[vmoov[0]:vmoov[1]] + bytes(trak)
            + struct.pack(">I4s", 8 + amdat[1] - amdat[0], b"mdat") + audio[amdat[0]:amdat[1]])


def claim_size(clip, width, height):
    """The clip with its tkhd claiming another size. The decoder does not read tkhd (cv2
    decodes the coded frames), so only the decoded-frame check can tell."""
    data = bytearray(clip)
    tkhd = e2e.mp4_box(data, *e2e.mp4_box(data, 0, len(data), "moov"), "trak", "tkhd")
    struct.pack_into(">II", data, tkhd[1] - 8, width << 16, height << 16)
    return bytes(data)


class Stub(http.server.ThreadingHTTPServer):
    """The app's routes as a video turn uses them, on a port of its own. `video` is what
    GET /api/models reports, `plan(user message)` the clip frame a turn ends with (a list
    of them for a turn that makes more than one, None for a turn that is stopped), and
    `files` are served under /uploads with Range and HEAD. `fault` breaks that file route:
    "ranges" answers every request with the whole file, as a plain file server does,
    "mid-range" serves the slice one byte past the one asked for after the first, and
    "head" answers HEAD as a route table with only GET does. `save` False leaves the turn
    out of the saved conversation, and a dict changes what is saved."""
    daemon_threads = True

    def __init__(self, video=None, plan=None, files=None, fault=None, save=True):
        super().__init__(("127.0.0.1", 0), StubHandler)
        self.video, self.plan, self.files, self.fault, self.save = video, plan, dict(files or {}), fault, save
        self.sessions, self.conversations, self.chats, self.uploads = {}, {}, [], 0
        self.url = f"http://127.0.0.1:{self.server_address[1]}/"
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()


class StubHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self):
        self.send_file(head=True)

    def do_GET(self):
        s, route = self.server, self.path.split("?")[0]
        if route.startswith("/uploads/"):
            return self.send_file()
        if route == "/api/agent/engine":
            return self.reply(200, {"engine": "chat-e2e self-test stub", "model": {"id": "stub", "state": "Loaded"}})
        if route == "/api/models":
            return self.reply(200, {"architecture": "minimax-h3" if s.video else "gemma4", "video": s.video})
        found = re.fullmatch(r"/api/agent/conversations/([^/]+)", route)
        if found and found.group(1) in s.conversations:
            return self.reply(200, {"id": found.group(1), "messages": s.conversations[found.group(1)]})
        self.reply(404, {"error": f"no route {route}"})

    def do_POST(self):
        s, route = self.server, self.path.split("?")[0]
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if route == "/api/sessions":
            n = len(s.sessions) + 1
            s.sessions[f"s{n}"], s.conversations[f"c{n}"] = f"c{n}", []
            return self.reply(200, {"sessionId": f"s{n}", "conversationId": f"c{n}", "activeTurn": None})
        if route == "/api/upload":
            s.uploads += 1
            return self.reply(200, {"file": f"upload-{s.uploads}.png", "fileName": "image.png", "mediaType": "image"})
        if route == "/api/chat":
            return self.turn(json.loads(body))
        self.reply(404, {"error": f"no route {route}"})

    def turn(self, body):
        """One video turn's frames: text-encode, every denoising step (each also as the
        phase-less step frame the server's own stream sends), the decodes and the encode,
        then the clip and done, with a keep-alive comment the reader must skip. A stopped
        turn ends partway through denoising with an aborted done and nothing saved."""
        s, steps = self.server, 8
        s.chats.append(body)
        user = body["messages"][-1]
        plan = s.plan(user)
        clips = [] if plan is None else plan if isinstance(plan, list) else [plan]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        frames = [{"video_step": 0, "video_steps": steps, "video_phase": "text-encode", "elapsed": 0.0, "eta": -1}]
        for k in range(1, steps + 1):
            frames += [{"video_step": k, "video_steps": steps},
                       {"video_step": k, "video_steps": steps, "video_phase": "denoise",
                        "elapsed": 0.01 * k, "eta": 0.01 * (steps - k)}]
        frames += [{"video_step": steps, "video_steps": steps, "video_phase": p, "elapsed": 0.1, "eta": -1}
                   for p in ("vae-decode", "audio-decode", "encode")]
        self.wfile.write(b": keep-alive\n\n")
        for frame in (frames if clips else frames[:5]) + clips:
            if frame.get("video_phase") == "denoise":
                time.sleep(0.005)
            self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
            self.wfile.flush()
        if not clips:
            done = {"done": True, "aborted": True, "sessionId": body["sessionId"]}
        else:
            if s.save:
                kept = {"role": "assistant", "content": "", "videoUrl": clips[0]["videoUrl"],
                        "audioUrl": clips[0].get("audioUrl")}
                kept.update(s.save if isinstance(s.save, dict) else {})
                s.conversations[s.sessions[body["sessionId"]]] += [user, kept]
            done = {"done": True, "sessionId": body["sessionId"], "tokenCount": 0, "elapsed": 0.2, "tokPerSec": 0.0,
                    "truncated": False}
        self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")

    def send_file(self, head=False):
        s = self.server
        if self.path not in s.files:
            return self.reply(404, {"error": "no such file"})
        if head and s.fault == "head":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data, kind = s.files[self.path]
        wanted = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "") if s.fault != "ranges" else None
        start, end = 0, len(data) - 1
        if wanted:
            start, end = int(wanted.group(1)), min(int(wanted.group(2) or end), end)
        shift = 1 if wanted and start and s.fault == "mid-range" else 0
        self.send_response(206 if wanted else 200)
        if wanted:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if not head:
            self.wfile.write(data[start + shift:end + 1 + shift])


def run(folder):
    def path(name):
        return os.path.join(folder, name)

    def read(name):
        with open(path(name), "rb") as f:
            return f.read()

    try:
        import cv2
    except ImportError:
        cv2 = None
    afconvert = shutil.which("afconvert")

    # The progress frames: the contract's order, and the four ways a stream can break it.
    def frames(*phases, last=8):
        return [(0.1 * i, {"video_step": step, "video_steps": last, "video_phase": phase})
                for i, (phase, step) in enumerate(phases)]
    good = frames(("text-encode", 0), *[("denoise", k) for k in range(1, 9)], ("vae-decode", 8), ("encode", 8))
    check("progress: text-encode, 8 denoising steps, vae-decode, encode", e2e.progress_problem(good))
    timing = e2e.video_timing(good, 0.1, 1.2)
    check("progress: timeline and seconds per step",
          "" if [p["phase"] for p in timing["timeline"]] == ["text-encode", "denoise", "vae-decode", "encode"]
          and timing["secondsPerStep"] == 0.1 and timing["steps"] == 8 else f"read {timing}",
          f"{timing['secondsPerStep']} s a step")
    fails("progress: a phase outside the contract", e2e.progress_problem(good + frames(("done", 8))), "unknown phase")
    fails("progress: a phase going back", e2e.progress_problem(good + frames(("denoise", 8))), "went backwards")
    fails("progress: denoising stopped early", e2e.progress_problem(good[:5]), "stopped at step 4 of 8")
    fails("progress: no denoising at all", e2e.progress_problem(frames(("text-encode", 0))), "no denoising")

    # The WAV checks.
    for name, seconds, extra in (("tone", SECONDS, {}), ("silent", SECONDS, {"amplitude": 0}),
                                 ("mono", SECONDS, {"channels": 1, "rate": 44100}), ("short", 0.5, {})):
        make_wav(path(f"{name}.wav"), seconds, **extra)
    tone = read("tone.wav")
    facts = e2e.wav_facts(tone)
    check("wav: 16-bit stereo at 32 kHz, the clip's length, audible", e2e.wav_problem(facts, SECONDS),
          f"{facts['seconds']} s, RMS {facts['rms']}")
    fails("wav: silence", e2e.wav_problem(e2e.wav_facts(read("silent.wav")), SECONDS), "silent")
    fails("wav: mono at 44.1 kHz", e2e.wav_problem(e2e.wav_facts(read("mono.wav")), SECONDS), "1 channels at 44100 Hz")
    fails("wav: half a second", e2e.wav_problem(e2e.wav_facts(read("short.wav")), SECONDS), "lasts 0.500 s")
    try:
        e2e.wav_facts(b"RIFX not a wave")
        problem = "it parsed"
    except wave.Error as ex:
        problem = f"wave.Error: {ex}"
    fails("wav: bytes that are not a WAV raise wave.Error", problem, "wave.Error")

    # The MP4 parser on what it must refuse.
    for name, data in (("no moov", b"\0\0\0\x10ftypisom\0\0\0\0" + b"\0\0\0\x08free"),
                       ("a box running past the file", b"\0\0\0\x10ftypisom\0\0\0\0\0\0\x10\0moov")):
        try:
            e2e.mp4_tracks(data)
            problem = "it parsed"
        except (ValueError, struct.error) as ex:
            problem = f"{type(ex).__name__}: {ex}"
        fails(f"mp4: {name} is refused", problem, "Error")

    clip_ok = cv2 is not None and make_clip(path("wide.mp4"), 640, 384) and make_clip(path("photo.mp4"), 624, 416) \
        and make_clip(path("blank.mp4"), 640, 384, blank=True)
    if not clip_ok:
        for name in ("mp4: the AVFoundation clip", "frame: the middle frame", "mp4: the muxed clip"):
            skip(name, "cv2 is not importable" if cv2 is None else "cv2 could not open an AVFoundation 'avc1' writer")
    else:
        wide = read("wide.mp4")
        tracks = e2e.mp4_tracks(wide)
        v = tracks[0] if len(tracks) == 1 else {}
        check("mp4: AVFoundation H.264 clip -> one 'vide' avc1 track, 22 samples at 24 fps, 640x384",
              "" if [(v.get(k)) for k in ("handler", "codec", "samples", "timedSamples", "fps", "width", "height")]
              == ["vide", "avc1", 22, 22, 24.0, 640.0, 384.0] else f"read {tracks}", describe(v))
        frame = e2e.frame_facts(wide, FRAMES) or {}
        check("frame: cv2 decodes the middle frame at 640x384 and it is not blank",
              "" if frame.get("decoded") and (frame["width"], frame["height"]) == (640, 384) and frame["std"] > 5
              else f"read {frame}", f"frame {frame.get('frame')}, pixel std {frame.get('std')}")
        frame = e2e.frame_facts(read("blank.mp4"), FRAMES) or {}
        check("frame: an all-black clip reads as blank", "" if frame.get("decoded") and frame["std"] <= 5
              else f"read {frame}", f"pixel std {frame.get('std')}")

    muxed = None
    if not afconvert:
        skip("mp4: AAC from afconvert", "afconvert is macOS's; it is not on PATH")
    else:
        for container, name in (("mp4f", "tone.mp4"), ("m4af", "tone.m4a")):
            subprocess.run([afconvert, "-f", container, "-d", "aac", path("tone.wav"), path(name)],
                           check=True, capture_output=True)
            tracks = e2e.mp4_tracks(read(name))
            s = tracks[0] if len(tracks) == 1 else {}
            check(f"mp4: afconvert AAC ({container}) -> one 'soun' mp4a track, stereo 32000 Hz, the clip's length",
                  "" if [s.get(k) for k in ("handler", "codec", "channels", "rate")] == ["soun", "mp4a", 2, 32000]
                  and abs(s["seconds"] - SECONDS) <= 0.15 else f"read {tracks}", describe(s))
        if clip_ok:
            muxed = mux(read("wide.mp4"), read("tone.mp4"))
            with open(path("muxed.mp4"), "wb") as f:
                f.write(muxed)
            info = subprocess.run(["afinfo", path("muxed.mp4")], capture_output=True, text=True).stdout
            found = re.search(r"2 ch,\s+32000 Hz, aac", info)
            check("fixture: afinfo reads the muxed clip's AAC track", "" if found else f"afinfo said {info[:300]!r}",
                  found and found.group(0))
            tracks = e2e.mp4_tracks(muxed)
            check("mp4: the muxed clip -> a 'vide' and a 'soun' track",
                  "" if [t["handler"] for t in tracks] == ["vide", "soun"] else f"read {tracks}",
                  "; ".join(describe(t) for t in tracks))
            reported = {"videoUrl": "/uploads/video-muxed.mp4", "width": 640, "height": 384, "frames": FRAMES,
                        "fps": FPS, "hasAudio": True}
            makes_sound = {"supportsAudio": True}
            only_video = e2e.mp4_tracks(read("wide.mp4"))
            check("clip: the muxed clip is what the turn reported", e2e.clip_problem(reported, tracks, makes_sound))
            fails("clip: a frame count the file does not have", e2e.clip_problem({**reported, "frames": 21}, tracks),
                  "the turn reported 21 frames")
            fails("clip: a frame rate the file does not have", e2e.clip_problem({**reported, "fps": 25}, tracks),
                  "the turn reported 25")
            fails("clip: a size the file does not have", e2e.clip_problem({**reported, "width": 320}, tracks),
                  "the turn reported 320x384")
            fails("clip: sound both muxed and a separate WAV",
                  e2e.clip_problem({**reported, "audioUrl": "/uploads/x.wav"}, tracks), "play it twice")
            fails("clip: a sound track in a clip that says it has none",
                  e2e.clip_problem({**reported, "hasAudio": False}, tracks), "hasAudio is false")
            fails("clip: no sound from a model that makes it",
                  e2e.clip_problem({**reported, "hasAudio": False}, only_video, makes_sound), "makes sound")
            fails("clip: muxed sound missing from the MP4", e2e.clip_problem(reported, only_video),
                  "has 0 sound tracks")
            # The sound track's own facts, read from what afconvert makes of the wrong WAVs.
            for name, words in (("mono", "1 channels at 44100 Hz"), ("short", "the sound track lasts")):
                subprocess.run([afconvert, "-f", "mp4f", "-d", "aac", path(f"{name}.wav"), path(f"{name}.mp4")],
                               check=True, capture_output=True)
                fails(f"clip: muxed sound that is {name}",
                      e2e.clip_problem(reported, e2e.mp4_tracks(mux(read("wide.mp4"), read(f"{name}.mp4")))), words)

    # The Range/HEAD probe, on the clip when there is one and on the WAV otherwise.
    served = read("wide.mp4") if clip_ok else tone
    files = {"/uploads/video-x.mp4": (served, "video/mp4")}
    with Stub(files=files) as honest:
        check("range: a server that honours Range and HEAD",
              e2e.range_problem(e2e.App(honest.url, "f00d"), "/uploads/video-x.mp4", served), f"{len(served)} bytes")
    for fault, name, words in (("ranges", "answers Range with the whole file", "Range bytes=0-1 answered 200"),
                               ("mid-range", "serves the wrong slice from the middle", "not that slice of the file"),
                               ("head", "does not answer HEAD", "HEAD answered 404")):
        with Stub(files=files, fault=fault) as broken:
            fails(f"range: a server that {name}",
                  e2e.range_problem(e2e.App(broken.url, None), "/uploads/video-x.mp4", served), words)

    if muxed is None:
        skip("scenarios: film, animate and reference against the app stub", "they need the clips and the muxed fixture")
        return
    media = path("media")
    os.mkdir(media)
    import numpy as np
    cv2.imwrite(os.path.join(media, "image.png"), np.full((836, 1254, 3), 128, np.uint8))
    keyframes = {"family": "minimax-h3", "supportsAudio": True, "supportsImageConditioning": True,
                 "supportsEndImageConditioning": True, "supportsReferenceConditioning": False,
                 "maxReferenceImages": 0}
    references = {**keyframes, "supportsEndImageConditioning": False, "supportsReferenceConditioning": True,
                  "maxReferenceImages": 9}
    muxed_clip = {"videoUrl": "/uploads/video-muxed.mp4", "audioUrl": None, "width": 640, "height": 384,
                  "frames": FRAMES, "fps": FPS, "seed": 12345, "hasAudio": True}
    photo_clip = {"videoUrl": "/uploads/video-photo.mp4", "audioUrl": "/uploads/video-photo.wav", "width": 624,
                  "height": 416, "frames": FRAMES, "fps": FPS, "seed": 12345, "hasAudio": True}
    files = {"/uploads/video-muxed.mp4": (muxed, "video/mp4"), "/uploads/video-photo.mp4": (read("photo.mp4"), "video/mp4"),
             "/uploads/video-photo.wav": (tone, "audio/wav")}

    def by_photo(user):
        return photo_clip if user.get("stillImagePaths") else muxed_clip

    def scenarios(name, stub, wanted, show=False):
        """chat-e2e.py's own main() against the stub: (exit code, report, what it printed)."""
        out, report = io.StringIO(), path(f"{name}.json")
        with contextlib.redirect_stdout(out):
            code = e2e.main(["--base", stub.url, "--scenarios", wanted, "--media", media, "--out", report])
        if show:
            print("\n".join("     | " + line for line in out.getvalue().rstrip().splitlines()))
        with open(report, encoding="utf-8") as f:
            return code, json.load(f), out.getvalue()

    with Stub(keyframes, by_photo, files) as stub:
        code, report, _ = scenarios("keyframes", stub, "film,animate", show=True)
        film, animate = (report["rows"] + [{}, {}])[:2]
        sent = stub.chats[-1]["messages"][-1] if stub.chats else {}
        check("scenarios: film (muxed sound) and animate (a separate WAV) pass on the keyframes entry",
              "" if code == 0 and film.get("ok") and animate.get("ok") else f"exit {code}, rows {report['rows']}")
        check("scenarios: animate sent the uploaded photo as stillImagePaths",
              "" if sent.get("stillImagePaths") == ["upload-1.png"] else f"sent {sent}")
        video = film.get("video") or {}
        tracks = ((film.get("files") or {}).get("mp4") or {}).get("tracks", [])
        wav = (animate.get("files") or {}).get("wav") or {}
        check("report: phase timeline, step rate, first frame and file facts",
              "" if [p["phase"] for p in video.get("timeline", [])] == list(e2e.VIDEO_PHASES)
              and video.get("steps") == 8 and (video.get("secondsPerStep") or 0) > 0 and video.get("firstFrame")
              and [t["handler"] for t in tracks] == ["vide", "soun"] and wav.get("channels") == 2
              and "progress" not in film else f"film row {film}",
              f"{video.get('secondsPerStep')} s a step, first frame {video.get('firstFrame') or 0:.3f} s")
    with Stub(references, lambda user: muxed_clip, files) as stub:
        code, report, _ = scenarios("references", stub, "reference", show=True)
        sent = stub.chats[-1]["messages"][-1] if stub.chats else {}
        check("scenarios: reference passes on the references entry, the photo sent as stillImagePaths",
              "" if code == 0 and report["rows"][0]["ok"] and sent.get("stillImagePaths") == ["upload-1.png"]
              else f"exit {code}, rows {report['rows']}")
        # Without cv2 the frame check is reported as skipped: in the output, in the report,
        # and in the last line, and the scenario still passes on what could be checked.
        sys.modules["cv2"] = None  # makes `import cv2` raise ImportError
        try:
            code, report, text = scenarios("no-cv2", stub, "film")
        finally:
            sys.modules["cv2"] = cv2
        check("scenarios: without cv2 the frame decode is reported as skipped, not passed",
              "" if code == 0 and report["skipped"] == ["film: middle-frame decode: cv2 is not importable"]
              and "1 check(s) skipped" in text else f"exit {code}, skipped {report['skipped']}",
              "; ".join(report["skipped"]))

    def must_fail(name, wanted, words, sent=True, **stub):
        """A run against the keyframes stub, changed by `stub`, that must exit 1 for this
        reason; `sent` False when it must be refused before any turn reaches /api/chat."""
        with Stub(**{"video": keyframes, "plan": by_photo, "files": files, **stub}) as s:
            code, report, _ = scenarios(name, s, wanted)
        why = report["rows"][0]["why"] if report["rows"] else ""
        check(f"scenarios: {name}", "" if code == 1 and words in why and bool(s.chats) == sent
              else f"exit {code}, {len(s.chats)} turn(s) sent, why {why!r}", why)

    # One run per way a scenario must notice that something is wrong, so a check that
    # stopped checking fails here too.
    blank = {**photo_clip, "videoUrl": "/uploads/video-blank.mp4", "width": 640, "height": 384}
    resized = {**photo_clip, "videoUrl": "/uploads/video-resized.mp4", "width": 640, "height": 384}
    files.update({"/uploads/video-blank.mp4": (read("blank.mp4"), "video/mp4"),
                  "/uploads/video-resized.mp4": (claim_size(read("photo.mp4"), 640, 384), "video/mp4")})
    must_fail("reference on the keyframes entry is refused before any turn", "reference",
              "needs the MiniMax-H3 References entry", sent=False)
    must_fail("animate on the references entry is refused before any turn", "animate",
              "needs the MiniMax-H3 keyframes entry", sent=False, video=references)
    must_fail("film with no video model loaded is refused before any turn", "film",
              "no video model is loaded", sent=False, video=None)
    must_fail("a stopped turn fails film", "film", "(stopped)", plan=lambda user: None)
    must_fail("two clips from one turn fail film", "film", "exactly one videoUrl, got 2",
              plan=lambda user: [muxed_clip, muxed_clip])
    must_fail("a frame count the MP4 does not have fails film", "film", "the turn reported 21 frames",
              plan=lambda user: {**muxed_clip, "frames": 21})
    must_fail("an MP4 served as another type fails film", "film", "not an MP4",
              files={**files, "/uploads/video-muxed.mp4": (muxed, "application/octet-stream")})
    must_fail("a WAV served as another type fails animate", "animate", "not a WAV",
              files={**files, "/uploads/video-photo.wav": (tone, "application/octet-stream")})
    must_fail("a blank clip fails film", "film", "the middle frame is blank", plan=lambda user: blank)
    must_fail("frames that decode at another size than the MP4 says fail film", "film",
              "the decoded frame is 624x416", plan=lambda user: resized)
    must_fail("a clip that is not the photo's shape fails animate", "animate", "not the photo's shape",
              plan=lambda user: muxed_clip)
    must_fail("a clip served without Range fails film", "film", "Range bytes=0-1 answered 200", fault="ranges")
    must_fail("a clip missing from the saved conversation fails film", "film", "no clip", save=False)
    must_fail("another clip in the saved conversation fails film", "film", "the saved clip is",
              save={"videoUrl": "/uploads/video-other.mp4"})
    must_fail("a saved chat that lost the WAV fails animate", "animate", "the saved audioUrl",
              save={"audioUrl": None})


def main():
    folder = tempfile.mkdtemp(prefix="chat-e2e-selftest-")
    try:
        run(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(f"self-test: {counts['ok']} passed, {counts['FAIL']} failed, {counts['skip']} skipped")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
