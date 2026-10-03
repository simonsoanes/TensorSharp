#!/usr/bin/env python3
"""Validate TensorAgent's desktop host over its real HTTP/SSE interface.

The connection file is emitted by eng/validation/TensorAgentHost or a
MAUI app launched with TENSORAGENT_VALIDATION_ROOT. Evidence must
be stored under ignored artifacts/ or docs/validation/. No third-party modules.
This checks a small deterministic rubric; it is not a general model quality eval.
"""
import argparse
import json
import math
import pathlib
import re
import statistics
import struct
import time
import urllib.parse
import urllib.request
import uuid
import zlib


def circle_png():
    """An unambiguous synthetic vision fixture: one red circle on white."""
    size = 224
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            raw.extend((235, 15, 15) if (x - 112) ** 2 + (y - 112) ** 2 < 72 ** 2 else (255, 255, 255))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


class Benchmark:
    def __init__(self, connection, output):
        self.connection = connection
        self.output = output
        self.rows = []
        self.checks = []
        self.report = {'connection': {k: v for k, v in connection.items() if k not in ('cookie', 'entryUrl')},
                       'rows': self.rows, 'checks': self.checks,
                       'limitations': ['Single local model and GPU; no statistical model quality or cross-hardware comparison.',
                                       'Measured TTFT includes HTTP and model work; decode rate is (reported tokens - 1)/(wall time - TTFT).',
                                       'Uses AgentAppHost; does not validate the native MAUI window or model download/catalog installation.']}

    def request(self, path, data=None, method=None, content_type='application/json'):
        encoded = json.dumps(data).encode() if data is not None and not isinstance(data, bytes) else data
        return urllib.request.urlopen(urllib.request.Request(self.connection['baseUrl'] + path, data=encoded,
            method=method, headers={'Cookie': self.connection['cookie'], 'Content-Type': content_type}), timeout=600)

    def json(self, path, data=None, method=None):
        with self.request(path, data, method) as response:
            return json.load(response)

    def save(self):
        (self.output / 'results.json').write_text(json.dumps(self.report, indent=2), encoding='utf-8')

    def check(self, name, passed, detail=None):
        self.checks.append({'name': name, 'passed': bool(passed), 'detail': detail})
        print(json.dumps(self.checks[-1]), flush=True)
        self.save()

    def session(self, conversation='new'):
        return self.json('/api/sessions?conversation=' + urllib.parse.quote(conversation), method='POST')

    def ask(self, session, history, scenario, prompt, expected=None, tokens=32, stop_after=None, extra=None):
        message = {'role': 'user', 'content': prompt}
        message.update(extra or {})
        history.append(message)
        body = {'sessionId': session['sessionId'], 'messages': history, 'maxTokens': tokens,
                'think': False, 'temperature': 0, 'topK': 1, 'seed': 42}
        started = time.perf_counter()
        row = {'scenario': scenario, 'prompt': prompt, 'answer': '', 'frames': [], 'ttftSeconds': None}
        stop_sent = False
        token_frames = 0
        try:
            with self.request('/api/chat', body) as response:
                turn = response.headers.get('X-TensorAgent-Turn')
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    frame = json.loads(line[6:])
                    row['frames'].append(frame)
                    if frame.get('token') or frame.get('thinking'):
                        if row['ttftSeconds'] is None:
                            row['ttftSeconds'] = time.perf_counter() - started
                    if isinstance(frame.get('token'), str):
                        row['answer'] += frame['token']
                        token_frames += 1
                    if isinstance(frame.get('replace'), str):
                        row['answer'] = frame['replace']
                    if stop_after and token_frames >= stop_after and not stop_sent and turn:
                        row['stopResponse'] = self.json('/api/agent/turns/' + urllib.parse.quote(turn) + '/stop', method='POST')
                        stop_sent = True
                    if frame.get('error'):
                        row['error'] = frame['error']
                    if frame.get('done'):
                        row['done'] = frame
        except Exception as error:
            row['error'] = str(error)
        row['totalSeconds'] = time.perf_counter() - started
        done = row.get('done', {})
        count = done.get('tokenCount', 0)
        # Agentic tokenCount includes hidden tool-call rounds that precede the
        # first displayed token. Dividing it by final-answer time inflates rates.
        has_tools = any('tool_progress' in frame for frame in row['frames'])
        row['decodeTokensPerSecond'] = ((count - 1) / (row['totalSeconds'] - row['ttftSeconds'])
            if not has_tools and count > 1 and row['ttftSeconds'] is not None and row['totalSeconds'] > row['ttftSeconds'] else None)
        row['passed'] = bool(done) and not row.get('error') and (bool(stop_after) or not done.get('aborted'))
        if expected is not None:
            row['expected'] = expected
            row['passed'] &= re.sub(r'[^a-z0-9]', '', row['answer'].lower()) == expected
        if stop_after:
            row['passed'] &= stop_sent and done.get('aborted') is True
        history.append({'role': 'assistant', 'content': row['answer']})
        self.rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'frames'}), flush=True)
        self.save()
        return row

    def artifact(self):
        result = self.ask(self.session(), [], 'python-artifact',
            'Use the shell tool to run exactly this command:\n'
            'python3 -c "from pathlib import Path; Path(\'validation-result.txt\').write_text(\'TensorAgent validation: 42\', encoding=\'utf-8\'); print(6*7)"\n'
            'Then reply with the number it printed.', tokens=256)
        files = [file for frame in result['frames'] for file in frame.get('files', [])
                 if file.get('name') == 'validation-result.txt']
        self.check('Python execution publishes the generated artifact', bool(files))
        if files:
            with self.request(files[0]['url']) as response:
                content = response.read()
                disposition = response.headers.get('Content-Disposition', '')
            self.check('generated artifact downloads with exact bytes and attachment disposition',
                content == b'TensorAgent validation: 42' and 'attachment' in disposition.lower())
        return result

    def run(self, vision=False, tools=False, throughput_runs=3):
        settings = self.json('/api/agent/settings')
        self.report['settings'] = settings
        self.report['models'] = self.json('/api/models')
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            self.json('/api/agent/settings')
            latencies.append((time.perf_counter() - start) * 1000)
        self.report['settingsLatencyMs'] = {'samples': len(latencies), 'median': statistics.median(latencies),
                                           'p95': sorted(latencies)[math.ceil(len(latencies) * .95) - 1], 'max': max(latencies)}
        session, history = self.session(), []
        for index, (prompt, expected) in enumerate([
            ('Say the single word: apple.', 'apple'), ('Now say the single word: banana.', 'banana'),
            ('What is 17 times 23? Reply with just the number.', '391'),
            ('What is 17 times 24? Reply with just the number.', '408'),
            ('Now say the single word: cherry.', 'cherry')]):
            self.ask(session, history, 'first-chat' if index == 0 else 'follow', prompt, expected)
        saved = self.json('/api/agent/conversations/' + session['conversationId'])
        self.check('persisted transcript contains all five answers',
                   len(saved.get('messages', [])) == 10 and saved['messages'][-1]['content'] == history[-1]['content'])
        reopened = self.session(session['conversationId'])
        self.check('reopening a conversation restores transcript', len(reopened.get('messages', [])) == 10)
        self.ask(reopened, history, 'reopen', 'Now say the single word: pear.', 'pear')
        fresh, fresh_history = self.session(), []
        self.ask(fresh, fresh_history, 'newchat', 'Say the single word: kiwi.', 'kiwi')
        throughput = []
        for index in range(throughput_runs):
            result = self.ask(fresh, fresh_history, 'throughput',
                f'Write a detailed factual explanation of how rain forms, version {index + 1}. Continue for at least 200 words.', tokens=128)
            self.check(f'throughput sample {index + 1} generated 128 tokens', result.get('done', {}).get('tokenCount') == 128)
            if result['passed'] and result['decodeTokensPerSecond'] is not None:
                throughput.append(result['decodeTokensPerSecond'])
        self.report['throughputTokensPerSecond'] = {'samples': len(throughput),
            'median': statistics.median(throughput) if throughput else None,
            'min': min(throughput) if throughput else None, 'max': max(throughput) if throughput else None}
        self.ask(fresh, fresh_history, 'stop', 'Write a long essay of at least 1000 words about the oceans.', tokens=512, stop_after=8)
        self.ask(fresh, fresh_history, 'after-stop', 'Now say the single word: mango.', 'mango')
        if tools:
            changed = dict(settings, allowUnconfinedExecution=True)
            self.json('/api/agent/settings', changed, 'POST')
            try:
                tool_session = self.session()
                result = self.ask(tool_session, [], 'tool', 'Use the shell tool to run exactly this command: Write-Output hello-from-shell\nThen tell me what it printed, nothing else.', tokens=256)
                tool_frames = [f for f in result['frames'] if any('tool' in k.lower() for k in f)]
                self.check('shell tool emitted trace and returned expected output',
                           any(f.get('tool') == 'shell' and f.get('tool_progress') == 'finished' for f in tool_frames)
                           and any('hello-from-shell' in (f.get('text') or '') and f.get('tool_progress') == 'running' for f in tool_frames)
                           and 'hello-from-shell' in result['answer'], tool_frames)
                self.artifact()
            finally:
                self.json('/api/agent/settings', settings, 'POST')
        if vision:
            png = circle_png()
            (self.output / 'red-circle.png').write_bytes(png)
            boundary = 'TensorAgentBench' + uuid.uuid4().hex
            data = ('--' + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="red-circle.png"\r\nContent-Type: image/png\r\n\r\n').encode() + png + ('\r\n--' + boundary + '--\r\n').encode()
            with self.request('/api/upload', data, content_type='multipart/form-data; boundary=' + boundary) as response:
                upload = json.load(response)
            self.report['upload'] = upload
            file = upload.get('file')
            if file is None:
                file = upload.get('files', [{}])[0].get('file')
            self.check('image upload accepted', bool(file), upload)
            if file:
                result = self.ask(self.session(), [], 'vision', 'Describe the color and shape in this image in one short sentence.', tokens=64,
                    extra={'imagePaths': [file], 'attachments': [{'file': file, 'fileName': 'red-circle.png', 'mediaType': 'image'}]})
                answer = result['answer'].lower()
                self.check('vision identifies red circle', 'red' in answer and any(word in answer for word in ('circle', 'circular', 'disc', 'dot', 'round')), answer)
        self.save()
        return all(r['passed'] for r in self.rows) and all(c['passed'] for c in self.checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--tools', action='store_true', help='Temporarily enable unconfined execution for a harmless Windows Write-Output check.')
    parser.add_argument('--throughput-runs', type=int, default=3)
    args = parser.parse_args()
    if args.throughput_runs < 1:
        parser.error('--throughput-runs must be positive')
    repo = pathlib.Path(__file__).resolve().parents[2]
    output = args.out.resolve()
    if not any(output.is_relative_to(repo / allowed) for allowed in ('artifacts', 'docs/validation')):
        parser.error('--out must be under repository artifacts/ or docs/validation/')
    output.mkdir(parents=True, exist_ok=True)
    bench = Benchmark(json.loads(args.connection.read_text(encoding='utf-8-sig')), output)
    try:
        return 0 if bench.run(args.vision, args.tools, args.throughput_runs) else 1
    finally:
        bench.save()


if __name__ == '__main__':
    raise SystemExit(main())
