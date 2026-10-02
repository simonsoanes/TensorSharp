#!/usr/bin/env python3
"""Exercise the shipped mask editor and Server Chat in isolated desktop/mobile Chromium.

Uses a loopback fixture for uploads/SSE; no model inference is claimed. Checks
native-size grayscale export, brush/erase/undo, touch coordinates, pan, selection
reuse and request wiring. Real model tests use qwen-image21-mask-bench.py.
"""
import argparse
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def fixture():
    uploads, requests = {}, []
    image = io.BytesIO()
    Image.new('RGB', (640, 360), (90, 130, 180)).save(image, format='PNG')
    uploads['source.png'] = image.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type='application/json'):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                self.respond((ROOT / 'TensorSharp.Server.Host/wwwroot/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path in ('/mask-editor.js', '/mask-editor.css'):
                self.respond((ROOT / 'TensorSharp.Chat/WebUi' / path[1:]).read_bytes(),
                             'text/javascript' if path.endswith('js') else 'text/css')
            elif path.startswith('/uploads/') and path[9:] in uploads:
                self.respond(uploads[path[9:]], 'image/png')
            elif path == '/api/models':
                self.respond(b'{"loaded":"Mask editor fixture","architecture":"qwen_image"}')
            elif path == '/api/queue/status':
                self.respond(b'{"processing":0,"pending_requests":0}')
            elif path == '/api/skills':
                self.respond(b'{"skills":[]}')
            else:
                self.send_error(404)

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            if self.path == '/api/sessions':
                self.respond(b'{"sessionId":"mask-fixture"}')
            elif self.path == '/api/upload':
                message = BytesParser(policy=default).parsebytes(
                    ('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + data)
                part = next(message.iter_parts())
                filename = 'selection.png' if part.get_filename() == 'selection.png' else 'source.png'
                uploads[filename] = part.get_payload(decode=True)
                self.respond(json.dumps(dict(ok=True, file=filename, fileName=filename,
                    mediaType='image', url='/uploads/' + filename)).encode())
            elif self.path == '/api/image-edit/stream':
                requests.append(json.loads(data))
                self.respond(b'data: {"done":true,"url":"/uploads/source.png","width":640,"height":360}\n\n', 'text/event-stream')
            else:
                self.send_error(404)

        def do_DELETE(self):
            self.respond(b'{}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', uploads, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def stroke(page, canvas, start, end, touch):
    box = canvas.bounding_box()
    x0, y0 = box['x'] + start[0] * box['width'], box['y'] + start[1] * box['height']
    x1, y1 = box['x'] + end[0] * box['width'], box['y'] + end[1] * box['height']
    if touch:
        # Browser-native touch input, not direct manipulation of canvas state.
        cdp = page.context.new_cdp_session(page)
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x0, 'y': y0, 'id': 1}]})
        for step in range(1, 11):
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [
                {'x': x0 + (x1 - x0)*step/10, 'y': y0 + (y1 - y0)*step/10, 'id': 1}]})
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        cdp.detach()
    else:
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()


def exercise(browser, url, uploads, requests, mobile, out):
    name = 'mobile' if mobile else 'desktop'
    context = browser.new_context(viewport={'width': 390 if mobile else 1280, 'height': 844 if mobile else 900},
                                  is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    page.locator('#file-input').set_input_files({'name': 'source.png', 'mimeType': 'image/png', 'buffer': uploads['source.png']})
    page.get_by_role('button', name='Select area', exact=True).click()
    dialog = page.get_by_role('dialog', name='Select image area to edit')
    canvas = dialog.locator('canvas.ts-mask-canvas')
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    dialog.get_by_role('button', name='Use selection').click()
    expect(dialog.get_by_role('status')).to_contain_text('Paint an area')
    stroke(page, canvas, (.25, .5), (.75, .5), mobile)
    painted_raster = canvas.evaluate('(canvas) => canvas.toDataURL()')
    dialog.get_by_role('button', name='Erase', exact=True).click()
    stroke(page, canvas, (.5, .48), (.5, .52), mobile)
    erased_raster = canvas.evaluate('(canvas) => canvas.toDataURL()')
    assert erased_raster != painted_raster, 'Erase must change the selected raster'
    dialog.get_by_role('button', name='Undo', exact=True).click()
    assert canvas.evaluate('(canvas) => canvas.toDataURL()') == painted_raster, 'Undo must restore every mask sample exactly'
    dialog.get_by_role('button', name='Redo', exact=True).click()
    assert canvas.evaluate('(canvas) => canvas.toDataURL()') == erased_raster, 'Redo must restore every mask sample exactly'
    dialog.get_by_role('button', name='Undo', exact=True).click()
    assert canvas.evaluate('(canvas) => canvas.toDataURL()') == painted_raster
    zoom = dialog.get_by_role('slider', name='Zoom (%)')
    zoom.focus()
    zoom.press('Home')
    for _ in range(7):
        zoom.press('ArrowRight')
    expect(zoom).to_have_value('200')
    dialog.get_by_role('button', name='Pan', exact=True).click()
    stroke(page, canvas, (.25, .25), (.18, .2), mobile)
    # Panning must not paint; the exported raster below remains the same stripe.
    zoom.press('Home')
    for _ in range(3):
        zoom.press('ArrowRight')
    page.screenshot(path=str(out / (name + '-editor.png')))
    with page.expect_response(lambda response: response.url.endswith('/api/upload')) as upload_response:
        dialog.get_by_role('button', name='Use selection').click()
    assert upload_response.value.ok
    expect(dialog).not_to_be_visible()
    expect(page.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
    mask = np.asarray(Image.open(io.BytesIO(uploads['selection.png'])).convert('RGBA'))
    assert mask.shape == (360, 640, 4), mask.shape
    assert np.all(mask[:, :, 0] == mask[:, :, 1]) and np.all(mask[:, :, 1] == mask[:, :, 2])
    assert np.all(mask[:, :, 3] == 255)
    assert mask[180, 320, 0] == 255 and mask[20, 20, 0] == 0
    Image.fromarray(mask).save(out / (name + '-mask.png'))
    page.locator('#message-input').fill('Change only the selected stripe.')
    page.locator('#btn-send').click()
    expect(page.get_by_role('button', name='Edit again', exact=True)).to_be_visible()
    request = requests[-1]
    assert request['imagePaths'] == ['source.png'], request
    assert request['maskPath'] == 'selection.png' and request['maskMode'] == 'grayscale', request
    page.get_by_role('button', name='Show original', exact=True).click()
    expect(page.get_by_role('button', name='Show edited', exact=True)).to_have_attribute('aria-pressed', 'true')
    page.get_by_role('button', name='Edit again', exact=True).click()
    expect(page.locator('#message-input')).to_have_value('Change only the selected stripe.')
    page.get_by_role('button', name='Edit selection', exact=True).click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    dialog.get_by_role('button', name='Invert', exact=True).click()
    with page.expect_response(lambda response: response.url.endswith('/api/upload')) as upload_response:
        dialog.get_by_role('button', name='Use selection').click()
    assert upload_response.value.ok
    expect(dialog).not_to_be_visible()
    inverted = np.asarray(Image.open(io.BytesIO(uploads['selection.png'])).convert('RGBA'))
    assert inverted[180, 320, 0] == 0 and inverted[20, 20, 0] == 255
    page.get_by_role('button', name='Edit selection', exact=True).click()
    dialog.get_by_role('button', name='Remove selection', exact=True).click()
    expect(page.get_by_role('button', name='Select area', exact=True)).to_be_visible()
    assert not errors, errors
    context.close()
    return {'surface': name, 'status': 'passed', 'native_dimensions': [640, 360], 'input': 'touch' if mobile else 'mouse',
            'checks': ['empty guard', 'paint', 'erase', 'pixel-identical undo/redo', 'zoom/pan', 'opaque grayscale export', 'source coordinate alignment',
                       'mask excluded from image references', 'chat request', 'compare', 'edit again', 'reopen/invert', 'remove']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/browser')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with fixture() as (url, uploads, requests), sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
        try:
            for mobile in (False, True):
                results.append(exercise(browser, url, uploads, requests, mobile, args.out))
        finally:
            browser.close()
    report = {'runs': results, 'limitations': 'Loopback fixture with mocked inference; mobile is Chromium touch emulation, not MAUI/iOS/Android device validation.'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
