#!/usr/bin/env python3
"""Drive a real Server.Host masked edit through its shipped Chat page.

Uses isolated headless Chromium, a real image upload, selection upload and SSE
inference. Run the Qwen server at a practical default sampling resolution, e.g.
TS_QWEN_IMAGE_WIDTH=512 and TS_QWEN_IMAGE_HEIGHT=512. No endpoint is mocked.
Also checks multipart no-op RGBA preservation and invalid-mask HTTP responses.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import requests
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def contracts(url):
    # Odd geometry, partial transparency and hidden RGB must survive a no-op edit.
    y, x = np.indices((97, 161))
    source = np.stack([x % 256, y % 256, (x + y) % 256, (x * 3 + y) % 256], axis=-1).astype('uint8')
    black = np.zeros((97, 161), dtype='uint8')
    files = {'image': ('source.png', png(source), 'image/png'), 'mask': ('mask.png', png(black), 'image/png')}
    response = requests.post(url + '/api/image-edit', files=files, data={'prompt': 'No-op edit', 'width': '64', 'height': '64'}, timeout=120)
    response.raise_for_status()
    result = response.json()
    edited = requests.get(url + result['url'], timeout=30)
    edited.raise_for_status()
    output = np.asarray(Image.open(io.BytesIO(edited.content)).convert('RGBA'))
    assert np.array_equal(source, output), 'Multipart no-op changed original RGBA pixels'
    checks = ['multipart mask separation and empty-mask no-op at original odd dimensions with alpha']
    for name, mask_data, fields in [
        ('dimension mismatch', png(np.zeros((8, 8), dtype='uint8')), {}),
        ('bad mode', png(black), {'maskMode': 'guess'}),
        ('invalid feather', png(black), {'maskFeather': '-1'}),
        ('invalid crop flag', png(black), {'maskCrop': 'perhaps'}),
        ('corrupt mask', b'not an image', {}),
    ]:
        response = requests.post(url + '/api/image-edit', files={
            'image': files['image'], 'mask': ('mask.png', mask_data, 'image/png')},
            data={'prompt': 'Edit', 'width': '64', 'height': '64', **fields}, timeout=30)
        assert response.status_code == 400, (name, response.status_code, response.text)
        checks.append(name + ' rejected with HTTP 400')
    for asset in ('mask-editor.js', 'mask-editor.css'):
        response = requests.get(url + '/' + asset, timeout=15)
        response.raise_for_status()
        assert 'ts-mask' in response.text
        checks.append(asset + ' served by real host')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/live-web')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checks = contracts(args.url)
    errors = []
    spec = importlib.util.spec_from_file_location('mask_bench', Path(__file__).with_name('qwen-image21-mask-bench.py'))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 900})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(args.url)
            page.locator('#file-input').set_input_files(str(args.image.resolve()))
            page.get_by_role('button', name='Select area', exact=True).click()
            dialog = page.get_by_role('dialog', name='Select image area to edit')
            expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
            brush = dialog.get_by_role('slider', name='Brush (image pixels)')
            brush.focus()
            brush.press('End')
            canvas = dialog.locator('canvas.ts-mask-canvas').bounding_box()
            page.mouse.move(canvas['x'] + canvas['width'] * .5, canvas['y'] + canvas['height'] * .43)
            page.mouse.down()
            page.mouse.move(canvas['x'] + canvas['width'] * .5, canvas['y'] + canvas['height'] * .73, steps=12)
            page.mouse.up()
            dialog.get_by_role('checkbox').check()
            dialog.get_by_role('button', name='Use selection').click()
            expect(page.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
            page.locator('#message-input').fill('Change the red cube to a blue cube. Keep the same lighting and shape.')
            started = time.perf_counter()
            with page.expect_request('**/api/image-edit/stream') as captured:
                page.locator('#btn-send').click()
            payload = captured.value.post_data_json
            expect(page.get_by_role('button', name='Edit again', exact=True)).to_be_visible(timeout=600000)
            seconds = time.perf_counter() - started
            result_url = page.get_by_role('link', name='Download image').get_attribute('href')
            result = requests.get(args.url + result_url, timeout=30)
            result.raise_for_status()
            output = args.out / 'result.png'
            output.write_bytes(result.content)
            mask_response = requests.get(args.url + '/uploads/' + Path(payload['maskPath']).name, timeout=30)
            mask_response.raise_for_status()
            mask = args.out / 'painted-mask.png'
            mask.write_bytes(mask_response.content)
            pixels = bench.measure_pixels(args.image, output, mask)
            assert pixels['changed_editable_pixels'] > 0
            page.screenshot(path=str(args.out / 'result-page.png'), full_page=True)
            page.get_by_role('button', name='Edit again', exact=True).click()
            expect(page.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
            assert not errors, errors
            report = {'status': 'passed', 'seconds': seconds, 'request': payload, 'pixels': pixels,
                      'checks': checks + ['real browser upload/draw/upload/submit/SSE/result/reuse'],
                      'limitations': 'One real CUDA model scenario; visual quality is not a general quality score. Desktop Chromium, not native mobile.'}
            (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
            print(json.dumps(report, indent=2))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
