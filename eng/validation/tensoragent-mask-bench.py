#!/usr/bin/env python3
"""Run a real TensorAgent image edit through its desktop/mobile WebUI.

Launch eng/validation/TensorAgentHost with a Qwen-Image 2.1 model first, then pass
its private connection.json. No inference or HTTP responses are mocked. Requires
Playwright, Pillow, numpy and requests. Mobile mode emulates touch in Chromium;
it does not validate a physical iPhone or its Metal backend.
"""
import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path
import time

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import requests

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mask_bench', Path(__file__).with_name('qwen-image21-mask-bench.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/tensoragent-live')
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    parser.add_argument('--prompt', default='Change the red cube to a blue cube. Keep the same lighting and shape.')
    parser.add_argument('--mobile', action='store_true')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
    client = requests.Session()
    client.headers['Cookie'] = connection['cookie']
    errors, report = [], {'status': 'failed', 'surface': 'mobile touch' if args.mobile else 'desktop mouse',
        'model': connection.get('model'), 'backend': connection.get('backend'),
        'limitations': ['Single real local model and GPU, no controlled repeated timing or general quality score.',
            'Chromium WebUI uses the real AgentAppHost; native MAUI picker, iOS WebKit and Metal are not validated.',
            'TensorAgent requests its normal 1024x1024 area and default40steps with region cropping enabled; result retains source dimensions.']}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
        context = browser.new_context(viewport={'width': 390 if args.mobile else 1280, 'height': 844 if args.mobile else 900},
            has_touch=args.mobile, is_mobile=args.mobile)
        cookie_name, cookie_value = connection['cookie'].split('=', 1)
        context.add_cookies([{'name': cookie_name, 'value': cookie_value, 'url': connection['baseUrl']}])
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_default_timeout(60000)
        try:
            page.goto(connection['baseUrl'])
            page.locator('#file-input').set_input_files(str(args.image.resolve()))
            page.get_by_role('button', name='Select area', exact=True).click()
            dialog = page.get_by_role('dialog', name='Select image area to edit')
            canvas = dialog.locator('.ts-mask-canvas')
            expect(dialog.get_by_role('button', name='Use selection', exact=True)).to_be_enabled()
            width, height = Image.open(args.image).size
            brush = min(256, int(min(width, height) * .50))
            dialog.get_by_label('Brush (image pixels)').fill(str(brush))
            box = canvas.bounding_box()
            x0, x1 = box['x'] + box['width'] * .38, box['x'] + box['width'] * .62
            y = box['y'] + box['height'] * .5
            if args.mobile:
                cdp = context.new_cdp_session(page)
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x0, 'y': y, 'id': 1}]})
                for step in range(1, 11):
                    cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [{'x': x0+(x1-x0)*step/10, 'y': y, 'id': 1}]})
                cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
                cdp.detach()
            else:
                page.mouse.move(x0, y); page.mouse.down(); page.mouse.move(x1, y, steps=10); page.mouse.up()
            dialog.get_by_text('Soften inside edge (pixels)', exact=False).locator('input').fill('8')
            dialog.get_by_label('Process selected region only', exact=False).check()
            page.screenshot(path=str(args.out / 'selection.png'))
            dialog.get_by_role('button', name='Use selection', exact=True).click()
            expect(page.get_by_role('button', name='Selection saved · Adjust', exact=True)).to_be_enabled()
            page.locator('#text').fill(args.prompt)
            started = time.perf_counter()
            with page.expect_response(lambda response: response.url.endswith('/api/chat'), timeout=args.timeout*1000) as response_info:
                page.locator('#send').click()
            response = response_info.value
            report['http_status'] = response.status
            report['request'] = response.request.post_data_json
            transcript = response.body().decode('utf-8')
            report['seconds'] = time.perf_counter() - started
            frames = [json.loads(line[6:]) for line in transcript.splitlines() if line.startswith('data: ')]
            report['frames'] = [{key: value for key, value in frame.items() if key != 'preview'} for frame in frames]
            if response.status != 200 or any(frame.get('error') for frame in frames):
                raise AssertionError('Image turn failed: ' + str(report['frames']))
            result = next(frame['imageUrl'] for frame in frames if frame.get('imageUrl'))
            sent = report['request']['messages'][-1]
            assert len(sent['stillImagePaths']) == 1 and sent['maskPath'] not in sent['imagePaths']
            mask = client.get(connection['baseUrl'] + '/uploads/' + sent['maskPath'], timeout=60)
            mask.raise_for_status(); (args.out / 'painted-mask.png').write_bytes(mask.content)
            picture = client.get(connection['baseUrl'] + result, timeout=60)
            picture.raise_for_status(); (args.out / 'result.png').write_bytes(picture.content)
            report['pixels'] = benchmark.measure_pixels(args.image, io.BytesIO(picture.content), io.BytesIO(mask.content))
            report['preview_preservation'] = [benchmark.measure_pixels(args.image,
                io.BytesIO(base64.b64decode(frame['preview'].split(',', 1)[1])), io.BytesIO(mask.content))
                for frame in frames if frame.get('preview')]
            assert report['pixels']['changed_editable_pixels'] > 0
            page.wait_for_function('!window.TensorAgent.isGenerating()', timeout=args.timeout*1000)
            page.screenshot(path=str(args.out / 'result-ui.png'))
            page.get_by_role('button', name='Compare original', exact=True).last.click()
            page.get_by_role('button', name='Show result', exact=True).last.click()
            page.get_by_role('button', name='Edit again', exact=True).last.click()
            expect(page.locator('#text')).to_have_value(args.prompt)
            expect(page.get_by_role('button', name='Selection saved · Adjust', exact=True)).to_be_enabled()
            assert not errors, errors
            report['status'] = 'passed'
        except Exception as error:
            report['error'] = str(error)
            page.screenshot(path=str(args.out / 'failure.png'))
        finally:
            report['browser_errors'] = errors
            (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            browser.close()
    print(json.dumps({key: report[key] for key in ('status', 'surface', 'seconds', 'pixels', 'error') if key in report}))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
