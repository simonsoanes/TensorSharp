#!/usr/bin/env python3
"""Run a real TensorAgent image edit through its desktop/mobile WebUI.

Launch eng/validation/TensorAgentHost with a Qwen-Image 2.1 model first, then pass
its private connection.json. No inference or HTTP responses are mocked. Requires
Playwright, Pillow, numpy and requests. Mobile mode emulates touch in Chromium;
it does not validate a physical iPhone or its Metal backend.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

from PIL import Image, ImageOps
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
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--target-index', type=int, default=0,
                        help='Attachment position of --image before selection promotes it to source.')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/tensoragent-live')
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    parser.add_argument('--prompt', default='Change the red cube to a blue cube. Keep the same lighting and shape.')
    parser.add_argument('--mobile', action='store_true')
    parser.add_argument('--native-cdp', help='Loopback WebView2 debugging endpoint of an isolated Windows app.')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if not 0 <= args.target_index <= len(args.reference):
        parser.error('--target-index must be between 0 and the number of references')
    if args.native_cdp and args.mobile:
        parser.error('--native-cdp uses the actual app viewport, not mobile emulation')
    args.out.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
    client = requests.Session()
    client.headers['Cookie'] = connection['cookie']
    errors, report = [], {'status': 'failed', 'surface': 'native Windows WebView2' if args.native_cdp else 'mobile touch' if args.mobile else 'desktop mouse',
        'model': connection.get('model'), 'backend': connection.get('backend'),
        'target_attachment_index': args.target_index, 'reference_image_count': len(args.reference),
        'limitations': ['Single real local model and GPU, no controlled repeated timing or general quality score.',
            'Chromium WebUI uses the real AgentAppHost; native MAUI picker, iOS WebKit and Metal are not validated.',
            'TensorAgent requests its normal 1024x1024 area and default40steps with region cropping enabled; result retains source dimensions.']}
    if args.native_cdp:
        report['limitations'][1] = 'Actual Windows MAUI WebView2; file-input uploads bypass the native file dialog. Physical mobile devices and Metal are not validated.'
    with sync_playwright() as playwright:
        if args.native_cdp:
            browser = playwright.chromium.connect_over_cdp(args.native_cdp)
            pages = [page for context in browser.contexts for page in context.pages
                     if page.url.startswith(connection['baseUrl'])]
            assert len(pages) == 1, 'Expected one native WebView connected to the validation host'
            page = pages[0]
            context = page.context
        else:
            browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
            context = browser.new_context(viewport={'width': 390 if args.mobile else 1280, 'height': 844 if args.mobile else 900},
                has_touch=args.mobile, is_mobile=args.mobile)
            cookie_name, cookie_value = connection['cookie'].split('=', 1)
            context.add_cookies([{'name': cookie_name, 'value': cookie_value, 'url': connection['baseUrl']}])
            page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.set_default_timeout(60000)
        try:
            if args.native_cdp:
                page.evaluate('window.TensorAgent.refreshModel()')
                expect(page.locator('#text')).to_have_attribute('placeholder', 'Describe a picture… or attach a photo and say what to change')
            else:
                page.goto(connection['baseUrl'])
            attached_images = list(args.reference)
            attached_images.insert(args.target_index, args.image)
            with page.expect_response(lambda response: response.url.endswith('/api/upload')) as source_upload:
                page.locator('#file-input').set_input_files([str(path.resolve()) for path in attached_images])
            uploaded = source_upload.value.json()
            uploaded_files = uploaded.get('files', [uploaded])
            assert len(uploaded_files) == len(attached_images)
            target_upload = uploaded_files[args.target_index]
            expected_source = target_upload['file']
            expected_references = [item['file'] for index, item in enumerate(uploaded_files) if index != args.target_index]
            metric_source = args.image
            if target_upload.get('editUrl'):
                canonical_source = client.get(connection['baseUrl'] + target_upload['editUrl'], timeout=60)
                canonical_source.raise_for_status()
                metric_source = args.out / 'canonical-source.png'
                metric_source.write_bytes(canonical_source.content)
            with Image.open(metric_source) as source_image:
                width, height = ImageOps.exif_transpose(source_image).size
            report['source'] = {'path': str(args.image.resolve()),
                'sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
                'width': width, 'height': height, 'edit_url': target_upload.get('editUrl'),
                'metric_source_sha256': hashlib.sha256(metric_source.read_bytes()).hexdigest()}
            controls = page.locator('#chips .mask-select')
            expect(controls).to_have_count(len(attached_images))
            if args.native_cdp:
                # Exercise the shipped button's keyboard activation in the real WebView.
                controls.nth(args.target_index).focus()
                page.keyboard.press('Enter')
            else:
                controls.nth(args.target_index).click()
            dialog = page.get_by_role('dialog', name='Select image area to edit')
            canvas = dialog.locator('.ts-mask-canvas')
            expect(dialog.get_by_role('button', name='Use selection', exact=True)).to_be_enabled()
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
            expect(page.locator('#chips .mask-select').first).to_have_text('Selection saved · Adjust')
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
            assert sent['stillImagePaths'] == [expected_source, *expected_references]
            assert sent['imagePaths'] == sent['stillImagePaths']
            assert sent['maskPath'] not in sent['imagePaths']
            assert all('maskPath' not in attachment for attachment in sent['attachments'][1:])
            mask = client.get(connection['baseUrl'] + '/uploads/' + sent['maskPath'], timeout=60)
            mask.raise_for_status(); (args.out / 'painted-mask.png').write_bytes(mask.content)
            picture = client.get(connection['baseUrl'] + result, timeout=60)
            picture.raise_for_status(); (args.out / 'result.png').write_bytes(picture.content)
            report['pixels'] = benchmark.measure_pixels(metric_source, io.BytesIO(picture.content), io.BytesIO(mask.content))
            report['preview_preservation'] = [benchmark.measure_pixels(metric_source,
                io.BytesIO(base64.b64decode(frame['preview'].split(',', 1)[1])), io.BytesIO(mask.content))
                for frame in frames if frame.get('preview')]
            assert report['pixels']['changed_editable_pixels'] > 0
            page.wait_for_function('!window.TensorAgent.isGenerating()', timeout=args.timeout*1000)
            page.screenshot(path=str(args.out / 'result-ui.png'))
            page.get_by_role('button', name='Compare original', exact=True).last.click()
            page.get_by_role('button', name='Show result', exact=True).last.click()
            page.get_by_role('button', name='Edit again', exact=True).last.click()
            expect(page.locator('#text')).to_have_value(args.prompt)
            expect(page.locator('#chips .mask-select').first).to_have_text('Selection saved · Adjust')
            expect(page.locator('#chips .mask-select')).to_have_count(len(attached_images))
            assert not errors, errors
            report['status'] = 'passed'
        except Exception as error:
            report['error'] = str(error)
            page.screenshot(path=str(args.out / 'failure.png'))
        finally:
            report['browser_errors'] = errors
            (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            if not args.native_cdp:
                browser.close()
    print(json.dumps({key: report[key] for key in ('status', 'surface', 'seconds', 'pixels', 'error') if key in report}))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
