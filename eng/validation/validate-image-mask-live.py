#!/usr/bin/env python3
"""Drive a real Server.Host masked edit through its shipped Chat page.

Uses isolated headless Chromium, a real image upload, selection upload and SSE
inference. Run the Qwen server at a practical default sampling resolution, e.g.
TS_QWEN_IMAGE_WIDTH=512 and TS_QWEN_IMAGE_HEIGHT=512. No endpoint is mocked.
Also checks multipart no-op RGBA preservation and invalid-mask HTTP responses.
Use --reference reference.png --target-index 1 to upload the target after a
reference, then verify that saving its selection promotes it to the edit source.
HEIC/HEIF validation needs pillow-heif for source dimensions; exact pixel checks
use the server's full-resolution editUrl PNG to avoid decoder rounding differences.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import requests
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def picture_info(path):
    if path.suffix.lower() in ('.heic', '.heif'):
        # Optional validation dependency only; production HEIC decoding belongs
        # to the server's platform codec, not Pillow.
        import pillow_heif
        pillow_heif.register_heif_opener()
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'width': width, 'height': height}


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
    parser.add_argument('--reference', type=Path, action='append', default=[],
                        help='Additional reference photo; repeat to preserve their specified order.')
    parser.add_argument('--target-index', type=int, default=0,
                        help='Zero-based attachment position at which --image is inserted among --reference photos (default: 0).')
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/live-web')
    args = parser.parse_args()
    if not 0 <= args.target_index <= len(args.reference):
        parser.error('--target-index must be between 0 and the number of --reference photos')
    attachment_paths = list(args.reference)
    attachment_paths.insert(args.target_index, args.image)
    for path in attachment_paths:
        if not path.is_file():
            parser.error(f'Image file does not exist: {path}')
    pictures = [picture_info(path) for path in attachment_paths]
    target_picture = pictures[args.target_index]
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
            upload_started = time.perf_counter()
            with page.expect_response(lambda response: response.url.endswith('/api/upload')) as uploaded_response:
                page.locator('#file-input').set_input_files([str(path.resolve()) for path in attachment_paths])
            upload_seconds = time.perf_counter() - upload_started
            assert uploaded_response.value.ok
            uploaded_body = uploaded_response.value.json()
            uploaded = uploaded_body.get('files', [uploaded_body])
            assert len(uploaded) == len(attachment_paths), uploaded_body
            assert all(item.get('ok') and item.get('file') and item.get('mediaType') == 'image' for item in uploaded), uploaded_body
            uploaded_paths = [item['file'] for item in uploaded]
            # Different HEIC decoder libraries may round RGB differently. The
            # server's full-resolution PNG is the same decoded canvas its model
            # sees; use that authoritative reference for exact protected pixels.
            pixel_reference = args.image
            target_upload = uploaded[args.target_index]
            if target_upload.get('editUnavailableReason'):
                raise AssertionError(target_upload['editUnavailableReason'])
            if target_upload.get('editUrl'):
                decoded = requests.get(args.url + target_upload['editUrl'], timeout=60)
                decoded.raise_for_status()
                pixel_reference = args.out / 'decoded-target.png'
                pixel_reference.write_bytes(decoded.content)
                decoded_info = picture_info(pixel_reference)
                assert (decoded_info['width'], decoded_info['height']) == (target_picture['width'], target_picture['height']), decoded_info
            expected_paths = [uploaded_paths[args.target_index]] + [
                path for index, path in enumerate(uploaded_paths) if index != args.target_index]
            expect(page.locator('.image-edit-attachment')).to_have_count(len(attachment_paths))
            page.locator('.image-edit-attachment').nth(args.target_index).get_by_role('button', name='Select area', exact=True).click()
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
            actual_order = page.locator('.image-edit-attachment').evaluate_all('(nodes) => nodes.map(node => node.dataset.file)')
            assert actual_order == expected_paths, (actual_order, expected_paths)
            expect(page.locator('.image-edit-attachment').first.locator('.image-selection-role')).to_have_text('Editing target')
            page.locator('#message-input').fill('Change the red cube to a blue cube. Keep the same lighting and shape.')
            started = time.perf_counter()
            with page.expect_request('**/api/image-edit/stream') as captured:
                page.locator('#btn-send').click()
            payload = captured.value.post_data_json
            assert payload['imagePaths'] == expected_paths, (payload, expected_paths)
            assert payload.get('maskMode') == 'grayscale' and payload.get('maskPath'), payload
            assert payload['maskPath'] not in payload['imagePaths'], payload
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
            with Image.open(mask) as painted_mask:
                assert painted_mask.size == (target_picture['width'], target_picture['height']), painted_mask.size
            pixels = bench.measure_pixels(pixel_reference, output, mask)
            assert pixels['protected_pixels'] > 0, 'This validation requires some unselected pixels'
            assert pixels['changed_editable_pixels'] > 0
            page.screenshot(path=str(args.out / 'result-page.png'), full_page=True)
            page.get_by_role('button', name='Edit again', exact=True).click()
            expect(page.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
            reused_order = page.locator('.image-edit-attachment').evaluate_all('(nodes) => nodes.map(node => node.dataset.file)')
            assert reused_order == expected_paths, (reused_order, expected_paths)
            assert not errors, errors
            report = {'status': 'passed', 'seconds': seconds, 'request': payload, 'pixels': pixels,
                      'upload_seconds': upload_seconds, 'pixel_reference': picture_info(pixel_reference),
                      'target_attachment_index': args.target_index, 'target': target_picture,
                      'original_attachments': [dict(picture, uploaded_path=uploaded_paths[index],
                                                   preview_url=uploaded[index].get('previewUrl'), edit_url=uploaded[index].get('editUrl'))
                                               for index, picture in enumerate(pictures)],
                      'result_sha256': hashlib.sha256(result.content).hexdigest(),
                      'mask_sha256': hashlib.sha256(mask_response.content).hexdigest(),
                      'checks': checks + ['real browser upload/draw/upload/submit/SSE/result/reuse',
                                         'selected photo promoted before ordered reference photos',
                                         'selection geometry and protected pixels match the selected photo',
                                         'edit again preserves selected source and reference order'],
                      'limitations': 'One real Server.Host model scenario; visual quality is not a general quality score. Desktop Chromium, not native mobile. Backend is selected by the running host; this script does not assert CUDA.'}
            (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
            print(json.dumps(report, indent=2))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
