#!/usr/bin/env python3
"""Check real TensorAgent area selection without loading an inference model.

Launch eng/validation/TensorAgentHost without --weights first. This checks real
uploads, the shared editor, mask dimensions, and desktop/touch visibility. It
does not validate a native MAUI picker, iOS WebKit, or inference quality.
"""
import argparse
import io
import json
from pathlib import Path
import time

from PIL import Image
from playwright.sync_api import expect, sync_playwright
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
    client = requests.Session()
    client.headers['Cookie'] = connection['cookie']
    models = client.get(connection['baseUrl'] + '/api/models', timeout=15)
    models.raise_for_status()
    assert not models.json().get('loaded'), 'Run this check with no loaded model.'
    with Image.open(args.image) as source:
        source_size = source.size
    report = {'status': 'failed', 'surfaces': [], 'limitations': [
        'Real AgentAppHost in Chromium; native MAUI, photo pickers and iOS WebKit are not exercised.',
        'No inference model loaded; this validates selection preparation and visibility, not generated edits.',
        'Single interaction timing per viewport; no controlled performance comparison.',
        'Touch checks pause 0.5 seconds after the stroke; immediate synthetic follow-up taps did not generate a click in headless Chromium.',
    ]}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
            try:
                for name, width, touch in [('desktop', 1280, False), ('touch-390', 390, True), ('touch-320', 320, True)]:
                    errors, chat_requests = [], []
                    context = browser.new_context(viewport={'width': width, 'height': 900 if not touch else 844},
                                                  has_touch=touch, is_mobile=touch)
                    page = context.new_page()
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.on('request', lambda request: chat_requests.append(request.url)
                            if request.url.endswith('/api/chat') else None)
                    page.set_default_timeout(30000)
                    page.goto(connection['entryUrl'])
                    page.evaluate("""() => {
                      window.maskValidationEvents = [];
                      ['pointerdown', 'pointerup', 'click'].forEach(type => document.addEventListener(type, e => {
                        window.maskValidationEvents.push({type, target: e.target.className, label: e.target.tagName === 'BUTTON' ? e.target.textContent : '', x: e.clientX, y: e.clientY});
                      }, true));
                    }""")
                    expect(page.locator('#send')).to_be_disabled()
                    page.locator('#file-input').set_input_files(str(args.image.resolve()))
                    select = page.locator('#chips .mask-select').first
                    expect(select).to_have_text('Select area')
                    expect(select).to_be_visible()
                    expect(select).to_be_enabled()
                    box = select.bounding_box()
                    assert box['height'] >= 44, f'{name}: selection target too small'
                    assert 0 <= box['x'] and box['x'] + box['width'] <= width, f'{name}: selection control outside viewport'
                    assert page.locator('#chips').evaluate('(e) => e.scrollWidth <= e.clientWidth'), f'{name}: attachment overflows'
                    page.screenshot(path=str(args.out / f'{name}-button.png'))
                    start = time.perf_counter()
                    if touch:
                        cdp = context.new_cdp_session(page)
                        cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': box['x'] + box['width']/2, 'y': box['y'] + box['height']/2, 'id': 0}]})
                        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
                    else:
                        select.click()
                    dialog = page.get_by_role('dialog', name='Select image area to edit')
                    expect(dialog.get_by_role('button', name='Use selection', exact=True)).to_be_enabled()
                    canvas = dialog.locator('.ts-mask-canvas')
                    bounds = canvas.bounding_box()
                    x0, x1 = bounds['x'] + bounds['width'] * .4, bounds['x'] + bounds['width'] * .6
                    y = bounds['y'] + bounds['height'] * .5
                    if touch:
                        canvas.evaluate("e => e.addEventListener('pointerup', () => e.dataset.validationPointerEnded = 'true', {once: true})")
                        cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x0, 'y': y, 'id': 0}]})
                        for step in range(1, 6):
                            cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [{'x': x0 + (x1-x0)*step/5, 'y': y, 'id': 0}]})
                        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
                        expect(canvas).to_have_attribute('data-validation-pointer-ended', 'true')
                        # Model the pause between painting and choosing a toolbar action.
                        # Immediate synthetic follow-up taps did not generate a click.
                        time.sleep(.5)
                    else:
                        page.mouse.move(x0, y)
                        page.mouse.down()
                        page.mouse.move(x1, y, steps=5)
                        page.mouse.up()
                    page.screenshot(path=str(args.out / f'{name}-editor.png'))
                    try:
                        with page.expect_response(lambda response: response.url.endswith('/api/upload'), timeout=10000) as uploaded:
                            save = dialog.get_by_role('button', name='Use selection', exact=True)
                            if touch:
                                save_box = save.bounding_box()
                                save_x, save_y = save_box['x'] + save_box['width']/2, save_box['y'] + save_box['height']/2
                                cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': save_x, 'y': save_y, 'id': 0}]})
                                cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
                            else:
                                save.click()
                    except Exception:
                        page.screenshot(path=str(args.out / f'{name}-save-failed.png'))
                        report['save_failure'] = page.evaluate("""() => ({
                          events: window.maskValidationEvents,
                          status: document.querySelector('.ts-mask-status')?.textContent,
                          saveBounds: document.querySelector('.ts-mask-primary')?.getBoundingClientRect().toJSON()
                        })""")
                        raise
                    response = uploaded.value
                    if touch:
                        cdp.detach()
                    assert response.ok, f'{name}: mask upload failed ({response.status})'
                    data = response.json()
                    item = data['files'][0] if 'files' in data else data
                    assert item.get('ok') and item.get('file'), f'{name}: upload did not return a file'
                    saved = page.locator('#chips .mask-select').first
                    expect(saved).to_have_text('Selection saved · Adjust')
                    expect(saved).to_be_enabled()
                    saved_box = saved.bounding_box()
                    assert 0 <= saved_box['x'] and saved_box['x'] + saved_box['width'] <= width, f'{name}: saved selection control outside viewport'
                    assert page.locator('#chips').evaluate('(e) => e.scrollWidth <= e.clientWidth'), f'{name}: saved selection overflows'
                    expect(page.locator('.image-selection-hint')).to_contain_text('Load Qwen-Image 2.1')
                    expect(page.locator('.image-selection-hint').get_by_role('button', name='Open Models')).to_be_enabled()
                    expect(page.locator('#send')).to_be_disabled()
                    elapsed = time.perf_counter() - start
                    mask_response = client.get(connection['baseUrl'] + '/uploads/' + item['file'], timeout=15)
                    mask_response.raise_for_status()
                    (args.out / f'{name}-mask.png').write_bytes(mask_response.content)
                    with Image.open(io.BytesIO(mask_response.content)) as mask:
                        assert mask.size == source_size, f'{name}: mask/source dimension mismatch'
                        assert mask.convert('L').getextrema() == (0, 255), f'{name}: painted and protected pixels missing'
                    assert not errors, f'{name}: page errors: {errors}'
                    assert not chat_requests, f'{name}: preparing selection submitted a chat turn'
                    page.screenshot(path=str(args.out / f'{name}-saved.png'))
                    report['surfaces'].append({'name': name, 'status': 'passed', 'selection_target': box,
                                               'mask_size': source_size, 'prepare_seconds': elapsed})
                    context.close()
                report['status'] = 'passed'
            finally:
                browser.close()
    finally:
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
