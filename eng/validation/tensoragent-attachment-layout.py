#!/usr/bin/env python3
"""Check actual TensorAgent HTML/JS attachment overflow in software Chromium.

API/model responses and editor completion are mocked. This validates reachable
controls and emitted chat attachments, not native pickers, inference, or WebKit.
"""
import argparse
import base64
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aE6kAAAAASUVORK5CYII=")


def exercise(browser, width, height, output):
    context = browser.new_context(viewport={"width": width, "height": height}, is_mobile=True, has_touch=True)
    page = context.new_page()
    page.set_default_timeout(5000)
    errors, sent, ready = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    html = (ROOT / "TensorAgent/src/TensorAgent.Maui/wwwroot/index.html").read_text(encoding="utf-8")
    html = html.replace("</body>", '<script src="/tensoragent.js"></script></body>')

    def route(request):
        path = urlparse(request.request.url).path
        if path == "/":
            request.fulfill(body=html, content_type="text/html; charset=utf-8")
            return
        if path == "/tensoragent.js":
            request.fulfill(body=(ROOT / "TensorAgent/src/TensorAgent.Core/WebUi/tensoragent.js").read_bytes(),
                            content_type="text/javascript")
            return
        if path.startswith("/uploads/"):
            request.fulfill(body=PNG, content_type="image/png")
            return
        if path == "/api/chat":
            sent.append(request.request.post_data_json)
            request.fulfill(body='data: {"done":true,"sessionId":"fixture","tokenCount":0}\n\n',
                            content_type="text/event-stream")
            return
        data = {"ok": True}
        if path == "/api/models":
            data = {"loaded": "Qwen image layout fixture", "architecture": "qwen_image"}
        elif path == "/api/agent/engine":
            data = {"model": {"loaded": True, "name": "Qwen image layout fixture"}}
        elif path == "/api/agent/conversations":
            data = {"conversations": []}
        elif path == "/api/agent/launch":
            data = {"cold": True}
        elif path == "/api/sessions":
            data = {"sessionId": "fixture", "conversationId": "fixture", "messages": []}
        elif path == "/api/agent/share/claim":
            data = {"ok": True, "shares": []}
        elif path == "/api/agent/events" and request.request.post_data_json.get("type") == "ready":
            ready.append(True)
        request.fulfill(json=data)

    page.route("**/*", route)
    result = {"viewport": {"width": width, "height": height}, "status": "failed"}
    name = f"{width}x{height}"
    try:
        page.goto("http://tensoragent-layout.test/")
        page.wait_for_function("window.TensorAgent && window.TensorAgent.hasModel()")
        page.wait_for_timeout(200)
        assert ready, "App startup did not finish"
        page.evaluate("""() => {
          window.__layoutEditorSources = [];
          window.TensorSharpMaskEditor = {open: (options) => {
            window.__layoutEditorSources.push(options.sourceUrl); return Promise.resolve(null);
          }};
          for (let i=1; i<=8; i++) window.TensorAgent.addAttachment({ok:true,
            file:'photo-'+i+'.png', fileName:'Photo '+i+'.png', mediaType:'image'});
        }""")
        expect(page.locator("#chips .mask-select")).to_have_count(8)
        page.locator("#text").fill("Edit the selected photo")
        geometry = page.evaluate("""() => {
          const box = id => { const e=document.getElementById(id), r=e.getBoundingClientRect();
            return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,
              scrollHeight:e.scrollHeight,clientHeight:e.clientHeight,
              overflowY:getComputedStyle(e).overflowY}; };
          return {send:box('send'),text:box('text'),chips:box('chips'),viewport:visualViewport.height};
        }""")
        result["geometry"] = geometry
        page.screenshot(path=str(output / f"{name}-initial.png"))
        for key in ("send", "text"):
            box = geometry[key]
            assert 0 <= box["y"] and box["bottom"] <= geometry["viewport"], f"{key} is outside the visible viewport"
        assert geometry["chips"]["y"] >= 0, "Attachment list starts above the viewport; earlier photos cannot be reached"
        chips = page.locator("#chips")
        if geometry["chips"]["scrollHeight"] > geometry["chips"]["clientHeight"]:
            assert geometry["chips"]["overflowY"] in ("auto", "scroll"), "Attachment list cannot be scrolled"
            chips.hover()
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(150)
        last = page.locator("#chips .mask-select").last
        expect(last).to_be_in_viewport(ratio=1)
        bounds = last.bounding_box()
        assert bounds["height"] >= 44, "Selection target is too short for touch"
        assert 0 <= bounds["x"] and bounds["x"] + bounds["width"] <= width, "Selection target overflows horizontally"
        page.screenshot(path=str(output / f"{name}-last-photo.png"))
        last.tap()
        expect(page.locator("#send")).to_be_enabled()
        assert page.evaluate("window.__layoutEditorSources") == ["/uploads/photo-8.png"]
        if geometry["chips"]["scrollHeight"] > geometry["chips"]["clientHeight"]:
            chips.hover()
            page.mouse.wheel(0, -5000)
            page.wait_for_timeout(150)
        first = page.locator("#chips .mask-select").first
        expect(first).to_be_in_viewport(ratio=1)
        first.focus()
        page.keyboard.press("Enter")
        assert page.evaluate("window.__layoutEditorSources") == ["/uploads/photo-8.png", "/uploads/photo-1.png"]
        page.screenshot(path=str(output / f"{name}-first-photo.png"))
        page.locator("#send").tap()
        page.wait_for_timeout(150)
        assert len(sent) == 1, "Send did not reach /api/chat"
        photos = sent[0]["messages"][-1]["stillImagePaths"]
        assert photos == [f"photo-{i}.png" for i in range(1, 9)]
        assert not errors, errors
        result.update(status="passed", touch_editor_source="photo-8.png", keyboard_editor_source="photo-1.png",
                      selection_bounds=bounds, sent_photos=photos)
    except Exception as error:
        result["error"] = str(error)
        page.screenshot(path=str(output / f"{name}-failure.png"))
    finally:
        result["page_errors"] = errors
        context.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="C:/Program Files/Google/Chrome/Application/chrome.exe")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True, args=["--disable-gpu"])
        results = [exercise(browser, width, height, args.out) for width, height in [(390, 844), (320, 844), (390, 420)]]
        browser.close()
    report = {"passed": all(r["status"] == "passed" for r in results), "cases": results,
              "limitations": ["Software Chromium mobile viewport and reduced-height keyboard approximation; no physical mobile/WebKit coverage.",
                              "Uses shipped HTML/JS; boot/chat/editor results are mocked; no model inference."]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
