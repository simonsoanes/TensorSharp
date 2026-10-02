// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Plays a file the TensorAgent app serves in a real WebKit <video> (or <audio>) and reports what
// the media element saw. A byte fetch says the file is there; only WebKit says it PLAYS: its
// media loader asks for byte ranges, needs the index it can reach early, and decodes the codecs
// itself, and none of that is exercised by curl, by the unit suite, or by chat-e2e.py.
//
//   swift eng/validation/webview-media-check.swift <origin> <token> <path> [video|audio] [seconds]
//
//   origin   http://127.0.0.1:<port> from the app's "entry URL" log line (Debug builds)
//   token    the ?token= value from the same line; sent as the app's tensoragent_token cookie
//   path     e.g. /uploads/video-<id>.mp4
//
// Prints one JSON line and exits 0 when the element reached HAVE_ENOUGH_DATA and played forward,
// 1 otherwise. The page is offscreen and muted, which WebKit allows to autoplay.
import AppKit
import WebKit

let args = CommandLine.arguments
guard args.count >= 4, let origin = URL(string: args[1]) else {
    FileHandle.standardError.write("usage: webview-media-check.swift <origin> <token> <path> [video|audio] [seconds]\n".data(using: .utf8)!)
    exit(2)
}
let token = args[2]
let path = args[3]
let kind = args.count > 4 ? args[4] : "video"
let seconds = args.count > 5 ? Double(args[5]) ?? 2.0 : 2.0

final class Probe: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    let view: WKWebView
    var window: NSWindow?
    let origin: URL, token: String, path: String, kind: String, seconds: Double

    init(origin: URL, token: String, path: String, kind: String, seconds: Double) {
        self.origin = origin; self.token = token; self.path = path; self.kind = kind; self.seconds = seconds
        let config = WKWebViewConfiguration()
        config.mediaTypesRequiringUserActionForPlayback = []
        let controller = WKUserContentController()
        config.userContentController = controller
        view = WKWebView(frame: NSRect(x: 0, y: 0, width: 640, height: 400), configuration: config)
        super.init()
        controller.add(self, name: "done")
        view.navigationDelegate = self
    }

    func start() {
        // A real window, off screen: WebKit throttles media in a view that is in none.
        let w = NSWindow(contentRect: NSRect(x: -2000, y: -2000, width: 640, height: 400),
                         styleMask: [.borderless], backing: .buffered, defer: false)
        w.contentView = view
        w.orderFrontRegardless()
        window = w

        let cookie = HTTPCookie(properties: [
            .domain: origin.host ?? "127.0.0.1", .path: "/", .name: "tensoragent_token", .value: token,
        ])!
        view.configuration.websiteDataStore.httpCookieStore.setCookie(cookie) { [self] in
            let src = path.replacingOccurrences(of: "\"", with: "")
            let tag = kind == "audio" ? "audio" : "video"
            let html = """
            <!doctype html><html><body>
            <\(tag) id="m" preload="auto" muted playsinline src="\(src)"></\(tag)>
            <script>
            const m = document.getElementById('m');
            const out = (ok, why) => window.webkit.messageHandlers.done.postMessage(JSON.stringify({
              ok, why, readyState: m.readyState, networkState: m.networkState,
              duration: m.duration, currentTime: m.currentTime,
              width: m.videoWidth || 0, height: m.videoHeight || 0,
              error: m.error ? m.error.code : 0,
              audioTracks: m.audioTracks ? m.audioTracks.length : -1,
              videoTracks: m.videoTracks ? m.videoTracks.length : -1,
              audioBytes: m.webkitAudioDecodedByteCount ?? -1,
              videoBytes: m.webkitVideoDecodedByteCount ?? -1 }));
            m.addEventListener('error', () => out(false, 'error'));
            m.addEventListener('canplaythrough', () => {
              m.play().then(() => setTimeout(() => out(m.currentTime > 0, 'played'), \(Int(seconds * 1000))),
                            e => out(false, 'play() refused: ' + e));
            }, { once: true });
            setTimeout(() => out(false, 'timeout'), 30000);
            </script></body></html>
            """
            view.loadHTMLString(html, baseURL: origin)
        }
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        let text = message.body as? String ?? "{}"
        print(text)
        let ok = text.contains("\"ok\":true")
        exit(ok ? 0 : 1)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let probe = Probe(origin: origin, token: token, path: path, kind: kind, seconds: seconds)
probe.start()
app.run()
