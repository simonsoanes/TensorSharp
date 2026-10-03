// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// ============================================================================
// What Safari runs inside the page when the user shares it into TensorAgent.
//
// This file is the single biggest quality difference in the whole feature, and the
// reason is simple: without it a browser hands a share extension a URL and nothing
// else, so a model running entirely on the phone -- with no way to fetch that URL --
// can say nothing about the page at all. With it, the extension receives the article's
// text, its title and whatever the reader had selected, and the answer is about the
// thing the user was actually looking at.
//
// Only Safari runs it. Chrome, Edge and Firefox construct their share item eagerly
// from the URL with no page access, so they deliver public.url and the extension falls
// back to sharing the link (see ShareItemReader). That is not a bug to work around; it
// is what those browsers vend.
//
// The contract, which is unforgiving:
//   * a GLOBAL named exactly ExtensionPreprocessingJS, holding an INSTANCE,
//   * whose prototype has run(arguments), which must call arguments.completionFunction
//     with a dictionary,
//   * the file must be a bundle RESOURCE of the extension, and
//   * NSExtensionJavaScriptPreprocessingFile names it WITHOUT the .js extension.
// Get any of those wrong and nothing happens, with no error anywhere.
//
// It must also be defensive to the point of paranoia: it runs inside somebody else's
// page, where document.title can be missing, getSelection can throw inside a
// cross-origin frame, and an exception here means the share silently produces nothing.
// ============================================================================

var TensorAgentPreprocessor = function () {};

TensorAgentPreprocessor.prototype = {
  run: function (args) {
    var payload = { title: '', URL: '', selection: '', selectionTruncated: false, text: '', truncated: false, metadataTruncated: false };
    try {
      var title = shortenMiddle(String(document.title || ''), 1024);
      payload.title = title.text;
      payload.metadataTruncated = payload.metadataTruncated || title.truncated;
    } catch (e) {}
    try {
      var address = String(document.baseURI || (document.location && document.location.href) || '');
      if (address.length > 8192) {
        address = address.slice(0, 8191) + '…';
        payload.metadataTruncated = true;
      }
      payload.URL = address;
    } catch (e) {}
    try {
      var selected = readSelection();
      payload.selection = selected.text;
      payload.selectionTruncated = selected.truncated;
    } catch (e) {}
    try {
      var body = readArticle();
      payload.text = body.text;
      payload.truncated = body.truncated;
    } catch (e) {}
    args.completionFunction(payload);
  },

  // Called when the extension finishes. Nothing to undo -- this reads the page and
  // changes nothing in it -- but the method has to exist or Safari logs a complaint.
  finalize: function () {},
};

function readSelection() {
  var selection = window.getSelection();
  return shortenMiddle(selection ? String(selection.toString() || '') : '', 60000);
}

function shortenMiddle(text, limit) {
  if (text.length <= limit) return { text: text, truncated: false };
  // English, as the string tables are out of reach in here: ShareItemReader looks for this
  // exact marker (ScriptMiddleShortenedMarker) to put the interface language's in its place.
  var marker = '\n\n… [middle shortened by TensorAgent] …\n\n';
  var room = Math.max(0, limit - marker.length);
  var head = Math.ceil(room / 2);
  return {
    text: text.slice(0, head) + marker + text.slice(text.length - (room - head)),
    truncated: true,
  };
}

// The article, not the page.
//
// A rendered news page is mostly navigation, related-story rails, cookie prose and
// footer links, and every character of that is a token the phone spends before it
// reaches a sentence anyone wrote. This is a deliberately small readability: find the
// element that looks most like the article, take its innerText, and fall back to the
// body when nothing stands out. innerText rather than textContent because innerText is
// what is VISIBLE -- it already drops script, style and hidden elements, and it inserts
// the line breaks the layout implies.
function readArticle() {
  // 60,000 characters is far more than the app will put in front of the model
  // (ShareCompositionOptions caps at 24,000).
  // The extra headroom is deliberate: the app shortens from the MIDDLE, keeping the
  // beginning and the end, and it can only do that with more than it needs.
  var LIMIT = 60000;
  var element = pickArticle();
  var text = '';
  try { text = String((element && element.innerText) || ''); } catch (e) { text = ''; }
  if (!text && document.body) {
    try { text = String(document.body.innerText || ''); } catch (e) { text = ''; }
  }
  return shortenMiddle(text, LIMIT);
}

function pickArticle() {
  var candidates = [];
  try {
    // In order of how strongly each says "this is the article".
    var selectors = ['article', 'main', '[role="main"]', '#content', '.post', '.article', '.entry-content'];
    for (var s = 0; s < selectors.length; s++) {
      var found = document.querySelectorAll(selectors[s]);
      for (var i = 0; i < found.length; i++) candidates.push(found[i]);
    }
  } catch (e) {}
  if (!candidates.length) return document.body;

  // The longest of them, because a page with several <article> elements is a list of
  // teasers with one real article among them.
  var best = null, bestLength = 0;
  for (var c = 0; c < candidates.length; c++) {
    var length = 0;
    try { length = ((candidates[c].innerText || '')).length; } catch (e) { length = 0; }
    if (length > bestLength) { best = candidates[c]; bestLength = length; }
  }

  // If the best candidate is a small fraction of the page, the page is not built the
  // way this expects and the whole body is the safer answer: showing the model too
  // much is recoverable, showing it a navigation bar is not.
  var bodyLength = 0;
  try { bodyLength = ((document.body && document.body.innerText) || '').length; } catch (e) {}
  if (bestLength < 500 || (bodyLength > 0 && bestLength * 4 < bodyLength)) return document.body;
  return best;
}

var ExtensionPreprocessingJS = new TensorAgentPreprocessor();
