// The page's strings, in the language the app is showing.
//
// Served as /i18n.js, after one line PageStrings writes in front of this file:
//   window.TensorAgentI18n = { lang: "zh-Hans", strings: { "page.key": "…", … } };
// The tables are TensorAgent.Core's Localization/<tag>/<area>.json, the same ones the
// native screens read, with English filling any gap. This file only reads them:
//
//   TensorAgentI18n.t(key, params)          text for a key, {name} placeholders filled
//   TensorAgentI18n.tn(key, count, params)  key.one / key.other by the language's
//                                           plural rule, the count as {count}
//   TensorAgentI18n.apply(root)             the markup's data-i18n attributes, each
//       holding a key: data-i18n for the element's text (so it must hold text only),
//       data-i18n-placeholder, data-i18n-title and data-i18n-aria-label for those
//
// A key with no text shows as the key itself, and is recorded in TensorAgentI18n.missing,
// so a gap is visible rather than blank.
(function () {
  'use strict';
  var I = window.TensorAgentI18n;
  if (!I || typeof I !== 'object') I = window.TensorAgentI18n = { lang: 'en', strings: {} };
  if (!I.strings || typeof I.strings !== 'object') I.strings = {};
  var missing = I.missing = I.missing || {};

  function format(text, params) {
    return String(text).replace(/\{\{|\}\}|\{([A-Za-z0-9_]+)\}/g, function (match, name) {
      if (match === '{{') return '{';
      if (match === '}}') return '}';
      return params && Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match;
    });
  }

  function lookup(key) {
    var text = I.strings[key];
    if (typeof text === 'string') return text;
    missing[key] = true;
    return key;
  }

  function t(key, params) { return format(lookup(key), params); }

  // The CLDR cardinal categories of the shipped languages, for integers: the same rule
  // StringCatalog.PluralCategory applies on the native side.
  function category(n) {
    switch (I.lang) {
      case 'zh-Hans': case 'zh-Hant': case 'ja': case 'ko': return 'other';
      case 'fr': return (n === 0 || n === 1) ? 'one' : 'other';
      default: return n === 1 ? 'one' : 'other';
    }
  }

  function tn(key, count, params) {
    var filled = {};
    if (params) {
      for (var name in params) {
        if (Object.prototype.hasOwnProperty.call(params, name)) filled[name] = params[name];
      }
    }
    // Digits as the native side writes a count, without grouping: "1000 chats" on both.
    filled.count = String(count);
    return format(lookup(key + '.' + category(count)), filled);
  }

  var ATTRIBUTES = [
    ['data-i18n-placeholder', 'placeholder'],
    ['data-i18n-title', 'title'],
    ['data-i18n-aria-label', 'aria-label']
  ];

  function apply(root) {
    root = root || document;
    if (!root || typeof root.querySelectorAll !== 'function') return;
    var texts = root.querySelectorAll('[data-i18n]');
    for (var i = 0; i < texts.length; i++) texts[i].textContent = t(texts[i].getAttribute('data-i18n'));
    ATTRIBUTES.forEach(function (pair) {
      var nodes = root.querySelectorAll('[' + pair[0] + ']');
      for (var j = 0; j < nodes.length; j++) nodes[j].setAttribute(pair[1], t(nodes[j].getAttribute(pair[0])));
    });
  }

  I.t = t;
  I.tn = tn;
  I.format = format;
  I.apply = apply;

  // mask-editor.js is shared with the server's own Web UI, which has no tables: it asks
  // window.TensorSharpI18n for a key and keeps its own English when nothing answers.
  window.TensorSharpI18n = {
    t: function (key, fallback, params) {
      var text = I.strings[key];
      return format(typeof text === 'string' ? text : fallback, params);
    }
  };

  if (typeof document !== 'undefined' && document && document.documentElement
      && typeof document.documentElement.setAttribute === 'function') {
    document.documentElement.setAttribute('lang', I.lang);
  }
  if (typeof document !== 'undefined') {
    function paint() {
      apply(document);
      var gate = document.getElementById('tensoragent-language-loading');
      if (gate && gate.parentNode) gate.parentNode.removeChild(gate);
    }
    // Loaded in the head, before any visible markup. Reveal the page only once the
    // body has been translated, so a first launch does not flash English.
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', paint);
    else paint();
  }
})();
