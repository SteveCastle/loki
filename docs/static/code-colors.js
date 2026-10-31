// Syntax roles for <code> across the docs site.
//
// One colour per structural role, the same on every page (see the .c-* rules
// in index.css):
//   c-cmd    literal words you type         lokictl, job copy, field names
//   c-flag   options                        --follow, -o
//   c-ph     things you substitute          <pool>, {id}, N (in signatures)
//   c-path   locations                      /api/media, C:/photos, https://..
//   c-env    environment variables          LOKICTL_TOKEN, $RUNPOD_API_KEY
//   c-val    values                         "completed", 30s, true, null
//   c-punct  structure                      [ ] ( ) | ... , JSON braces
//   c-cmt    shell comments                 # ...
//
// Colour is an enhancement only: without this script code renders in one
// neutral colour and reads the same. Text nodes are rewritten in place, so
// structural spans already in the markup (.tok, which keeps a token on one
// line) are preserved.
(function () {
  'use strict';

  var FLAG = /^--?[A-Za-z][\w-]*$/;
  var FLAG_EQ = /^(--?[A-Za-z][\w-]*)=(.*)$/;
  var ENV = /^\$?[A-Z][A-Z0-9]*_[A-Z0-9_]+$|^\$[A-Z][A-Z0-9_]*$/;
  var ENV_EQ = /^(\$?[A-Z][A-Z0-9]*_[A-Z0-9_]+)=(.*)$/;
  var PH = /^<[^<>]+>(\.\.\.)?$|^\{[\w.-]+\}$/;
  var METHOD = /^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)$/;
  var METAVAR =/^[A-Z][A-Z0-9]{0,5}$/; // N, K, ID, URL: placeholders in signatures
  var PATH = /^(https?:\/\/|s3:\/\/|\/|~\/|\.\.?\/|[A-Za-z]:[\\/]|%[A-Z]+%)/;
  var VAL = /^(-?\d+(\.\d+)?[a-zA-Z%]*|true|false|null)$/;

  function span(cls, text) {
    var s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    return s;
  }

  // Path with {param} segments: the param is a placeholder.
  function pathNodes(text, out) {
    var re = /\{[\w.-]+\}/g, last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(span('c-path', text.slice(last, m.index)));
      out.push(span('c-ph', m[0]));
      last = re.lastIndex;
    }
    if (last < text.length) out.push(span('c-path', text.slice(last)));
  }

  // Classify one bare token (no surrounding punctuation).
  function word(tok, st, out) {
    var m;
    if (tok === '...' || tok === '…') return out.push(span('c-punct', tok));
    if ((m = FLAG_EQ.exec(tok))) {
      st.args = true;
      out.push(span('c-flag', m[1]), span('c-punct', '='));
      return word(m[2], st, out);
    }
    if (FLAG.test(tok)) { st.args = true; return out.push(span('c-flag', tok)); }
    if ((m = ENV_EQ.exec(tok))) {
      out.push(span('c-env', m[1]), span('c-punct', '='));
      return word(m[2], st, out);
    }
    if (METHOD.test(tok)) return out.push(span('m-' + tok.toLowerCase(), tok));
    if (ENV.test(tok)) return out.push(span('c-env', tok));
    // In a signature, repeatable (args...) and key=value slots are placeholders too.
    if (PH.test(tok) || (st.signature && (METAVAR.test(tok) || /\w\.\.\.$/.test(tok) || /^[a-z]+=[a-z]+$/.test(tok)))) {
      st.args = true;
      return out.push(span('c-ph', tok));
    }
    if (PATH.test(tok)) { st.args = true; return pathNodes(tok, out); }
    if (VAL.test(tok) || /^(["']).*\1$/.test(tok)) return out.push(span('c-val', tok));
    // Alternatives a|b|c: each alternative classified, bars are structure.
    if (tok.indexOf('|') > 0) {
      tok.split(/(\|)/).forEach(function (p) {
        if (p === '|') out.push(span('c-punct', p));
        else if (p) word(p, st, out);
      });
      return;
    }
    // Leading words of a command or signature are the literal command.
    if (st.command && !st.args) return out.push(span('c-cmd', tok));
    out.push(document.createTextNode(tok));
  }

  // Classify a whitespace-free chunk, splitting off edge punctuation.
  function chunk(text, st, out) {
    var lead = '', trail = '', m;
    while ((m = /^[\[\](){},;]/.exec(text)) && text.length > 1) { lead += m[0]; text = text.slice(1); }
    while ((m = /[\[\](){},;]$/.exec(text)) && text.length > 1 && !PH.test(text)) { trail = m[0] + trail; text = text.slice(0, -1); }
    if (lead) out.push(span('c-punct', lead));
    if (text) word(text, st, out);
    if (trail) out.push(span('c-punct', trail));
  }

  // JSON-looking code: keys are literal names, values are values.
  function json(text, out) {
    var re = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)|([{}\[\],:])|([^"{}\[\],:\s]+|\s+)/g, m;
    while ((m = re.exec(text))) {
      if (m[1]) {
        out.push(span(m[2] ? 'c-cmd' : 'c-val', m[1]));
        if (m[2]) out.push(span('c-punct', m[2]));
      } else if (m[3]) out.push(span('c-val', m[3]));
      else if (m[4]) out.push(span('c-punct', m[4]));
      else out.push(document.createTextNode(m[5]));
    }
  }

  function textNodes(el) {
    var w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT), n, list = [];
    while ((n = w.nextNode())) list.push(n);
    return list;
  }

  function colour(code) {
    if (code.dataset.colored) return;
    code.dataset.colored = '1';
    var full = code.textContent.trim();
    var inBlock = !!code.closest('.code-block');
    var st = {
      signature: !!code.closest('.ref-key'),
      command: !!code.closest('.ref-key') || inBlock,
      args: false,
      comment: false,
    };
    var isJSON = /^[\[{]/.test(full) && /[\]}]$/.test(full) && /["\d]|true|false|null/.test(full) && !st.signature;
    textNodes(code).forEach(function (node) {
      var text = node.nodeValue, out = [];
      if (!text) return;
      if (isJSON) {
        json(text, out);
      } else {
        // Quoted strings come out whole (spaces included) as one part.
        (text.match(/"[^"]*"?|'[^']*'?|\s+|[^\s"']+/g) || []).forEach(function (part) {
          if (st.comment) return out.push(span('c-cmt', part));
          if (/^\s+$/.test(part)) return out.push(document.createTextNode(part));
          if (inBlock && part.charAt(0) === '#') { st.comment = true; return out.push(span('c-cmt', part)); }
          // A quoted location is still a location; any other quoted text is a value.
          if (/^["']/.test(part)) return out.push(span(PATH.test(part.slice(1)) ? 'c-path' : 'c-val', part));
          chunk(part, st, out);
        });
      }
      var frag = document.createDocumentFragment();
      out.forEach(function (n) { frag.appendChild(n); });
      node.parentNode.replaceChild(frag, node);
    });
  }

  function run() {
    document.querySelectorAll('main code, .subpage-hero code').forEach(colour);
    document.querySelectorAll('.ref-method').forEach(function (m) {
      var verb = (m.textContent.trim().split(/[,\s]/)[0] || '').toLowerCase();
      if (verb) m.classList.add('m-' + verb);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
  else run();
})();
