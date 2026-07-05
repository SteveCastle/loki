// src/renderer/query/parse.ts
import type { Predicate, PredicateType, Query } from './types';

// Tokenize respecting double-quoted phrases (mirrors src/main/media.ts
// parseSearchString, kept here so the renderer owns all parsing).
function tokenize(input: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < input.length; i++) {
    const c = input[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (/\s/.test(c) && !inQuotes) {
      if (current.trim()) tokens.push(current);
      current = '';
    } else {
      current += c;
    }
  }
  if (current.trim()) tokens.push(current);
  return tokens;
}

const PREFIXES: Array<{ prefix: string; type: PredicateType }> = [
  { prefix: 'in:', type: 'category' },
  { prefix: 'path:', type: 'path' },
  { prefix: 'description:', type: 'description' },
  { prefix: 'hash:', type: 'hash' },
  { prefix: 'similar:', type: 'similar' },
  { prefix: 'visual:', type: 'visual' },
];

// Strip surrounding quotes that survived tokenization of a prefixed value
// (e.g. path:"a b" -> the token is path:a b already; quotes only appear when
// the prefix itself is quoted, which we do not support — defensive trim).
function clean(value: string): string {
  return value.replace(/^"|"$/g, '').trim();
}

export function parseQuery(input: string): Predicate[] {
  const preds: Predicate[] = [];
  for (let token of tokenize(input)) {
    let exclude = false;
    if (token.startsWith('-')) {
      exclude = true;
      token = token.slice(1);
    }
    if (token.startsWith('#')) {
      const value = clean(token.slice(1));
      if (value) preds.push({ type: 'tag', value, exclude });
      continue;
    }
    let matched = false;
    for (const { prefix, type } of PREFIXES) {
      if (token.startsWith(prefix)) {
        const value = clean(token.slice(prefix.length));
        if (value) preds.push({ type, value, exclude });
        matched = true;
        break;
      }
    }
    if (matched) continue;
    const value = clean(token);
    if (value) preds.push({ type: 'description', value, exclude });
  }
  return preds;
}

const TYPE_PREFIX: Record<PredicateType, string> = {
  tag: '#',
  category: 'in:',
  path: 'path:',
  description: 'description:',
  hash: 'hash:',
  similar: 'similar:',
  visual: 'visual:',
  // Not parseable from typed text (the value is a PNG data URL); serialization
  // only — the query persists as JSON, never through parseQuery.
  clip: 'clip:',
};

export function serializePredicate(p: Predicate): string {
  const sign = p.exclude ? '-' : '';
  const needsQuotes = /\s/.test(p.value);
  const value = needsQuotes ? `"${p.value}"` : p.value;
  return `${sign}${TYPE_PREFIX[p.type]}${value}`;
}

export function serializeQuery(q: Query): string {
  return q.predicates.map(serializePredicate).join(' ');
}
