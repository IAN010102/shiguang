// Cloudflare Pages Function: POST /api/ai
// Two-stage pipeline on Nebius Token Factory:
//   1) (optional) vision model reads a screenshot/photo -> plain text, flags unclear parts as [uncertain]
//   2) NVIDIA Nemotron classifies the text into one of the user's tabs and extracts fields
// The API key lives only in the NEBIUS_API_KEY environment secret. It never reaches the browser.

const BASE_URL = 'https://api.tokenfactory.nebius.com/v1/';
const VISION_MODEL = 'zai-org/GLM-5.3-Flash';
const TEXT_MODEL = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B';

const KIND_DESC = {
  q: 'quotes: a sentence or line worth keeping',
  c: 'cafes: a cafe, restaurant or shop to visit',
  e: 'events: a concert, exhibition, market, talk or workshop with a date',
  r: 'recs: something someone recommends (a show, song, game, video)',
  w: 'wishes: something to buy, with a price or shop link',
  b: 'books: a book, with author or pages',
  p: 'places: a non-food place to visit (park, trail, attraction)',
  i: 'ideas: the user\'s own idea or note',
  l: 'links: a bare link to read later',
  a: 'dates: a birthday, anniversary or recurring date',
  t: 'trips: a trip, itinerary, flight or hotel',
  o: 'opps: an opportunity with a deadline (internship, scholarship, application, competition)',
  g: 'gifts: a gift idea for someone',
};

const MAX_TEXT = 4000;
const MAX_IMAGE_CHARS = 4_500_000; // ~3.3 MB of base64
const PER_IP_PER_HOUR = 20;
const hits = new Map(); // best-effort, per isolate

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' },
  });
}

function limited(ip, cap) {
  const now = Date.now();
  const arr = (hits.get(ip) || []).filter((t) => now - t < 3600_000);
  if (arr.length >= cap) { hits.set(ip, arr); return true; }
  arr.push(now); hits.set(ip, arr);
  if (hits.size > 5000) hits.clear();
  return false;
}

async function chat(env, model, messages, opts = {}) {
  const res = await fetch(BASE_URL + 'chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + env.NEBIUS_API_KEY },
    body: JSON.stringify({ model, messages, temperature: 0.1, max_tokens: 2048, ...opts }),
  });
  if (!res.ok) throw new Error('upstream ' + res.status);
  const data = await res.json();
  return (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || '';
}

// Models sometimes put raw line breaks inside JSON strings; escape them so JSON.parse accepts the text.
function escapeControlInStrings(t) {
  let out = '', inStr = false, esc = false;
  for (const ch of t) {
    if (inStr) {
      if (esc) { out += ch; esc = false; continue; }
      if (ch === '\\') { out += ch; esc = true; continue; }
      if (ch === '"') { inStr = false; out += ch; continue; }
      if (ch === '\n') { out += '\\n'; continue; }
      if (ch === '\r') { continue; }
      if (ch === '\t') { out += '\\t'; continue; }
      out += ch;
    } else {
      if (ch === '"') inStr = true;
      out += ch;
    }
  }
  return out;
}

function extractJson(s) {
  s = String(s).replace(/<think>[\s\S]*?<\/think>/g, '');
  const a = s.indexOf('{'), b = s.lastIndexOf('}');
  if (a < 0 || b <= a) throw new Error('no json');
  const body = s.slice(a, b + 1);
  try { return JSON.parse(body); } catch { return JSON.parse(escapeControlInStrings(body)); }
}

// Optional stage 3: look the item up on the web with Tavily and let the model compare.
// It only reports what a quoted snippet supports; anything else is dropped. Failure never breaks the main result.
const VERIFY_KINDS = ['e', 'c', 'o', 'p'];
function withTimeout(p, ms) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
}
async function verifyOnline(env, item, lang, today) {
  const q = [item.name, item.place].filter(Boolean).join(' ').slice(0, 200);
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + env.TAVILY_API_KEY },
    body: JSON.stringify({ query: q, search_depth: 'basic', max_results: 4, include_answer: false }),
  });
  if (!res.ok) throw new Error('tavily ' + res.status);
  const data = await res.json();
  const results = (data.results || []).filter((r) => r && r.url && r.content).slice(0, 4)
    .map((r) => ({ title: String(r.title || '').slice(0, 120), url: String(r.url), content: String(r.content).slice(0, 700) }));
  if (!results.length) return null;
  const claim = { name: item.name, date: item.date, time: item.time, place: item.place, deadline: item.deadline };
  const sys = [
    'You fact-check details a user saved against web search snippets. Today is ' + today + '.',
    'Reply with ONE JSON object only: {"status": "confirmed | conflict | unverified", "source": <index of the snippet used, or -1>, "evidence": "<exact short quote copied from that snippet, max 160 chars, or empty>", "note": "<one short sentence about what matches or differs>", "suggest": {"date": "", "deadline": "", "place": ""}}',
    '"confirmed" = a snippet states the same event/place with matching details. "conflict" = a snippet clearly states a different date, deadline or place for the same thing; put the snippet\'s value in "suggest". "unverified" = snippets do not mention it or are about something else. Never guess; when unsure choose "unverified". "evidence" must be copied verbatim from the snippet.',
    lang === 'en' ? 'Write "note" in English.' : '"note" 請用繁體中文書寫。',
  ].join('\n');
  const out = await chat(env, TEXT_MODEL, [
    { role: 'system', content: sys },
    { role: 'user', content: 'Saved details: ' + JSON.stringify(claim) + '\nSnippets:\n' + results.map((r, i) => `[${i}] ${r.title}\n${r.content}`).join('\n\n') },
  ], { max_tokens: 700 });
  const v = extractJson(out);
  const idx = Number(v.source);
  const src = Number.isInteger(idx) && idx >= 0 && idx < results.length ? results[idx] : null;
  const ev = typeof v.evidence === 'string' ? v.evidence.trim() : '';
  const norm = (t) => String(t).replace(/\s+/g, ' ').toLowerCase();
  const quoted = src && ev.length >= 6 && norm(src.content + ' ' + src.title).includes(norm(ev));
  if (!['confirmed', 'conflict'].includes(v.status) || !quoted) return { status: 'unverified', checked: results.length };
  const sg = v.suggest && typeof v.suggest === 'object' ? v.suggest : {};
  const dOk = (x) => (/^\d{4}-\d{2}-\d{2}$/.test(x || '') ? x : '');
  return {
    status: v.status,
    note: typeof v.note === 'string' ? v.note.slice(0, 200) : '',
    evidence: ev.slice(0, 160),
    source: { title: src.title, url: src.url },
    suggest: v.status === 'conflict' ? { date: dOk(sg.date), deadline: dOk(sg.deadline), place: typeof sg.place === 'string' ? sg.place.slice(0, 120) : '' } : undefined,
    checked: results.length,
  };
}

export async function onRequestPost({ request, env }) {
  if (!env.NEBIUS_API_KEY) return json({ error: 'not_configured' }, 503);

  const ip = request.headers.get('cf-connecting-ip') || 'x';
  if (limited(ip, Number(env.PER_IP_PER_HOUR) || PER_IP_PER_HOUR)) return json({ error: 'rate_limited' }, 429);

  let body;
  try { body = await request.json(); } catch { return json({ error: 'bad_request' }, 400); }

  let text = typeof body.text === 'string' ? body.text.slice(0, MAX_TEXT) : '';
  const image = typeof body.image === 'string' ? body.image : '';
  const lang = body.lang === 'en' ? 'en' : 'zh';
  const today = /^\d{4}-\d{2}-\d{2}$/.test(body.today || '') ? body.today : new Date(Date.now() + 8 * 3600_000).toISOString().slice(0, 10);
  const kinds = (Array.isArray(body.kinds) ? body.kinds : []).filter((k) => KIND_DESC[k]);
  if (!kinds.length) return json({ error: 'bad_request' }, 400);
  if (!text.trim() && !image) return json({ error: 'empty' }, 400);
  if (image && (!/^data:image\/(png|jpe?g|webp);base64,/.test(image) || image.length > MAX_IMAGE_CHARS)) {
    return json({ error: 'bad_image' }, 400);
  }

  try {
    // Stage 1: read the image
    let transcript = '';
    if (image) {
      transcript = await chat(env, VISION_MODEL, [{
        role: 'user',
        content: [
          { type: 'text', text: 'Transcribe all text in this image exactly as written (Chinese and English). Ignore the phone status bar (clock, battery, signal), keyboard, and any map or app background around the main content; transcribe only the main card or post. Mark anything unclear or hidden as [uncertain]; never guess. Output only the transcription.' },
          { type: 'image_url', image_url: { url: image } },
        ],
      }], { max_tokens: 2048 });
      text = (text ? text + '\n' : '') + transcript;
    }

    // Stage 2: classify + extract
    const system = [
      'You sort things a person saved (screenshots, copied text) into the right tab of a small notebook app, and extract key fields.',
      'Allowed tabs (use the single-letter key):',
      ...kinds.map((k) => `- ${k} = ${KIND_DESC[k]}`),
      'Reply with ONE JSON object and nothing else:',
      `Today is ${today}.`,
      '{"kind": "<key>", "confidence": 0-1, "purpose": "remind | review | none", "name": "<short title>", "date": "event date YYYY-MM-DD or empty", "time": "HH:MM 24h or empty", "place": "venue or address, or empty", "deadline": "sign-up/application deadline YYYY-MM-DD or empty", "how_to": "how to sign up / buy / get there, or empty", "next_step": "one concrete next action", "notes": "see rules", "summary": "see rules", "recall_q": "see rules", "recall_a": "see rules", "uncertain": ["fields you are not sure about"]}',
      'Rules: never invent facts. If a field is not stated, use an empty string. "date" is when the event itself happens; "deadline" is the last day to sign up or apply; never put one in the other. If a date has no year, use the nearest upcoming occurrence on or after today and add "year" to "uncertain". Text marked [uncertain] must be listed in "uncertain". If nothing fits well, pick the closest tab and set confidence below 0.5.',
      '"purpose": "remind" = time-bound things that only need a nudge when the time comes (events, deadlines, appointments, birthdays); "review" = knowledge or inspiration worth remembering and revisiting (a quote, a book passage, lecture notes, a key idea); "none" = just storing it (a place, something to buy, a link). Never choose "review" for an event or a deadline.',
      '"notes": useful facts worth keeping about the thing (opening hours, payment method, what to order, tips), one fact per line, max 240 characters, no app or screen text such as ratings widgets, buttons or clock; empty if none.',
      'If purpose is "review" and the content is class/lecture notes or a long passage: "summary" = concise study notes in the content language, max 500 characters, single line breaks only, no blank lines; otherwise "summary" is empty. If purpose is "review": "recall_q" = one question that tests the key idea, "recall_a" = a short answer taken from the content; otherwise both empty.',
      'If the content clearly holds several separate things (for example a list or poster of several different events), reply {"items": [object, ...]} with at most 6 objects of the shape above, one per thing; otherwise reply with a single object. Never split one thing\'s details (its date, place, deadline) into several items.',
      lang === 'en' ? 'Write name, how_to, next_step, notes, summary, recall_q and recall_a in English.' : 'name、how_to、next_step、notes、summary、recall_q、recall_a 請用繁體中文書寫。',
    ].join('\n');

    const out = await chat(env, TEXT_MODEL, [
      { role: 'system', content: system },
      { role: 'user', content: text },
    ], { max_tokens: 4096 });

    const parsed = extractJson(out);
    const str = (v, n) => (typeof v === 'string' ? v.slice(0, n) : '');
    const shape = (r) => ({
      kind: kinds.includes(r.kind) ? r.kind : kinds[0],
      confidence: Math.max(0, Math.min(1, Number(r.confidence) || 0)),
      name: str(r.name, 80),
      time: /^([01]\d|2[0-3]):[0-5]\d$/.test(r.time || '') ? r.time : '',
      purpose: ['remind', 'review', 'none'].includes(r.purpose) ? r.purpose : 'none',
      notes: str(r.notes, 300).replace(/\n\s*\n/g, '\n'),
      summary: str(r.summary, 600).replace(/\n\s*\n/g, '\n'),
      recall_q: str(r.recall_q, 200),
      recall_a: str(r.recall_a, 300),
      place: str(r.place, 120),
      date: /^\d{4}-\d{2}-\d{2}$/.test(r.date || '') ? r.date : '',
      deadline: /^\d{4}-\d{2}-\d{2}$/.test(r.deadline || '') ? r.deadline : '',
      how_to: str(r.how_to, 300),
      next_step: str(r.next_step, 300),
      uncertain: Array.isArray(r.uncertain) ? r.uncertain.slice(0, 8).map((x) => str(String(x), 60)) : [],
    });
    const rawItems = Array.isArray(parsed.items) ? parsed.items.filter((x) => x && typeof x === 'object').slice(0, 6) : [parsed];
    const items = (rawItems.length ? rawItems : [parsed]).map(shape);
    let verify;
    if (env.TAVILY_API_KEY && items.length === 1 && VERIFY_KINDS.includes(items[0].kind) && items[0].name) {
      try { verify = await withTimeout(verifyOnline(env, items[0], lang, today), 9000); } catch { verify = undefined; }
    }
    return json({
      ...items[0],
      ...(verify ? { verify } : {}),
      ...(items.length > 1 ? { items } : {}),
      transcript: transcript ? transcript.slice(0, MAX_TEXT) : '',
      models: (image ? [VISION_MODEL, TEXT_MODEL] : [TEXT_MODEL]).concat(verify ? ['Tavily'] : []),
    });
  } catch (e) {
    return json({ error: 'ai_failed', detail: String((e && e.message) || e).slice(0, 120) }, 502);
  }
}

export async function onRequest({ request }) {
  return json({ error: 'method_not_allowed' }, 405);
}
