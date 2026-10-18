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
  c: 'cafes: any cafe, coffee shop, tea shop, bakery, brunch spot or restaurant to visit (also when the name contains words like cafe, coffee, 咖啡, 茶, 餐)',
  e: 'events: a concert, exhibition, market, talk or workshop with a date',
  r: 'recs: ONLY when someone explicitly recommends a show, song, game or video to watch, listen to or play; never study material or notes',
  w: 'wishes: something to buy, with a price or shop link',
  b: 'books: a book, with author or pages',
  p: 'places: a place to visit that is NOT a food or drink venue (park, trail, attraction, shop)',
  i: 'ideas: the user\'s own idea or note, and also study material: class or lecture notes, a photo of a blackboard, whiteboard, slide or notebook page, formulas, definitions',
  l: 'links: a bare link to read later',
  a: 'dates: a birthday, anniversary or recurring date',
  t: 'trips: a trip, itinerary, flight or hotel',
  o: 'opps: an opportunity with a deadline (internship, scholarship, application, competition)',
  g: 'gifts: a gift idea for someone',
};

const MAX_TEXT = 4000;
const MAX_IMAGE_CHARS = 4_500_000; // ~3.3 MB of base64
const PER_IP_PER_HOUR = 40;
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
  const { quick, ...rest } = opts;
  const send = (extra) => fetch(BASE_URL + 'chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + env.NEBIUS_API_KEY },
    body: JSON.stringify({ model, messages, temperature: 0.1, max_tokens: 2048, ...rest, ...extra }),
  });
  // "quick" turns off the model's long chain of thought (much faster for simple extraction). If the endpoint
  // rejects the flag we silently retry without it. Set AI_THINKING=1 to turn thinking back on everywhere.
  let res;
  if (quick && env.AI_THINKING !== '1') {
    res = await send({ chat_template_kwargs: { enable_thinking: false } });
    if (res.status === 400 || res.status === 422) res = await send({});
  } else res = await send({});
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
// Deterministic address check for Taiwan addresses: no model involved, so a wrong address cannot slip through.
const TW_ADDR = /(?:\d{3}\s*)?(?:臺北|台北|新北|桃園|臺中|台中|臺南|台南|高雄|基隆|新竹|嘉義|宜蘭|花蓮|臺東|台東|苗栗|彰化|南投|雲林|屏東|澎湖|金門|連江)[市縣]\s*([^\s,，。；;、]{1,6}?[區鄉鎮市])\s*([^\s,，。；;、]{0,12}?[路街道段][^\s,，。；;、]{0,10}?)\s*(\d+(?:之\d+)?(?:-\d+)?)\s*號/g;
const sq = (t) => String(t).replace(/\s+/g, '').replace(/臺/g, '台').toLowerCase();
function addressCheck(item, results, lang) {
  const N = lang === 'en' ? { ok: 'Address matches', diff: 'Address differs from what you saved' } : { ok: '地址與你存的相符', diff: '網路上的地址和你存的不同' };
  if (!['c', 'p'].includes(item.kind) || !item.place || !item.name) return null;
  const claim = [...String(item.place).matchAll(TW_ADDR)][0];
  if (!claim) return null;
  const key = (m) => sq(m[2]) + '|' + sq(m[3]);
  const claimKey = key(claim), claimDistrict = sq(claim[1]);
  const name = sq(item.name);
  let firstOther = null;
  for (const r of results) {
    const body = r.title + ' ' + r.content;
    if (!sq(body).includes(name)) continue; // only trust pages that name this exact shop
    for (const m of body.matchAll(TW_ADDR)) {
      if (key(m) === claimKey && sq(m[1]) === claimDistrict) {
        return { status: 'confirmed', note: N.ok, evidence: m[0].trim().slice(0, 160), source: { title: r.title, url: r.url } };
      }
      if (!firstOther) firstOther = { m, r };
    }
  }
  if (firstOther) {
    const { m, r } = firstOther;
    return { status: 'conflict', note: N.diff, evidence: m[0].trim().slice(0, 160), source: { title: r.title, url: r.url }, suggest: { date: '', deadline: '', place: m[0].replace(/^\d{3}\s*/, '').trim().slice(0, 120) } };
  }
  return null;
}

// Same idea for English street addresses (e.g. "66 Mint St").
const US_ADDR = /\b(\d{1,5})\s+((?:[NSEW]\.?\s+)?[A-Za-z0-9.'’-]+(?:\s+[A-Za-z0-9.'’-]+){0,2}?)\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Pl|Place|Ct|Court|Hwy|Highway|Pkwy|Parkway|Sq|Square)\b\.?/gi;
const TYPE_NORM = { street: 'st', avenue: 'ave', road: 'rd', boulevard: 'blvd', drive: 'dr', lane: 'ln', place: 'pl', court: 'ct', highway: 'hwy', parkway: 'pkwy', square: 'sq' };
function usAddressCheck(item, results, lang) {
  if (!['c', 'p'].includes(item.kind) || !item.place || !item.name) return null;
  const N = lang === 'en' ? { ok: 'Address matches', diff: 'Address differs from what you saved' } : { ok: '地址與你存的相符', diff: '網路上的地址和你存的不同' };
  const key = (m) => m[1] + '|' + m[2].toLowerCase().replace(/[^a-z0-9]/g, '') + '|' + (TYPE_NORM[m[3].toLowerCase()] || m[3].toLowerCase());
  const claim = [...String(item.place).matchAll(US_ADDR)][0];
  if (!claim) return null;
  const ck = key(claim), name = sq(item.name);
  let other = null;
  for (const r of results) {
    const body = r.title + ' ' + r.content;
    if (!sq(body).includes(name)) continue;
    for (const m of body.matchAll(US_ADDR)) {
      if (key(m) === ck) return { status: 'confirmed', note: N.ok, evidence: m[0].trim().slice(0, 160), source: { title: r.title, url: r.url } };
      if (!other) other = { m, r };
    }
  }
  if (other) return { status: 'conflict', note: N.diff, evidence: other.m[0].trim().slice(0, 160), source: { title: other.r.title, url: other.r.url }, suggest: { date: '', deadline: '', place: other.m[0].trim().slice(0, 120) } };
  return null;
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
  const det = addressCheck(item, results, lang) || usAddressCheck(item, results, lang);
  if (det) return { ...det, checked: results.length };
  // A snippet only counts if it is about this exact thing. Otherwise an unrelated article can "contradict" a made-up or
  // little-known event and push a wrong date onto the user.
  const same = (r) => sq(r.title + ' ' + r.content).includes(sq(item.name)) || grounding(item.name, r.title + ' ' + r.content) >= 0.75;
  const related = results.filter(same);
  if (!related.length) return { status: 'unverified', why: 'no_match', checked: results.length };
  results.length = 0; related.forEach((r) => results.push(r));
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
  ], { max_tokens: 3000, quick: true });
  let v;
  try { v = extractJson(out); } catch (e) { throw new Error('nojson:' + String(out).replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/\s+/g, ' ').trim().slice(0, 70) + (/<think>/.test(out) && !/<\/think>/.test(out) ? ' [cut during thinking]' : '')); }
  const idx = Number(v.source);
  const src = Number.isInteger(idx) && idx >= 0 && idx < results.length ? results[idx] : null;
  const ev = typeof v.evidence === 'string' ? v.evidence.trim() : '';
  const norm = (t) => String(t).replace(/\s+/g, ' ').toLowerCase();
  const quoted = src && ev.length >= 6 && norm(src.content + ' ' + src.title).includes(norm(ev));
  if (!['confirmed', 'conflict'].includes(v.status)) return { status: 'unverified', why: 'model_unsure', checked: results.length };
  if (!quoted) return { status: 'unverified', why: 'no_quote', checked: results.length };
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

// Models sometimes "fix" a proper name (e.g. read the word 打卡 as a verb and drop it). For places and shops, if the
// returned name is not literally in the source text, snap it back to the closest short line that is.
function snapName(name, text, kind) {
  if (!name || !['c', 'p'].includes(kind)) return name;
  const squash = (t) => String(t).replace(/\s+/g, '').toLowerCase();
  const lines = String(text).split('\n').map((l) => l.trim()).filter((l) => l.length >= 2 && l.length <= 30);
  const n = squash(name);
  const grams = (t) => { const g = []; for (let i = 0; i < t.length - 1; i++) g.push(t.slice(i, i + 2)); return g; };
  const dice = (a, b) => { const A = grams(a), B = grams(b); if (!A.length || !B.length) return 0; const m = new Map(); A.forEach((x) => m.set(x, (m.get(x) || 0) + 1)); let c = 0; B.forEach((x) => { if (m.get(x) > 0) { c++; m.set(x, m.get(x) - 1); } }); return (2 * c) / (A.length + B.length); };
  let best = null, bestScore = 0;
  for (const l of lines) {
    const q = squash(l);
    if (q === n) return name; // already verbatim as a line
    const score = q.includes(n) && q.endsWith(n) ? 0.9 : dice(n, q);
    if (score > bestScore) { bestScore = score; best = l; }
  }
  return best && bestScore >= 0.5 ? best.slice(0, 80) : name;
}

// A field the model filled in must be backed by the text. Drops invented places, notes and how-to lines, and flags invented names.
function grounding(field, text) {
  const grams = (t) => { const o = new Set(); const x = String(t).toLowerCase(); for (const m of x.matchAll(/[a-z0-9]{3,}/g)) o.add(m[0]); const c = x.replace(/[^\u4e00-\u9fff]/g, ''); for (let i = 0; i < c.length - 1; i++) o.add(c.slice(i, i + 2)); return o; };
  const f = grams(field), t = grams(text);
  if (!f.size) return 1;
  let hit = 0; f.forEach((g) => { if (t.has(g)) hit++; });
  return hit / f.size;
}

// Dated-and-timed mentions in the text (a sign-up deadline has no time, so it is not counted), first occurrence of each date.
function timedDates(t) {
  const s = String(t || '');
  const months = '(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\.?\\s+\\d{1,2}';
  const re = new RegExp('(\\d{4}-\\d{2}-\\d{2}|\\d{1,2}[/.月]\\d{1,2}|' + months + ')[^\\n]{0,15}?(?:\\d{1,2}[:：]\\d{2}|[上下]午|晚上|\\b\\d{1,2}\\s?(?:am|pm)\\b)', 'gi');
  const seen = new Set(), out = [];
  for (const m of s.matchAll(re)) { const k = m[1].toLowerCase(); if (!seen.has(k)) { seen.add(k); out.push({ key: k, index: m.index }); } }
  return out;
}
const countTimedDates = (t) => timedDates(t).length;

// Cut the text into one piece per event: each piece starts at the sentence that holds that event's date and time.
function splitByEvents(t) {
  const s = String(t || ''), hits = timedDates(s);
  if (hits.length < 2) return [s];
  const starts = hits.map((h, i) => {
    if (i === 0) return 0;
    const before = s.slice(0, h.index);
    let cut = 0;
    for (const m of before.matchAll(/[。！!？?\n；;]|\.\s|另外還有|另外|還有|\balso\b[:,]?/gi)) cut = Math.max(cut, m.index + m[0].length);
    return Math.max(cut, hits[i - 1].index + 1);
  });
  return starts.map((st, i) => s.slice(st, i + 1 < starts.length ? starts[i + 1] : s.length).trim()).filter(Boolean);
}

function looksLikeStudy(t) {
  const s = String(t || '');
  const hits = (s.match(/[=∫Σ∑πθωΔ∂√±]|\b[a-zA-Z]\w?\([a-z0-9]\)|\d\s*\/\s*\d/g) || []).length;
  const words = /lecture|notes?\b|筆記|上課|公式|定義|定理|theorem|proof|習題|例題|\bexample\b|考試|\bexam\b|chapter|章節|delay|延遲|transform|轉換|frequency|頻率|方程|equation|algorithm|演算法/i.test(s);
  return hits >= 4 || (hits >= 2 && words) || (words && /lecture|上課|筆記|notes?\b/i.test(s) && s.length > 40);
}

export async function onRequestPost({ request, env }) {
  const startedAt = Date.now();
  if (!env.NEBIUS_API_KEY) return json({ error: 'not_configured' }, 503);

  const ip = request.headers.get('cf-connecting-ip') || 'x';
  let body;
  try { body = await request.json(); } catch { return json({ error: 'bad_request' }, 400); }
  // Online checks follow every sort, so they get their own allowance and do not eat the sorting one.
  if (limited(body && body.mode === 'verify' ? ip + ':v' : ip, Number(env.PER_IP_PER_HOUR) || PER_IP_PER_HOUR)) return json({ error: 'rate_limited' }, 429);

  const today0 = /^\d{4}-\d{2}-\d{2}$/.test(body.today || '') ? body.today : new Date(Date.now() + 8 * 3600_000).toISOString().slice(0, 10);
  if (body.mode === 'verify') {
    // Second, separate request: the page shows the sorted card first and fills this in when it arrives.
    const it = body.item && typeof body.item === 'object' ? body.item : {};
    const s2 = (v, n) => (typeof v === 'string' ? v.slice(0, n) : '');
    const item = { kind: s2(it.kind, 2), name: s2(it.name, 80), place: s2(it.place, 120), date: s2(it.date, 10), time: s2(it.time, 5), deadline: s2(it.deadline, 10) };
    const lang2 = body.lang === 'en' ? 'en' : 'zh';
    if (!VERIFY_KINDS.includes(item.kind) || !item.name) return json({ verify_why: 'skipped' });
    if (!env.TAVILY_API_KEY) return json({ verify_why: 'no_key' });
    const t0 = Date.now();
    try {
      const v = await withTimeout(verifyOnline(env, item, lang2, today0), 20000);
      if (!v) return json({ verify_why: 'no_results', ms: Date.now() - t0 });
      if (v.status === 'unverified') return json({ verify_why: v.why || 'unverified', ms: Date.now() - t0 });
      return json({ verify: v, ms: Date.now() - t0 });
    } catch (e) {
      return json({ verify_why: String((e && e.message) || e).slice(0, 40), ms: Date.now() - t0 });
    }
  }

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
      const visionMsg = [{
        role: 'user',
        content: [
          { type: 'text', text: 'Transcribe all text in this image exactly as written (Chinese and English). If it is a photo of a blackboard, whiteboard, slide or notebook page, transcribe what is written on it (write formulas on one line, for example T_g(f) = -1/(2π) · dθ(f)/df), and ignore the room, ceiling, desks and people. Ignore the phone status bar (clock, battery, signal), keyboard, and any map or app background around the main content; transcribe only the main card or post. Mark anything unclear or hidden as [uncertain]; never guess. If nothing in the image is readable, output one short line starting with [photo] that says what it shows. Output only the transcription.' },
          { type: 'image_url', image_url: { url: image } },
        ],
      }];
      const readable = (t) => String(t || '').replace(/\[(?:uncertain|photo)\]/gi, '').replace(/\s+/g, '').length >= 12;
      // Thinking off first (fast). A dense blackboard photo can still come back empty, so retry once with more room.
      transcript = await chat(env, VISION_MODEL, visionMsg, { max_tokens: 3000, quick: true });
      if (!readable(transcript)) transcript = await chat(env, VISION_MODEL, visionMsg, { max_tokens: 4096 });
      if (!readable(transcript) && !text.trim()) return json({ error: 'unreadable' }, 422);
      text = (text ? text + '\n' : '') + transcript;
    }

    // Stage 2: classify + extract
    const runSort = async (kinds, extra, userText = text) => {
    const system = [
      'You sort things a person saved (screenshots, copied text) into the right tab of a small notebook app, and extract key fields.',
      'Allowed tabs (use the single-letter key):',
      ...kinds.map((k) => `- ${k} = ${KIND_DESC[k]}`),
      'Reply with ONE JSON object and nothing else:',
      `Today is ${today}.`,
      '{"kind": "<key>", "confidence": 0-1, "purpose": "remind | review | none", "name": "<the proper name exactly as written in the text, never reworded or shortened>", "date": "event date YYYY-MM-DD or empty", "time": "HH:MM 24h or empty", "place": "venue or address, or empty", "deadline": "sign-up/application deadline YYYY-MM-DD or empty", "how_to": "how to sign up / buy / get there, or empty", "next_step": "one concrete next action", "notes": "see rules", "summary": "see rules", "recall_q": "see rules", "recall_a": "see rules", "uncertain": ["fields you are not sure about"]}',
      'Rules: use ONLY what is in the text. If the text is too short or unclear to tell what it is, do not make anything up: pick the closest tab, leave every other field empty and set confidence below 0.3. Never invent facts. If a field is not stated, use an empty string. "date" is when the event itself happens; "deadline" is the last day to sign up or apply; never put one in the other. If a date has no year, use the nearest upcoming occurrence on or after today and add "year" to "uncertain". Text marked [uncertain] must be listed in "uncertain". If nothing fits well, pick the closest tab and set confidence below 0.5.',
      '"purpose": "remind" = time-bound things that only need a nudge when the time comes (events, deadlines, appointments, birthdays); "review" = knowledge or inspiration worth remembering and revisiting (a quote, a book passage, lecture notes, a key idea); "none" = just storing it (a place, something to buy, a link). Never choose "review" for an event or a deadline.',
      '"next_step" depends on the kind. For a quote or line the person loves (a favourite line from a show, film, book or song): a light, specific way to enjoy or keep it, never homework; vary it, choosing from: go back to the scene or page it came from, send it to the one person who would get it, or pair it with a song or a quiet café. For class or lecture notes, blackboard photos and formulas: study material is something to review, so use purpose "review", never "remind", and never kind recs; the next_step is a short, gentle suggestion for reviewing (never "ask the teacher" or "look it up"), phrased as an invitation and not a command (for example: you could try explaining it in your own words, see if you can answer the recall question without peeking, or look for one real example of it). For events, places and cafés: the concrete practical step. One sentence, under 40 Chinese characters or 90 English characters, warm and plain, no exclamation marks, and never state facts that are not in the text.',
      '"notes": useful facts worth keeping about the thing (opening hours, payment method, what to order, tips), one fact per line, max 240 characters, no app or screen text such as ratings widgets, buttons or clock; empty if none.',
      'If purpose is "review" and the content is class/lecture notes or a long passage: "summary" = concise study notes in the content language, max 500 characters, single line breaks only, no blank lines; otherwise "summary" is empty. If purpose is "review": "recall_q" = one question that tests the key idea, "recall_a" = a short answer taken from the content; otherwise both empty.',
      'If the content clearly holds several separate things (for example a list or poster of several different events), reply {"items": [object, ...]} with at most 6 objects of the shape above, one per thing; otherwise reply with a single object. Never split one thing\'s details (its date, place, deadline) into several items.',
      extra || '',
      lang === 'en' ? 'Write name, how_to, next_step, notes, summary, recall_q and recall_a in English.' : 'name、how_to、next_step、notes、summary、recall_q、recall_a 請用繁體中文書寫，用詞採台灣慣用語（例如「訊號」不寫「信號」、「軟體」不寫「軟件」、「資料」不寫「數據」）。',
    ].filter(Boolean).join('\n');

    const out = await chat(env, TEXT_MODEL, [
      { role: 'system', content: system },
      { role: 'user', content: userText },
    ], { max_tokens: 4096, quick: true });

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
    items.forEach((x) => { if (x.kind === 'q') x.summary = ''; }); // a quote stays as the user's own words
    items.forEach((x) => {
      ['place', 'how_to', 'notes'].forEach((f) => { if (x[f] && grounding(x[f], text) < 0.3) x[f] = ''; });
      if (['c', 'p', 'e', 'w', 'b'].includes(x.kind) && x.name && grounding(x.name, text) < 0.2) {
        x.name = ''; x.confidence = Math.min(x.confidence, 0.3); if (!x.uncertain.includes('name')) x.uncertain.push('name');
      }
    });
    return items;
    };
    let items = await runSort(kinds, '');
    // Study material (formulas, lecture or blackboard notes) must never land in recs/events/etc. The small model sometimes
    // does that anyway, so check the text itself and sort again with the choice narrowed to notes.
    // The small model often returns one event for a poster with several. Several timed dates in the text means several
    // events, so sort each piece on its own (in parallel) instead of trusting the model to split them.
    // (The model may call such a text a trip or a place, so the trigger does not depend on its first kind, except for
    // things that clearly are not event lists.)
    const tripish = /flight|hotel|itinerary|booking|check-?in|行程|機票|住宿|飯店|航班|訂房|旅館/i.test(text);
    if (items.length === 1 && kinds.includes('e') && !['q', 'c', 'b', 'w', 'i'].includes(items[0].kind) && !(items[0].kind === 't' && tripish) && countTimedDates(text) >= 2) {
      try {
        const pieces = splitByEvents(text).slice(0, 6);
        if (pieces.length >= 2) {
          const parts = await Promise.all(pieces.map((pc) => runSort(['e'], 'This text is one single event.', pc)));
          const evs = parts.map((a) => a[0]).filter((x) => x && x.kind === 'e' && x.name);
          if (evs.length >= 2) items = evs;
        }
      } catch { /* keep the first answer */ }
    }
    // A small model sometimes calls lecture notes a "book" (a title like "Lecture 6" looks like one). Only keep the book label when the text
    // really has book signs (author, ISBN, page count); otherwise sort it again as study material.
    const bookCue = /ISBN|作者|出版|著\b|\bauthor\b|paperback|hardcover|\bby\s+[A-Z][a-z]+|共\s*\d+\s*頁|\b\d+\s*pages\b/i.test(text);
    if (items.length === 1 && looksLikeStudy(text) && (items[0].kind === 'b' ? !bookCue : !items[0].date) && !['i', 'q'].includes(items[0].kind) && (kinds.includes('i') || kinds.includes('q'))) {
      try {
        const again = await runSort([kinds.includes('i') ? 'i' : 'q'], 'The content is study material (class or lecture notes, formulas, definitions). It is for review: purpose must be "review", and "summary", "recall_q" and "recall_a" must be filled in.');
        if (again.length === 1 && again[0].name) items = again;
      } catch { /* keep the first answer */ }
    }
    if (items.length === 1) {
      const it = items[0];
      if (it.kind === 'p' && kinds.includes('c') && /咖啡|cafe|café|coffee|茶|烘焙|bakery|早午餐|brunch|餐廳|restaurant/i.test(it.name + ' ' + text.slice(0, 200))) it.kind = 'c';
      it.name = snapName(it.name, text, it.kind);
    }
    let verify, verifyWhy = '';
    if (body.verify !== false && items.length === 1 && VERIFY_KINDS.includes(items[0].kind) && items[0].name) {
      if (!env.TAVILY_API_KEY) verifyWhy = 'no_key';
      else {
        try { verify = await withTimeout(verifyOnline(env, items[0], lang, today), 16000); if (!verify) verifyWhy = 'no_results'; else if (verify.status === 'unverified') { verifyWhy = verify.why || 'unverified'; verify = undefined; } }
        catch (e) { verify = undefined; verifyWhy = String((e && e.message) || e).slice(0, 40); }
      }
    }
    return json({
      ...items[0],
      ...(verify ? { verify } : {}),
      ...(verifyWhy ? { verify_why: verifyWhy } : {}),
      ...(items.length > 1 ? { items } : {}),
      ms: Date.now() - startedAt,
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
