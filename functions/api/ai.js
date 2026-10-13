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

function extractJson(s) {
  s = String(s).replace(/<think>[\s\S]*?<\/think>/g, '');
  const a = s.indexOf('{'), b = s.lastIndexOf('}');
  if (a < 0 || b <= a) throw new Error('no json');
  return JSON.parse(s.slice(a, b + 1));
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
          { type: 'text', text: 'Transcribe all text in this image exactly as written (Chinese and English). Mark anything unclear or hidden as [uncertain]; never guess. Output only the transcription.' },
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
      '{"kind": "<key>", "confidence": 0-1, "name": "<short title>", "date": "YYYY-MM-DD or empty", "deadline": "YYYY-MM-DD or empty", "how_to": "how to sign up / buy / get there, or empty", "next_step": "one concrete next action", "uncertain": ["fields you are not sure about"], "question": "one self-check question about the content"}',
      'Rules: never invent facts. If a field is not stated, use an empty string. Text marked [uncertain] must be listed in "uncertain". If nothing fits well, pick the closest tab and set confidence below 0.5.',
      lang === 'en' ? 'Write name, how_to, next_step and question in English.' : 'name、how_to、next_step、question 請用繁體中文書寫。',
    ].join('\n');

    const out = await chat(env, TEXT_MODEL, [
      { role: 'system', content: system },
      { role: 'user', content: text },
    ], { max_tokens: 4096 });

    const r = extractJson(out);
    const kind = kinds.includes(r.kind) ? r.kind : kinds[0];
    const str = (v, n) => (typeof v === 'string' ? v.slice(0, n) : '');
    return json({
      kind,
      confidence: Math.max(0, Math.min(1, Number(r.confidence) || 0)),
      name: str(r.name, 80),
      date: /^\d{4}-\d{2}-\d{2}$/.test(r.date || '') ? r.date : '',
      deadline: /^\d{4}-\d{2}-\d{2}$/.test(r.deadline || '') ? r.deadline : '',
      how_to: str(r.how_to, 300),
      next_step: str(r.next_step, 300),
      uncertain: Array.isArray(r.uncertain) ? r.uncertain.slice(0, 8).map((x) => str(String(x), 60)) : [],
      question: str(r.question, 200),
      transcript: transcript ? transcript.slice(0, MAX_TEXT) : '',
      models: image ? [VISION_MODEL, TEXT_MODEL] : [TEXT_MODEL],
    });
  } catch (e) {
    return json({ error: 'ai_failed', detail: String((e && e.message) || e).slice(0, 120) }, 502);
  }
}

export async function onRequest({ request }) {
  return json({ error: 'method_not_allowed' }, 405);
}
