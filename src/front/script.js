/* ══════════════════════════════════════════════════════
   Digital Twin — Unified Frontend Logic
   ══════════════════════════════════════════════════════ */

// ─── State ──────────────────────────────────────────
let sessions = JSON.parse(localStorage.getItem('dt_sessions')) || [];
let messages = JSON.parse(localStorage.getItem('dt_messages')) || {};
let activeSession = null;
window.availablePersonas = [];

// ─── DOM ────────────────────────────────────────────
const $ = id => document.getElementById(id);

const viewWelcome   = $('viewWelcome');
const viewChat      = $('viewChat');
const wcGrid        = $('wcGrid');
const wcInput       = $('wcInput');
const wcSend        = $('wcSend');
const chatName      = $('chatName');
const chatBadge     = $('chatBadge');
const chatScroll    = $('chatScroll');
const chatMessages  = $('chatMessages');
const chatInput     = $('chatInput');
const chatSendBtn   = $('chatSendBtn');
const micBtn        = $('micBtn');
const fileInput     = $('fileInput');
const clearBtn      = $('clearBtn');
const exportBtn     = $('exportBtn');
const sbList        = $('sbList');
const sidebar       = $('sidebar');
const sbOverlay     = $('sbOverlay');
const menuBtn       = $('menuBtn');
const btnNewPersona = $('btnNewPersona');
const btnNewTutor   = $('btnNewTutor');
const personaModal  = $('personaModal');
const personaList   = $('personaList');
const modalCloseBtn = $('modalCloseBtn');
const themeBtn      = $('themeBtn');
const themeIcon     = $('themeIcon');
const themeLbl      = $('themeLbl');

// ─── Theme ──────────────────────────────────────────
function loadTheme() {
  const t = localStorage.getItem('dt_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', t);
  themeIcon.className = t === 'dark' ? 'ph ph-moon' : 'ph ph-sun';
  themeLbl.textContent = t === 'dark' ? 'DARK' : 'LIGHT';
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('dt_theme', next);
  themeIcon.className = next === 'dark' ? 'ph ph-moon' : 'ph ph-sun';
  themeLbl.textContent = next === 'dark' ? 'DARK' : 'LIGHT';
  // Update WebGL shader
  setTimeout(() => {
    if (window._shaderMat && window.THREE) {
      const s = getComputedStyle(document.documentElement);
      window._shaderMat.uniforms.u_colorCore.value = new THREE.Color(s.getPropertyValue('--shader-core').trim());
      window._shaderMat.uniforms.u_colorFringe.value = new THREE.Color(s.getPropertyValue('--shader-fringe').trim());
      window._shaderMat.uniforms.u_isLightMode.value = next === 'light' ? 1.0 : 0.0;
    }
  }, 60);
}

// ─── Persona API ────────────────────────────────────
async function fetchPersonas() {
  try {
    const r = await fetch('/api/personas');
    const d = await r.json();
    window.availablePersonas = d.personas || [];
  } catch { window.availablePersonas = []; }
}

// ─── Session Management ─────────────────────────────
function createSession(type, name, personaId) {
  const s = {
    id: Date.now(),
    type, // 'persona' | 'tutor'
    name,
    persona_id: personaId || null,
    session_id: (type === 'tutor' ? 'tutor-' : 'session_') + Date.now(),
  };
  sessions.push(s);
  persist();
  return s;
}

function deleteSession(id) {
  sessions = sessions.filter(s => s.id !== id);
  delete messages[id];
  if (activeSession && activeSession.id === id) { activeSession = null; showWelcome(); }
  persist();
  renderSidebar();
}

// ─── Render: Sidebar ────────────────────────────────
function renderSidebar() {
  sbList.innerHTML = '';
  sessions.forEach(s => {
    const el = document.createElement('div');
    el.className = 'sb-item' + (activeSession && activeSession.id === s.id ? ' active' : '');
    const isTutor = s.type === 'tutor';
    el.innerHTML = `
      <div class="sb-icon ${isTutor ? 'tutor' : 'persona'}">
        <i class="ph ${isTutor ? 'ph-book-open-text' : 'ph-user'}"></i>
      </div>
      <div class="sb-info">
        <div class="sb-name">${esc(s.name)}</div>
        <div class="sb-meta">${isTutor ? '课程助教' : '数字分身'}</div>
      </div>
      <button class="sb-del" title="删除"><i class="ph ph-x"></i></button>`;
    el.querySelector('.sb-del').addEventListener('click', e => { e.stopPropagation(); deleteSession(s.id); });
    el.addEventListener('click', () => { activeSession = s; enterChat(); renderSidebar(); closeMobileSidebar(); });
    sbList.appendChild(el);
  });
}

// ─── Render: Welcome Cards ──────────────────────────
function renderWelcomeCards() {
  wcGrid.innerHTML = '';
  window.availablePersonas.forEach(p => {
    const c = document.createElement('div');
    c.className = 'wc-card';
    c.innerHTML = `<div class="wc-card-name">${esc(p.name)}</div><div class="wc-card-meta">${p.doc_count || 0} 条记录</div>`;
    c.addEventListener('click', () => {
      activeSession = createSession('persona', p.name, p.id);
      enterChat(); renderSidebar();
    });
    wcGrid.appendChild(c);
  });
  // Tutor card
  const tc = document.createElement('div');
  tc.className = 'wc-card';
  tc.innerHTML = '<div class="wc-card-name">数字助教</div><div class="wc-card-meta">数据库课程</div>';
  tc.addEventListener('click', () => {
    activeSession = createSession('tutor', '数字助教');
    enterChat(); renderSidebar();
  });
  wcGrid.appendChild(tc);
}

// ─── View Switching ─────────────────────────────────
function showWelcome() {
  viewWelcome.classList.remove('hidden');
  viewChat.classList.add('hidden');
  activeSession = null;
  renderSidebar();
}

function enterChat() {
  viewWelcome.classList.add('hidden');
  viewChat.classList.remove('hidden');
  const s = activeSession;
  chatName.textContent = s.name;
  chatBadge.textContent = s.type === 'tutor' ? 'TUTOR' : 'PERSONA';
  if (!messages[s.id]) messages[s.id] = [];
  renderMessages();
  setTimeout(() => chatInput.focus(), 80);
}

// ─── Render: Messages ───────────────────────────────
function renderMessages() {
  if (!activeSession) return;
  const id = activeSession.id;
  const isTutor = activeSession.type === 'tutor';
  chatMessages.innerHTML = '';

  (messages[id] || []).forEach(m => {
    const row = document.createElement('div');
    row.className = 'msg ' + m.role + (isTutor && m.role === 'bot' ? ' tutor-msg' : '');

    if (m.role === 'bot') {
      const av = document.createElement('div');
      av.className = 'msg-avatar';
      av.innerHTML = isTutor ? '<i class="ph ph-book-open-text"></i>' : '<i class="ph ph-user"></i>';
      row.appendChild(av);
    }

    const wrap = document.createElement('div');
    wrap.className = 'msg-wrapper';

    const bub = document.createElement('div');
    bub.className = 'bubble' + (m.role === 'bot' && isTutor ? ' md' : '');

    if (m.type === 'file') {
      bub.innerHTML = fileHTML(m);
    } else if (m.role === 'bot' && isTutor && m.text) {
      bub.innerHTML = mdRender(m.text);
      bub.querySelectorAll('pre code').forEach(b => { try { hljs.highlightElement(b); } catch {} });
    } else if (m.typing) {
      bub.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
    } else {
      bub.textContent = m.text || '';
    }

    wrap.appendChild(bub);

    // Sources (tutor)
    if (m.sources && m.sources.length) {
      const sd = document.createElement('div');
      sd.className = 'sources';
      sd.innerHTML = '<div class="sources-title">REFERENCES</div>';
      m.sources.forEach(src => {
        const si = document.createElement('div');
        si.className = 'source-item';
        si.textContent = (src.chapter || '') + (src.page ? ' - P' + src.page : '') || '未知来源';
        sd.appendChild(si);
      });
      wrap.appendChild(sd);
    }

    row.appendChild(wrap);
    chatMessages.appendChild(row);
  });

  chatScroll.scrollTop = chatScroll.scrollHeight;
}

// ─── Markdown ───────────────────────────────────────
function mdRender(text) {
  try { return marked.parse(text); } catch { return text; }
}

// ─── Send: Dispatch ─────────────────────────────────
async function handleSend() {
  const raw = chatInput.value.trim();
  if (!raw || !activeSession) return;

  // File upload
  if (raw.startsWith('[文件] ') && fileInput.files.length > 0) {
    await sendFile(fileInput);
    chatInput.value = '';
    autoResize(chatInput);
    return;
  }

  const id = activeSession.id;
  if (!messages[id]) messages[id] = [];
  messages[id].push({ role: 'user', text: raw, ts: Date.now() });
  renderMessages(); persist();
  chatInput.value = '';
  autoResize(chatInput);

  if (activeSession.type === 'tutor') {
    await sendTutor(raw);
  } else {
    await sendPersona(raw);
  }
}

// ─── Send: Persona (JSON) ───────────────────────────
async function sendPersona(text) {
  const id = activeSession.id;
  messages[id].push({ role: 'bot', typing: true, ts: Date.now() });
  renderMessages();

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: activeSession.session_id, persona_id: activeSession.persona_id }),
    });
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    if (d.status === 'error') throw new Error(d.error);
    messages[id].pop();
    messages[id].push({ role: 'bot', text: d.reply, ts: Date.now() });
  } catch (e) {
    messages[id].pop();
    messages[id].push({ role: 'bot', text: '请求失败: ' + e.message, ts: Date.now() });
  }
  renderMessages(); persist();
}

// ─── Send: Tutor (Streaming SSE) ────────────────────
async function sendTutor(text) {
  const id = activeSession.id;
  const botMsg = { role: 'bot', text: '', ts: Date.now(), sources: null };
  messages[id].push(botMsg);
  renderMessages();

  // Get DOM refs for live update
  const lastRow = chatMessages.lastElementChild;
  const bubble = lastRow && lastRow.querySelector('.bubble');
  const wrap = lastRow && lastRow.querySelector('.msg-wrapper');
  if (bubble) bubble.classList.add('cursor-blink');

  let fullText = '';
  let sources = null;

  try {
    const r = await fetch('/tutor/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: activeSession.session_id, stream: true }),
    });

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6).trim());
          if (d.type === 'token') {
            fullText += d.content;
            if (bubble) { bubble.innerHTML = mdRender(fullText); chatScroll.scrollTop = chatScroll.scrollHeight; }
          } else if (d.type === 'sources') {
            sources = d.sources;
          } else if (d.error) {
            fullText += '\n\n⚠️ ' + d.error;
            if (bubble) bubble.innerHTML = mdRender(fullText);
          }
        } catch {}
      }
    }

    if (bubble) {
      bubble.classList.remove('cursor-blink');
      bubble.innerHTML = mdRender(fullText);
      bubble.querySelectorAll('pre code').forEach(b => { try { hljs.highlightElement(b); } catch {} });
    }

    // Append sources
    if (sources && sources.length && wrap) {
      const sd = document.createElement('div');
      sd.className = 'sources';
      sd.innerHTML = '<div class="sources-title">REFERENCES</div>';
      sources.forEach(s => {
        const si = document.createElement('div');
        si.className = 'source-item';
        si.textContent = (s.chapter || '') + (s.page ? ' - P' + s.page : '') || '未知来源';
        sd.appendChild(si);
      });
      wrap.appendChild(sd);
    }

    botMsg.text = fullText;
    botMsg.sources = sources;
  } catch (e) {
    if (bubble) { bubble.classList.remove('cursor-blink'); }
    botMsg.text = '请求失败: ' + e.message;
    if (bubble) bubble.innerHTML = mdRender(botMsg.text);
  }
  persist();
}

// ─── Welcome Input ──────────────────────────────────
async function handleWelcomeSend() {
  const text = wcInput.value.trim();
  if (!text) return;
  if (sessions.length === 0) { openModal(); return; }
  activeSession = sessions[0];
  enterChat(); renderSidebar();
  chatInput.value = text;
  wcInput.value = '';
  await handleSend();
}

// ─── Persona Modal ──────────────────────────────────
function openModal() {
  renderPersonaList();
  personaModal.classList.remove('hidden');
}
function closeModal() { personaModal.classList.add('hidden'); }

function renderPersonaList() {
  personaList.innerHTML = '';
  if (!window.availablePersonas.length) {
    personaList.innerHTML = '<div style="text-align:center;color:var(--text-2);padding:20px;font-size:12px">暂无分身，请先导入数据</div>';
    return;
  }
  window.availablePersonas.forEach(p => {
    const el = document.createElement('div');
    el.className = 'p-item';
    el.innerHTML = `
      <div class="p-item-icon"><i class="ph ph-user"></i></div>
      <div class="p-item-info">
        <div class="p-item-name">${esc(p.name)}</div>
        <div class="p-item-meta">${esc(p.collection)} · ${p.doc_count || 0} 条记录</div>
      </div>
      <button class="p-item-del" title="删除"><i class="ph ph-trash"></i></button>`;
    el.querySelector('.p-item-del').addEventListener('click', async e => {
      e.stopPropagation();
      if (confirm('确定删除分身「' + p.name + '」？')) await deletePersona(p.id);
    });
    el.addEventListener('click', () => {
      activeSession = createSession('persona', p.name, p.id);
      closeModal(); enterChat(); renderSidebar();
    });
    personaList.appendChild(el);
  });
}

async function deletePersona(pid) {
  try {
    const r = await fetch('/api/personas/' + pid, { method: 'DELETE' });
    const d = await r.json();
    if (d.status === 'success') {
      sessions = sessions.filter(s => s.persona_id !== pid);
      persist();
      await fetchPersonas();
      renderSidebar(); renderPersonaList(); renderWelcomeCards();
      toast('分身已删除');
    }
  } catch (e) { toast('删除失败'); }
}

// ─── Clear History ──────────────────────────────────
async function clearHistory() {
  if (!activeSession) return;
  if (!confirm('确定清除当前对话的所有历史消息？')) return;
  try {
    const ep = activeSession.type === 'tutor' ? '/tutor/reset' : '/reset';
    await fetch(ep, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: activeSession.session_id }) });
    messages[activeSession.id] = [];
    persist(); renderMessages();
    toast('历史已清除');
  } catch { toast('清除失败'); }
}

// ─── Export ─────────────────────────────────────────
function exportChat() {
  if (!activeSession) {
    dl(JSON.stringify({ sessions, messages }, null, 2), 'chat_all.json', 'application/json');
    return;
  }
  const id = activeSession.id;
  dl(JSON.stringify({ session: activeSession, messages: messages[id] || [] }, null, 2),
    'chat_' + activeSession.name + '.json', 'application/json');
}

function dl(content, name, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
}

// ─── File Upload ────────────────────────────────────
function onFileChange() {
  const f = fileInput.files && fileInput.files[0];
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) { toast('文件不能超过 10MB'); fileInput.value = ''; return; }
  chatInput.value = '[文件] ' + f.name;
  if (f.type.startsWith('image/')) {
    const r = new FileReader(); r.onload = e => { fileInput._img = e.target.result; }; r.readAsDataURL(f);
  } else if (f.type.startsWith('text/') || ['application/json','application/javascript','application/xml'].includes(f.type)) {
    const r = new FileReader(); r.onload = e => { fileInput._txt = e.target.result; }; r.readAsText(f);
  }
}

async function sendFile(fEl) {
  const f = fEl.files && fEl.files[0];
  if (!f || !activeSession) return;
  const id = activeSession.id;
  if (!messages[id]) messages[id] = [];

  const fm = { role: 'user', type: 'file', filename: f.name, filetype: f.type, filesize: f.size, ts: Date.now() };
  if (f.type.startsWith('image/') && fEl._img) fm.data = fEl._img;
  else if (fEl._txt) fm.content = fEl._txt;

  messages[id].push(fm);
  renderMessages(); persist();
  fEl.value = ''; delete fEl._img; delete fEl._txt;

  messages[id].push({ role: 'bot', typing: true, ts: Date.now() });
  renderMessages();

  try {
    const desc = fm.content
      ? '用户上传了文件: ' + f.name + '\n文件内容:\n' + fm.content.substring(0, 1000)
      : '用户上传了文件: ' + f.name + ' (' + f.type + ', ' + fmtSize(f.size) + ')';
    const body = activeSession.type === 'tutor'
      ? { message: desc, session_id: activeSession.session_id }
      : { message: desc, session_id: activeSession.session_id, persona_id: activeSession.persona_id };
    const ep = activeSession.type === 'tutor' ? '/tutor/chat' : '/chat';
    const r = await fetch(ep, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json();
    messages[id].pop();
    messages[id].push({ role: 'bot', text: d.reply || d.error || '完成', ts: Date.now() });
  } catch (e) {
    messages[id].pop();
    messages[id].push({ role: 'bot', text: '处理失败: ' + e.message, ts: Date.now() });
  }
  renderMessages(); persist();
}

function fmtSize(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function fileHTML(m) {
  const { filename, filetype, filesize, data, content } = m;
  let icon = '<i class="ph ph-file"></i>';
  if (filetype && filetype.startsWith('image/')) icon = '<i class="ph ph-image"></i>';
  else if (filetype && filetype.includes('pdf')) icon = '<i class="ph ph-file-pdf"></i>';
  let h = `<div class="file-msg"><div class="file-info"><div class="file-icon-box">${icon}</div><div><div class="file-name">${esc(filename)}</div><div class="file-meta">${fmtSize(filesize)}</div></div></div>`;
  if (filetype && filetype.startsWith('image/') && data) h += `<div class="file-preview"><img src="${data}" alt="${esc(filename)}" class="preview-img"></div>`;
  else if (content) { const t = content.length > 400 ? content.substring(0, 400) + '...' : content; h += `<div class="file-preview"><pre class="preview-text">${esc(t)}</pre></div>`; }
  return h + '</div>';
}

// ─── Voice Recognition ──────────────────────────────
let recog = null, isRec = false;
const speechLang = localStorage.getItem('dt_speech') || 'zh-CN';

function toggleMic() {
  if (isRec) { if (recog) try { recog.stop(); } catch {} return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { toast('浏览器不支持语音识别'); return; }
  if (!recog) {
    recog = new SR(); recog.interimResults = false; recog.maxAlternatives = 1;
    recog.onstart = () => { isRec = true; micBtn.classList.add('recording'); };
    recog.onend = () => { isRec = false; micBtn.classList.remove('recording'); };
    recog.onerror = () => { isRec = false; micBtn.classList.remove('recording'); };
    recog.onresult = e => {
      const t = e.results[0]?.[0]?.transcript || '';
      const target = activeSession ? chatInput : wcInput;
      target.value = target.value ? target.value + ' ' + t : t;
    };
  }
  recog.lang = speechLang;
  try { recog.start(); } catch {}
}

// ─── Mobile Sidebar ─────────────────────────────────
function openMobileSidebar() { sidebar.classList.add('mobile-open'); sbOverlay.classList.add('open'); }
function closeMobileSidebar() { sidebar.classList.remove('mobile-open'); sbOverlay.classList.remove('open'); }

// ─── Toast ──────────────────────────────────────────
function toast(msg) {
  const t = document.createElement('div'); t.className = 'toast'; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 2000);
}

// ─── Persistence ────────────────────────────────────
function persist() {
  localStorage.setItem('dt_sessions', JSON.stringify(sessions));
  localStorage.setItem('dt_messages', JSON.stringify(messages));
}

// ─── Utils ──────────────────────────────────────────
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }

// ─── WebGL Shader Background ────────────────────────
function initWebGL() {
  if (!window.THREE) return;
  const container = $('webglBg');
  const scene = new THREE.Scene();
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  cam.position.z = 1;

  const ren = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  ren.setSize(window.innerWidth, window.innerHeight);
  ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(ren.domElement);

  const vs = `varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position,1.0);}`;
  const fs = `
    uniform float u_time; uniform vec2 u_res; uniform vec2 u_mouse;
    uniform vec3 u_colorCore; uniform vec3 u_colorFringe; uniform float u_isLightMode;
    varying vec2 vUv;
    vec2 hash(vec2 p){p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));return -1.0+2.0*fract(sin(p)*43758.5453123);}
    float noise(vec2 p){const float K1=0.366025404;const float K2=0.211324865;vec2 i=floor(p+(p.x+p.y)*K1);vec2 a=p-i+(i.x+i.y)*K2;vec2 o=(a.x>a.y)?vec2(1,0):vec2(0,1);vec2 b=a-o+K2;vec2 c=a-1.0+2.0*K2;vec3 h=max(0.5-vec3(dot(a,a),dot(b,b),dot(c,c)),0.0);vec3 n=h*h*h*h*vec3(dot(a,hash(i)),dot(b,hash(i+o)),dot(c,hash(i+1.0)));return dot(n,vec3(70.0));}
    float sdArc(vec2 p,vec2 ctr,float rad,float w,float warp){p.y+=sin(p.x*3.0+u_time*0.5)*warp;p.x+=noise(p*2.0+u_time*0.2)*(warp*0.5);return abs(length(p-ctr)-rad)-w;}
    void main(){
      vec2 uv=gl_FragCoord.xy/u_res;vec2 st=uv;st.x*=u_res.x/u_res.y;
      st+=(u_mouse-0.5)*0.1;
      vec2 ctr=vec2(0.15,0.5);
      float d1=sdArc(st,ctr,0.8,0.01,0.1);
      float d2=sdArc(st,ctr,0.82,0.04,0.15);
      float core=exp(-d1*40.0);float fringe=exp(-d2*15.0);
      float wash=smoothstep(1.0,-0.2,st.x)*0.2;
      vec3 col=u_colorCore*core+u_colorFringe*fringe+u_colorFringe*wash*(sin(u_time)*0.1+0.9);
      float a=clamp(core+fringe+wash,0.0,1.0);
      col=vec3(1.0)-exp(-col*2.0);
      if(u_isLightMode>0.5) a=clamp(core*1.5+fringe+wash*0.5,0.0,0.5);
      gl_FragColor=vec4(col,a);
    }`;

  const cs = getComputedStyle(document.documentElement);
  const mat = new THREE.ShaderMaterial({
    vertexShader: vs, fragmentShader: fs,
    uniforms: {
      u_time: { value: 0 },
      u_res: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
      u_mouse: { value: new THREE.Vector2(0.5, 0.5) },
      u_colorCore: { value: new THREE.Color(cs.getPropertyValue('--shader-core').trim()) },
      u_colorFringe: { value: new THREE.Color(cs.getPropertyValue('--shader-fringe').trim()) },
      u_isLightMode: { value: document.documentElement.getAttribute('data-theme') === 'light' ? 1 : 0 },
    },
    transparent: true, blending: THREE.NormalBlending,
  });
  window._shaderMat = mat;

  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

  let mouse = new THREE.Vector2(0.5, 0.5);
  document.addEventListener('mousemove', e => { mouse.x = e.clientX / window.innerWidth; mouse.y = 1 - e.clientY / window.innerHeight; });

  const clock = new THREE.Clock();
  (function loop() {
    requestAnimationFrame(loop);
    mat.uniforms.u_time.value = clock.getElapsedTime();
    mat.uniforms.u_mouse.value.lerp(mouse, 0.05);
    ren.render(scene, cam);
  })();

  window.addEventListener('resize', () => {
    ren.setSize(window.innerWidth, window.innerHeight);
    mat.uniforms.u_res.value.set(window.innerWidth, window.innerHeight);
  });
}

// ─── Marked Config ──────────────────────────────────
if (window.marked) {
  marked.setOptions({
    highlight: (code, lang) => {
      try { return lang && hljs.getLanguage(lang) ? hljs.highlight(code, { language: lang }).value : hljs.highlightAuto(code).value; } catch { return code; }
    },
    breaks: true, gfm: true,
  });
}

// ─── Event Bindings ─────────────────────────────────
themeBtn.addEventListener('click', toggleTheme);
chatSendBtn.addEventListener('click', handleSend);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } });
chatInput.addEventListener('input', () => autoResize(chatInput));
wcSend.addEventListener('click', handleWelcomeSend);
wcInput.addEventListener('keydown', e => { if (e.key === 'Enter') handleWelcomeSend(); });
btnNewPersona.addEventListener('click', openModal);
btnNewTutor.addEventListener('click', () => {
  activeSession = createSession('tutor', '数字助教');
  enterChat(); renderSidebar(); closeMobileSidebar();
});
modalCloseBtn.addEventListener('click', closeModal);
document.querySelector('.modal-backdrop').addEventListener('click', closeModal);
clearBtn.addEventListener('click', clearHistory);
exportBtn.addEventListener('click', exportChat);
micBtn.addEventListener('click', toggleMic);
fileInput.addEventListener('change', onFileChange);
menuBtn.addEventListener('click', openMobileSidebar);
sbOverlay.addEventListener('click', closeMobileSidebar);

// ─── Init ───────────────────────────────────────────
(async function init() {
  loadTheme();
  initWebGL();
  await fetchPersonas();
  renderSidebar();
  renderWelcomeCards();
})();
