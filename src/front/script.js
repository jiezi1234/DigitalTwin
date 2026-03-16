// ---------- 数据与初始配置 ----------
// bot 对象结构: { id, name, persona_id, session_id }
let storedBots = JSON.parse(localStorage.getItem('bots_v2')) || [];
let bots = storedBots;
let selectedBot = null;
let messages = JSON.parse(localStorage.getItem('messages_v1')) || {}; // keyed by bot id

// 可用分身列表（从后端获取）
window.availablePersonas = [];

// ---------- DOM 元素 ----------
const botsList = document.getElementById('botsList');
const botCount = document.getElementById('botCount');
const welcomeCard = document.getElementById('welcomeCard');
const suggestionsWrap = document.getElementById('suggestions');
const centerInput = document.getElementById('centerInput');
const centerSend = document.getElementById('centerSend');
const centerFile = document.getElementById('centerFile');
const chatView = document.getElementById('chatView');
const messagesEl = document.getElementById('messages');
const inputDock = document.getElementById('inputDock');
const dockInput = document.getElementById('dockInput');
const dockSend = document.getElementById('dockSend');
const dockFile = document.getElementById('dockFile');
const contentScroll = document.getElementById('contentScroll');

const exportJsonBtn = document.getElementById('exportJsonBtn');
const exportTxtBtn = document.getElementById('exportTxtBtn');
const themeToggle = document.getElementById('themeToggle');
const langToggle = document.getElementById('langToggle');
const newChatBtn = document.getElementById('newChatBtn');
const centerMic = document.getElementById('centerMic');
const dockMic = document.getElementById('dockMic');

const personaModal = document.getElementById('personaModal');
const personaList = document.getElementById('personaList');
const modalClose = document.getElementById('modalClose');

// ---------- 示例问题 ----------
const examplePrompts = [];

// ---------- 语音识别 状态 ----------
let recognition = null;
let isRecording = false;
let speechLang = localStorage.getItem('speech_lang') || 'zh-CN';

// ---------- 分身 API ----------
async function fetchPersonas() {
  try {
    const resp = await fetch('/api/personas');
    const data = await resp.json();
    window.availablePersonas = data.personas || [];
  } catch (e) {
    console.error('获取分身列表失败:', e);
    window.availablePersonas = [];
  }
}

function renderPersonaModal() {
  personaList.innerHTML = '';
  if (window.availablePersonas.length === 0) {
    personaList.innerHTML = '<div style="color:var(--muted);font-size:13px;text-align:center;padding:16px">暂无分身，请先运行 test_csv_final.py 导入数据</div>';
    return;
  }
  window.availablePersonas.forEach(p => {
    const item = document.createElement('div');
    item.className = 'persona-item';
    item.innerHTML = `
      <div class="persona-item-info">
        <div class="persona-item-name">${p.name}</div>
        <div class="persona-item-meta">集合: ${p.collection} · ${p.doc_count} 条记录</div>
      </div>
      <button class="persona-item-del" title="删除分身记录" data-id="${p.id}">🗑️</button>`;

    item.querySelector('.persona-item-del').addEventListener('click', async (e) => {
      e.stopPropagation();
      if (confirm(`确定删除分身「${p.name}」的记录？（ChromaDB 数据不会删除）`)) {
        await deletePersona(p.id);
      }
    });

    item.addEventListener('click', () => {
      // 创建 bot 并关联此分身
      const newBot = {
        id: Date.now(),
        name: p.name,
        persona_id: p.id,
        session_id: `session_${Date.now()}`,
      };
      bots.push(newBot);
      persist();
      closePersonaModal();
      selectedBot = newBot;
      renderBots();
      enterChatView();
    });

    personaList.appendChild(item);
  });
}

async function deletePersona(personaId) {
  try {
    const resp = await fetch(`/api/personas/${personaId}`, { method: 'DELETE' });
    const data = await resp.json();
    if (data.status === 'success') {
      // 从 bots 中移除关联此分身的 bot
      bots = bots.filter(b => b.persona_id !== personaId);
      persist();
      await fetchPersonas();
      renderBots();
      renderPersonaModal();
      showToast('分身已删除');
    } else {
      showToast('删除失败: ' + (data.error || '未知错误'));
    }
  } catch (e) {
    showToast('删除失败: ' + e.message);
  }
}

function openPersonaModal() {
  renderPersonaModal();
  personaModal.classList.remove('hidden');
}

function closePersonaModal() {
  personaModal.classList.add('hidden');
}

// ---------- 渲染侧边栏机器人 ----------
function renderBots(){
  botsList.innerHTML = '';
  bots.forEach(bot => {
    const el = document.createElement('div');
    el.className = 'bot-item' + (selectedBot && selectedBot.id === bot.id ? ' active' : '');

    // 找到分身名称
    const persona = window.availablePersonas.find(p => p.id === bot.persona_id);
    const personaLabel = persona ? `<div style="font-size:11px;color:var(--muted)">${persona.name}</div>` : '';

    el.innerHTML = `
      <div class="avatar">🤖</div>
      <div style="flex:1;min-width:0">
        <div style="font-weight:600">${bot.name}</div>
        ${personaLabel}
      </div>
      <button class="bot-item-del" title="删除会话" data-id="${bot.id}">✕</button>`;

    el.querySelector('.bot-item-del').addEventListener('click', (e) => {
      e.stopPropagation();
      bots = bots.filter(b => b.id !== bot.id);
      if (selectedBot && selectedBot.id === bot.id) {
        selectedBot = null;
        exitChatView();
      }
      persist();
      renderBots();
    });

    el.addEventListener('click', () => {
      selectedBot = bot;
      enterChatView();
      renderBots();
    });
    botsList.appendChild(el);
  });
  botCount.textContent = bots.length;
}

// ---------- 渲染示例问题按钮 ----------
function renderSuggestions(){
  suggestionsWrap.innerHTML = '';
  examplePrompts.forEach(p => {
    const btn = document.createElement('button');
    btn.textContent = p;
    btn.addEventListener('click', ()=>{ centerInput.value = p; });
    suggestionsWrap.appendChild(btn);
  });
}

// ---------- 聊天视图相关 ----------
function enterChatView(){
  welcomeCard.classList.add('hidden');
  chatView.classList.remove('hidden');
  inputDock.classList.remove('hidden');
  const id = selectedBot.id;
  if(!messages[id]) messages[id] = [];
  renderMessages();
  setTimeout(()=> dockInput.focus(), 120);
}

function exitChatView(){
  welcomeCard.classList.remove('hidden');
  chatView.classList.add('hidden');
  inputDock.classList.add('hidden');
}

// ---------- 渲染消息 ----------
function renderMessages(){
  const id = selectedBot.id;
  messagesEl.innerHTML = '';
  (messages[id] || []).forEach(m => {
    const div = document.createElement('div');
    div.className = 'msg ' + (m.role === 'user' ? 'user' : 'bot');

    if(m.role === 'bot'){
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = '🤖';
      div.appendChild(avatar);
    }

    const b = document.createElement('div');
    b.className = 'bubble';

    if (m.type === 'file') {
      b.innerHTML = createFileMessageHTML(m);
    } else {
      b.textContent = m.text;
    }

    div.appendChild(b);
    messagesEl.appendChild(div);
  });
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// 创建文件消息的HTML
function createFileMessageHTML(fileMessage) {
  const { filename, filetype, filesize, data, content } = fileMessage;
  const formattedSize = formatFileSize(filesize);

  let fileIcon = '📄';
  if (filetype.startsWith('image/')) fileIcon = '🖼️';
  else if (filetype.startsWith('audio/')) fileIcon = '🎵';
  else if (filetype.startsWith('video/')) fileIcon = '🎥';
  else if (filetype.includes('pdf')) fileIcon = '📄';
  else if (filetype.includes('word') || filetype.includes('doc')) fileIcon = '📝';
  else if (filetype.includes('excel') || filetype.includes('spreadsheet')) fileIcon = '📊';
  else if (filetype.includes('powerpoint') || filetype.includes('presentation')) fileIcon = '📋';
  else if (filetype.includes('zip') || filetype.includes('archive')) fileIcon = '🗜️';

  let sizeIcon = filesize < 1024 ? '📄' : filesize < 1048576 ? '📁' : '📦';

  let html = `
    <div class="file-message">
      <div class="file-info">
        <div class="file-icon">${fileIcon}</div>
        <div class="file-details">
          <div class="file-name"><span>${filename}</span></div>
          <div class="file-meta"><span>${sizeIcon} ${formattedSize}</span></div>
        </div>
      </div>`;

  if (filetype.startsWith('image/') && data) {
    html += `<div class="file-preview"><img src="${data}" alt="${filename}" class="preview-image"></div>`;
  } else if (content && (filetype.startsWith('text/') ||
      filetype === 'application/json' ||
      filetype === 'application/javascript' ||
      filetype === 'application/xml')) {
    const previewContent = content.length > 500 ? content.substring(0, 500) + '...' : content;
    html += `<div class="file-preview"><pre class="preview-text">${previewContent}</pre></div>`;
  }

  html += `
      <div class="file-action">
        <button class="file-download" onclick="downloadFile('${filename}', '${filetype}', '${data || content || ''}')">📥 下载</button>
      </div>
    </div>`;

  return html;
}

// 下载文件
function downloadFile(filename, filetype, data) {
  if (!data) { alert('无法下载文件，没有文件数据'); return; }
  try {
    let blob;
    if (data.startsWith('data:')) {
      const byteString = atob(data.split(',')[1]);
      const mimeString = data.split(',')[0].split(':')[1].split(';')[0];
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
      blob = new Blob([ab], { type: mimeString });
    } else {
      blob = new Blob([data], { type: filetype });
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('下载文件失败:', error);
    alert('下载文件失败: ' + error.message);
  }
}

// ---------- 导出功能 ----------
function exportCurrentAsJSON(){
  if(!selectedBot){
    const data = { bots, messages };
    downloadBlob(JSON.stringify(data, null, 2), 'chat_export_all.json', 'application/json');
    return;
  }
  const id = selectedBot.id;
  const payload = { bot: selectedBot, messages: messages[id] || [] };
  downloadBlob(JSON.stringify(payload, null, 2), `chat_${selectedBot.name || id}.json`, 'application/json');
}

function exportCurrentAsTxt(){
  if(!selectedBot){
    let txt = '';
    for(const b of bots){
      txt += `=== ${b.name} ===\n`;
      (messages[b.id] || []).forEach(m=> txt += `[${m.role}] ${m.text}\n`);
      txt += '\n';
    }
    downloadBlob(txt, 'chat_export_all.txt', 'text/plain');
    return;
  }
  const id = selectedBot.id;
  const msgs = messages[id] || [];
  let txt = `=== ${selectedBot.name} ===\n`;
  msgs.forEach(m=> txt += `[${m.role}] ${m.text}\n`);
  downloadBlob(txt, `chat_${selectedBot.name || id}.txt`, 'text/plain');
}

function downloadBlob(content, filename, mime){
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

// ---------- 主题切换 ----------
function loadTheme(){
  const t = localStorage.getItem('ui_theme') || 'light';
  if(t === 'dark') document.documentElement.classList.add('dark');
  themeToggle.textContent = t === 'dark' ? '☀️' : '🌙';
}
function toggleTheme(){
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('ui_theme', isDark ? 'dark' : 'light');
  themeToggle.textContent = isDark ? '☀️' : '🌙';
}

// ---------- 语音识别（Web Speech API） ----------
function ensureRecognition(){
  if(recognition) return true;
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SpeechRecognition) return false;
  recognition = new SpeechRecognition();
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;
  recognition.lang = speechLang;

  recognition.onstart = ()=>{ isRecording = true; updateMicUI(true); };
  recognition.onend = ()=>{ isRecording = false; updateMicUI(false); };
  recognition.onerror = (e)=>{
    console.error('Speech recognition error', e);
    isRecording = false; updateMicUI(false);
    alert('语音识别出错：' + (e.error || e.message));
  };
  recognition.onresult = (evt)=>{
    const text = (evt.results[0] && evt.results[0][0] && evt.results[0][0].transcript) || '';
    if(!selectedBot){
      centerInput.value = centerInput.value ? centerInput.value + ' ' + text : text;
    } else {
      dockInput.value = dockInput.value ? dockInput.value + ' ' + text : text;
    }
  };
  return true;
}

function startRecognition(){
  if(!ensureRecognition()){
    alert('抱歉，你的浏览器不支持 Web Speech API（语音识别）。建议使用 Chrome/Edge 等支持的浏览器。');
    return;
  }
  recognition.lang = speechLang;
  try{ recognition.start(); }catch(e){ console.warn('recognition start err', e); }
}
function stopRecognition(){
  if(recognition){ try{ recognition.stop(); }catch(e){} }
}

function toggleMic(target){
  if(isRecording){ stopRecognition(); return; }
  if(ensureRecognition()){
    if(target === 'center') centerInput.focus(); else dockInput.focus();
    startRecognition();
  }
}

function updateMicUI(recording){
  [centerMic, dockMic].forEach(btn => {
    if(recording) btn.classList.add('recording'); else btn.classList.remove('recording');
  });
}

// ---------- 文件上传处理 ----------
function handleFileUpload(fileInput, targetInput) {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  const maxSize = 10 * 1024 * 1024;
  if (file.size > maxSize) {
    alert(`文件太大，请选择小于10MB的文件。当前文件大小: ${formatFileSize(file.size)}`);
    fileInput.value = '';
    return;
  }

  targetInput.value = `[文件] ${file.name}`;

  if (file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = e => { fileInput._imageData = e.target.result; };
    reader.readAsDataURL(file);
  } else if (file.type.startsWith('text/') ||
             file.type === 'application/json' ||
             file.type === 'application/javascript' ||
             file.type === 'application/xml' ||
             file.type === 'application/pdf' ||
             file.type === 'application/msword' ||
             file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    const reader = new FileReader();
    reader.onload = e => { fileInput._fileContent = e.target.result; };
    reader.readAsText(file);
  }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
  else return (bytes / 1048576).toFixed(2) + ' MB';
}

async function sendFileMessage(fileInput) {
  const file = fileInput.files && fileInput.files[0];
  if (!file || !selectedBot) return;

  const id = selectedBot.id;
  if (!messages[id]) messages[id] = [];

  const fileMessage = {
    role: 'user', type: 'file',
    filename: file.name, filetype: file.type, filesize: file.size, ts: Date.now()
  };

  if (file.type.startsWith('image/') && fileInput._imageData) {
    fileMessage.data = fileInput._imageData;
  } else if ((file.type.startsWith('text/') ||
              file.type === 'application/json' ||
              file.type === 'application/javascript' ||
              file.type === 'application/xml') && fileInput._fileContent) {
    fileMessage.content = fileInput._fileContent;
  }

  messages[id].push(fileMessage);
  renderMessages();
  persist();
  fileInput.value = '';
  delete fileInput._imageData;
  delete fileInput._fileContent;

  const loadingMessage = { role: 'bot', text: '正在处理文件...', ts: Date.now() };
  messages[id].push(loadingMessage);
  renderMessages();

  try {
    const fileDescription = fileMessage.content
      ? `用户上传了文件: ${file.name}\n文件内容:\n${fileMessage.content.substring(0, 1000)}`
      : `用户上传了文件: ${file.name} (${file.type}, ${formatFileSize(file.size)})`;

    const requestData = {
      message: fileDescription,
      session_id: selectedBot.session_id || `session_${selectedBot.id}_${Date.now()}`,
      persona_id: selectedBot.persona_id,
    };

    const response = await fetch('/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
    const data = await response.json();
    if (data.status === 'error') throw new Error(data.error || 'API返回错误');
    if (!data || typeof data.reply !== 'string') throw new Error('API返回格式错误');

    messages[id].pop();
    messages[id].push({ role: 'bot', text: data.reply, ts: Date.now() });
  } catch (error) {
    console.error('发送文件消息到后端失败:', error);
    messages[id].pop();
    messages[id].push({ role: 'bot', text: `处理文件失败: ${error.message}`, ts: Date.now() });
  }

  renderMessages();
  persist();
}

// ---------- 发送与保存 ----------
async function handleCenterSend() {
  const text = centerInput.value.trim();
  if (!text) return;

  if (text.startsWith('[文件] ') && centerFile.files.length > 0) {
    if (!selectedBot) {
      if (bots.length === 0) { openPersonaModal(); return; }
      selectedBot = bots[0];
      renderBots();
    }
    enterChatView();
    await sendFileMessage(centerFile);
    centerInput.value = '';
    return;
  }

  if (!selectedBot) {
    if (bots.length === 0) { openPersonaModal(); return; }
    selectedBot = bots[0];
    renderBots();
  }
  enterChatView();
  dockInput.value = text;
  centerInput.value = '';
  await handleDockSend();
}

async function handleDockSend() {
  const text = dockInput.value.trim();

  if (text.startsWith('[文件] ') && dockFile.files.length > 0) {
    await sendFileMessage(dockFile);
    dockInput.value = '';
    return;
  }

  if (!text) return;

  const id = selectedBot.id;
  if (!messages[id]) messages[id] = [];
  messages[id].push({ role: 'user', text, ts: Date.now() });
  renderMessages();
  persist();
  dockInput.value = '';

  const loadingMessage = { role: 'bot', text: '正在思考...', ts: Date.now() };
  messages[id].push(loadingMessage);
  renderMessages();

  try {
    const reply = await sendMessageToBackend(selectedBot, text);
    messages[id].pop();
    messages[id].push({ role: 'bot', text: reply, ts: Date.now() });
  } catch (e) {
    messages[id].pop();
    messages[id].push({ role: 'bot', text: `网络连接失败：${e.message}`, ts: Date.now() });
  }

  renderMessages();
  persist();
}

// ---------- 实际调用后端API ----------
async function sendMessageToBackend(bot, message) {
  if(!bot.session_id) bot.session_id = `session_${bot.id}_${Date.now()}`;

  const requestData = {
    message,
    session_id: bot.session_id,
    persona_id: bot.persona_id,
  };

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
    const data = await response.json();
    if (data.status === 'error') throw new Error(data.error || 'API返回错误');
    if (!data || typeof data.reply !== 'string') throw new Error('API返回格式错误');
    return data.reply;

  } catch (error) {
    console.error('发送消息到后端失败:', error);
    throw error;
  }
}

// ---------- 事件绑定 ----------
centerSend.addEventListener('click', handleCenterSend);
centerInput.addEventListener('keyup', (e)=>{ if(e.key === 'Enter') handleCenterSend(); });

dockSend.addEventListener('click', handleDockSend);
dockInput.addEventListener('keyup', (e)=>{ if(e.key === 'Enter') handleDockSend(); });

centerFile.addEventListener('change', (e)=>{ handleFileUpload(e.target, centerInput); });
dockFile.addEventListener('change', (e)=>{ handleFileUpload(e.target, dockInput); });

exportJsonBtn.addEventListener('click', exportCurrentAsJSON);
exportTxtBtn.addEventListener('click', exportCurrentAsTxt);
themeToggle.addEventListener('click', toggleTheme);

const clearHistoryBtn = document.getElementById('clearHistoryBtn');
clearHistoryBtn.addEventListener('click', clearChatHistory);

function loadSpeechLang(){
  speechLang = localStorage.getItem('speech_lang') || 'zh-CN';
  langToggle.textContent = speechLang === 'zh-CN' ? '🌐 中' : '🌐 EN';
}
function toggleSpeechLang(){
  speechLang = speechLang === 'zh-CN' ? 'en-US' : 'zh-CN';
  localStorage.setItem('speech_lang', speechLang);
  loadSpeechLang();
  if(recognition){ recognition.lang = speechLang; }
}
langToggle.addEventListener('click', toggleSpeechLang);

newChatBtn.addEventListener('click', openPersonaModal);
modalClose.addEventListener('click', closePersonaModal);
document.querySelector('.modal-backdrop').addEventListener('click', closePersonaModal);

centerMic.addEventListener('click', ()=> toggleMic('center'));
dockMic.addEventListener('click', ()=> toggleMic('dock'));

// ---------- 清除历史消息功能 ----------
async function clearChatHistory() {
  if (!selectedBot) {
    alert('请先选择一个数字分身');
    return;
  }

  if (confirm('确定要清除当前对话的所有历史消息吗？此操作不可撤销。')) {
    try {
      const session_id = selectedBot.session_id || `session_${selectedBot.id}_${Date.now()}`;
      const response = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id })
      });

      if (!response.ok) throw new Error(`后端清除失败: ${response.status}`);

      messages[selectedBot.id] = [];
      persist();
      renderMessages();
      showToast('历史消息已清除');

    } catch (error) {
      console.error('清除历史消息失败:', error);
      showToast('清除失败: ' + error.message);
    }
  }
}

function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  document.body.appendChild(toast);
  Object.assign(toast.style, {
    position: 'fixed', bottom: '20px', left: '50%',
    transform: 'translateX(-50%)',
    backgroundColor: 'rgba(0, 0, 0, 0.7)', color: 'white',
    padding: '10px 20px', borderRadius: '5px',
    zIndex: '1000', transition: 'opacity 0.3s ease',
  });
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => document.body.removeChild(toast), 300);
  }, 2000);
}

// ---------- 本地保存 ----------
function persist(){
  localStorage.setItem('bots_v2', JSON.stringify(bots));
  localStorage.setItem('messages_v1', JSON.stringify(messages));
}

// ---------- 初始化 ----------
async function init(){
  loadTheme();
  loadSpeechLang();
  await fetchPersonas();
  renderBots();
  renderSuggestions();
  exitChatView();

  contentScroll.addEventListener('click', ()=>{ if(!selectedBot) centerInput.focus(); });
  window.addEventListener('beforeunload', ()=>{ if(recognition) try{ recognition.abort(); }catch(e){} });
}

init();
