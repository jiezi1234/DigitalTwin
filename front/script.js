// ---------- 数据与初始配置 ----------
const defaultBots = [
  { id: 1, name: '小任同学' }
];

// 获取localStorage数据并
let storedBots = JSON.parse(localStorage.getItem('bots_v1')) || [];
// 过滤掉可能存在的旧数据
storedBots = storedBots.filter(bot => bot.id !== 2 && bot.id !== 3 && bot.id !== 4);
// 确保机器人对象结构正确（不包含style属性）
storedBots = storedBots.map(bot => {
  const { style, ...rest } = bot;
  return rest;
});
let bots = storedBots.length > 0 ? storedBots : defaultBots;
let selectedBot = null;
let messages = JSON.parse(localStorage.getItem('messages_v1')) || {}; // keyed by bot id

// 初始化时删除可能存在的不需要的消息数据
delete messages[2];
delete messages[3];
delete messages[4];

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

// ---------- 示例问题 ----------
const examplePrompts = [];

// ---------- 语音识别 状态 ----------
let recognition = null;
let isRecording = false;
// 保存语言代码，默认 zh-CN（可通过 langToggle 切换）
let speechLang = localStorage.getItem('speech_lang') || 'zh-CN';

// ---------- 渲染侧边栏机器人 ----------
function renderBots(){
  botsList.innerHTML = '';
  bots.forEach(bot => {
    const el = document.createElement('div');
    el.className = 'bot-item' + (selectedBot && selectedBot.id === bot.id ? ' active' : '');
    el.innerHTML = `
      <div class="avatar">🤖</div>
      <div>
        <div style="font-weight:600">${bot.name}</div>
      </div>`;
    el.addEventListener('click', ()=>{
      selectedBot = bot;
      // 一旦选择机器人，进入聊天视图
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
    btn.addEventListener('click', ()=>{
      // 填充输入框（欢迎/居中输入）
      centerInput.value = p;
    });
    suggestionsWrap.appendChild(btn);
  });
}

// ---------- 聊天视图相关 ----------
function enterChatView(){
  welcomeCard.classList.add('hidden');
  chatView.classList.remove('hidden');
  inputDock.classList.remove('hidden');
  // load messages for bot
  const id = selectedBot.id;
  if(!messages[id]) messages[id] = [];
  renderMessages();
  // 将输入焦点放到底部输入框
  setTimeout(()=> dockInput.focus(), 120);
}

function exitChatView(){
  welcomeCard.classList.remove('hidden');
  chatView.classList.add('hidden');
  inputDock.classList.add('hidden');
}

// ---------- 新建数字分身 ----------
function createNewBot() {
  // 弹出输入框让用户输入新数字分身的名称
  const botName = prompt('请输入新数字分身的名称：', '数字分身' + (bots.length + 1));
  
  // 如果用户取消或未输入名称，则不创建
  if (!botName || botName.trim() === '') {
    return;
  }
  
  // 创建新的数字分身对象
  const newBot = {
    id: Date.now(), // 使用时间戳作为唯一ID
    name: botName.trim()
  };
  
  // 添加到机器人列表
  bots.push(newBot);
  
  // 保存到本地存储
  persist();
  
  // 重新渲染机器人列表
  renderBots();
  
  // 选中新创建的机器人并进入聊天视图
  selectedBot = newBot;
  enterChatView();
  
  // 显示成功提示
  alert(`已创建新数字分身：${botName}`);
}

// ---------- 渲染消息 ----------
function renderMessages(){
  const id = selectedBot.id;
  messagesEl.innerHTML = '';
  (messages[id] || []).forEach(m => {
    const div = document.createElement('div');
    div.className = 'msg ' + (m.role === 'user' ? 'user' : 'bot');

    // 如果是机器人消息，添加头像
    if(m.role === 'bot'){
      const avatar = document.createElement('div');
      avatar.className = 'avatar';
      avatar.textContent = '🤖';
      div.appendChild(avatar);
    }

    const b = document.createElement('div');
    b.className = 'bubble';
    
    // 处理不同类型的消息
    if (m.type === 'file') {
      // 文件消息
      b.innerHTML = createFileMessageHTML(m);
    } else {
      // 文本消息
      b.textContent = m.text;
    }
    
    div.appendChild(b);
    messagesEl.appendChild(div);
  });
  // auto scroll
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// 创建文件消息的HTML
function createFileMessageHTML(fileMessage) {
  const { filename, filetype, filesize, data, content } = fileMessage;
  
  // 文件大小格式化
  const formattedSize = formatFileSize(filesize);
  
  // 文件图标
  let fileIcon = '📄';
  if (filetype.startsWith('image/')) {
    fileIcon = '🖼️';
  } else if (filetype.startsWith('audio/')) {
    fileIcon = '🎵';
  } else if (filetype.startsWith('video/')) {
    fileIcon = '🎥';
  } else if (filetype.includes('pdf')) {
    fileIcon = '📄';
  } else if (filetype.includes('word') || filetype.includes('doc')) {
    fileIcon = '📝';
  } else if (filetype.includes('excel') || filetype.includes('spreadsheet')) {
    fileIcon = '📊';
  } else if (filetype.includes('powerpoint') || filetype.includes('presentation')) {
    fileIcon = '📋';
  } else if (filetype.includes('zip') || filetype.includes('archive')) {
    fileIcon = '🗜️';
  }
  
  // 文件大小图标
  let sizeIcon = '📦';
  if (filesize < 1024) {
    sizeIcon = '📄';
  } else if (filesize < 1048576) {
    sizeIcon = '📁';
  } else {
    sizeIcon = '📦';
  }
  
  // 基础文件信息HTML
  let html = `
    <div class="file-message">
      <div class="file-info">
        <div class="file-icon">${fileIcon}</div>
        <div class="file-details">
          <div class="file-name">
            <span>${filename}</span>
          </div>
          <div class="file-meta">
            <span>${sizeIcon} ${formattedSize}</span>
          </div>
        </div>
      </div>
  `;
  
  // 如果是图片，显示预览
  if (filetype.startsWith('image/') && data) {
    html += `
      <div class="file-preview">
        <img src="${data}" alt="${filename}" class="preview-image">
      </div>
    `;
  }
  // 如果是文本文件，显示部分内容
  else if (content && (filetype.startsWith('text/') || 
                      filetype === 'application/json' || 
                      filetype === 'application/javascript' || 
                      filetype === 'application/xml')) {
    // 限制显示的内容长度
    const previewContent = content.length > 500 ? content.substring(0, 500) + '...' : content;
    html += `
      <div class="file-preview">
        <pre class="preview-text">${previewContent}</pre>
      </div>
    `;
  }
  
  // 添加下载按钮
  html += `
      <div class="file-action">
        <button class="file-download" onclick="downloadFile('${filename}', '${filetype}', '${data || content || ''}')">
          📥 下载
        </button>
      </div>
    </div>
  `;
  
  return html;
}

// 下载文件
function downloadFile(filename, filetype, data) {
  if (!data) {
    alert('无法下载文件，没有文件数据');
    return;
  }
  
  try {
    let blob;
    
    // 如果是base64编码的数据（图片）
    if (data.startsWith('data:')) {
      // 从base64字符串创建Blob
      const byteString = atob(data.split(',')[1]);
      const mimeString = data.split(',')[0].split(':')[1].split(';')[0];
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      
      blob = new Blob([ab], { type: mimeString });
    } 
    // 如果是文本内容
    else {
      blob = new Blob([data], { type: filetype });
    }
    
    // 创建下载链接
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
  } catch (error) {
    console.error('下载文件失败:', error);
    alert('下载文件失败: ' + error.message);
  }
}

// ---------- 导出功能 ----------
function exportCurrentAsJSON(){
  if(!selectedBot){
    // 导出所有
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
    // 所有机器人合并导出
    let txt = '';
    for(const b of bots){
      txt += `=== ${b.name} ===\n`;
      const msgs = messages[b.id] || [];
      msgs.forEach(m=> txt += `[${m.role}] ${m.text}\n`);
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

  recognition.onstart = ()=>{
    isRecording = true; updateMicUI(true);
  };
  recognition.onend = ()=>{
    isRecording = false; updateMicUI(false);
  };
  recognition.onerror = (e)=>{
    console.error('Speech recognition error', e);
    isRecording = false; updateMicUI(false);
    alert('语音识别出错：' + (e.error || e.message));
  };
  recognition.onresult = (evt)=>{
    const text = (evt.results[0] && evt.results[0][0] && evt.results[0][0].transcript) || '';
    // 将识别结果注入到当前活动输入框（如果在聊天视图则 dockInput，否则 centerInput）
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
  try{
    recognition.start();
  }catch(e){
    console.warn('recognition start err', e);
  }
}
function stopRecognition(){
  if(recognition){
    try{ recognition.stop(); }catch(e){}
  }
}

function toggleMic(target){
  // target: 'center' or 'dock'
  if(isRecording){
    stopRecognition();
    return;
  }
  // ensure recognition and start
  if(ensureRecognition()){
    // when start, focus appropriate input
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

  // 检查文件大小（限制为10MB）
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    alert(`文件太大，请选择小于10MB的文件。当前文件大小: ${formatFileSize(file.size)}`);
    fileInput.value = '';
    return;
  }

  // 根据文件类型处理
  const fileType = file.type;
  const fileName = file.name;
  
  // 在输入框中显示文件名
  targetInput.value = `[文件] ${fileName}`;
  
  // 如果是图片文件，预览图片
  if (fileType.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = function(e) {
      // 保存图片数据到临时变量
      fileInput._imageData = e.target.result;
    };
    reader.readAsDataURL(file);
  } 
  // 如果是文本文件，读取内容
  else if (fileType.startsWith('text/') || 
           fileType === 'application/json' || 
           fileType === 'application/javascript' || 
           fileType === 'application/xml' ||
           fileType === 'application/pdf' ||
           fileType === 'application/msword' ||
           fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    
    const reader = new FileReader();
    reader.onload = function(e) {
      // 保存文本内容到临时变量
      fileInput._fileContent = e.target.result;
    };
    reader.readAsText(file);
  }
}

// 格式化文件大小
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
  else return (bytes / 1048576).toFixed(2) + ' MB';
}

// 发送文件消息
async function sendFileMessage(fileInput) {
  const file = fileInput.files && fileInput.files[0];
  if (!file || !selectedBot) return;

  const id = selectedBot.id;
  if (!messages[id]) messages[id] = [];

  // 创建文件消息对象
  const fileMessage = {
    role: 'user',
    type: 'file',
    filename: file.name,
    filetype: file.type,
    filesize: file.size,
    ts: Date.now()
  };

  // 如果是图片，添加图片数据
  if (file.type.startsWith('image/') && fileInput._imageData) {
    fileMessage.data = fileInput._imageData;
  }
  // 如果是文本文件，添加文本内容
  else if ((file.type.startsWith('text/') || 
           file.type === 'application/json' || 
           file.type === 'application/javascript' || 
           file.type === 'application/xml') && fileInput._fileContent) {
    fileMessage.content = fileInput._fileContent;
  }

  // 添加到消息列表
  messages[id].push(fileMessage);
  renderMessages();
  persist();

  // 清空文件输入
  fileInput.value = '';
  delete fileInput._imageData;
  delete fileInput._fileContent;

  // 显示加载状态
  const loadingMessage = { role: 'bot', text: '正在处理文件...', ts: Date.now() };
  messages[id].push(loadingMessage);
  renderMessages();

  try {
    // 准备API请求数据（匹配app.py的期望格式）
    const fileDescription = fileMessage.content
      ? `用户上传了文件: ${file.name}\n文件内容:\n${fileMessage.content.substring(0, 1000)}`
      : `用户上传了文件: ${file.name} (${file.type}, ${formatFileSize(file.size)})`;

    const requestData = {
      message: fileDescription,
      session_id: selectedBot.session_id || `session_${selectedBot.id}_${Date.now()}`
    };

    // 发送POST请求到后端API
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    // 检查响应状态
    if (!response.ok) {
      throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
    }

    // 解析JSON响应
    const data = await response.json();

    // 检查响应数据格式
    if (data.status === 'error') {
      throw new Error(data.error || 'API返回错误');
    }

    if (!data || typeof data.reply !== 'string') {
      throw new Error('API返回格式错误');
    }

    // 移除加载状态消息
    messages[id].pop();
    messages[id].push({ role: 'bot', text: data.reply, ts: Date.now() });

  } catch (error) {
    console.error('发送文件消息到后端失败:', error);
    // 移除加载状态消息
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

  // 检查是否是文件上传
  if (text.startsWith('[文件] ') && centerFile.files.length > 0) {
    if (!selectedBot) {
      selectedBot = bots[0];
      renderBots();
    }
    enterChatView();
    await sendFileMessage(centerFile);
    centerInput.value = '';
    return;
  }

  if (!selectedBot) {
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
  
  // 检查是否是文件上传
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
  
  // 显示加载状态
  const loadingMessage = { role: 'bot', text: '正在思考...', ts: Date.now() };
  messages[id].push(loadingMessage);
  renderMessages();
  
  try {
    const reply = await sendMessageToBackend(selectedBot, text);
    // 移除加载状态消息
    messages[id].pop();
    messages[id].push({ role: 'bot', text: reply, ts: Date.now() });
  } catch (e) {
    // 移除加载状态消息
    messages[id].pop();
    messages[id].push({ role: 'bot', text: `网络连接失败：${e.message}`, ts: Date.now() });
  }
  
  renderMessages();
  persist();
}

// ---------- 实际调用后端API ----------
async function sendMessageToBackend(bot, message) {
  // 确保机器人有会话ID
  if(!bot.session_id) bot.session_id = `session_${bot.id}_${Date.now()}`;

  // 准备API请求数据（匹配app.py的期望格式）
  const requestData = {
    message: message,
    session_id: bot.session_id
  };

  try {
    // 发送POST请求到后端API
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    // 检查响应状态
    if (!response.ok) {
      throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
    }

    // 解析JSON响应
    const data = await response.json();

    // 检查响应数据格式
    if (data.status === 'error') {
      throw new Error(data.error || 'API返回错误');
    }

    if (!data || typeof data.reply !== 'string') {
      throw new Error('API返回格式错误');
    }

    // 返回机器人回复
    return data.reply;

  } catch (error) {
    console.error('发送消息到后端失败:', error);
    throw error; // 重新抛出错误以便上层处理
  }
}

// ---------- 事件绑定 ----------
centerSend.addEventListener('click', handleCenterSend);
centerInput.addEventListener('keyup', (e)=>{ if(e.key === 'Enter') handleCenterSend(); });

dockSend.addEventListener('click', handleDockSend);
dockInput.addEventListener('keyup', (e)=>{ if(e.key === 'Enter') handleDockSend(); });

// 文件上传事件
centerFile.addEventListener('change', (e)=>{ handleFileUpload(e.target, centerInput); });
dockFile.addEventListener('change', (e)=>{ handleFileUpload(e.target, dockInput); });

exportJsonBtn.addEventListener('click', exportCurrentAsJSON);
exportTxtBtn.addEventListener('click', exportCurrentAsTxt);
themeToggle.addEventListener('click', toggleTheme);

// 清除历史消息
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
clearHistoryBtn.addEventListener('click', clearChatHistory);

// 语言切换（中 / EN）
function loadSpeechLang(){
  speechLang = localStorage.getItem('speech_lang') || 'zh-CN';
  langToggle.textContent = speechLang === 'zh-CN' ? '🌐 中' : '🌐 EN';
}
function toggleSpeechLang(){
  speechLang = speechLang === 'zh-CN' ? 'en-US' : 'zh-CN';
  localStorage.setItem('speech_lang', speechLang);
  loadSpeechLang();
  // reinit recognition with new lang next time
  if(recognition){ recognition.lang = speechLang; }
}
langToggle.addEventListener('click', toggleSpeechLang);

// 新建数字分身按钮事件
newChatBtn.addEventListener('click', createNewBot);

// mic buttons
centerMic.addEventListener('click', ()=> toggleMic('center'));
dockMic.addEventListener('click', ()=> toggleMic('dock'));

// ---------- 清除历史消息功能 ----------
async function clearChatHistory() {
  if (!selectedBot) {
    alert('请先选择一个数字分身');
    return;
  }

  // 确认清除
  if (confirm('确定要清除当前对话的所有历史消息吗？此操作不可撤销。')) {
    try {
      // 调用后端API清除会话
      const session_id = selectedBot.session_id || `session_${selectedBot.id}_${Date.now()}`;
      const response = await fetch('/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ session_id: session_id })
      });

      if (!response.ok) {
        throw new Error(`后端清除失败: ${response.status}`);
      }

      // 清除前端消息
      messages[selectedBot.id] = [];

      // 保存到本地存储
      persist();

      // 更新界面
      renderMessages();

      // 显示提示
      showToast('历史消息已清除');

    } catch (error) {
      console.error('清除历史消息失败:', error);
      showToast('清除失败: ' + error.message);
    }
  }
}

// 显示提示消息
function showToast(message) {
  // 创建提示元素
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  
  // 添加到页面
  document.body.appendChild(toast);
  
  // 添加样式
  toast.style.position = 'fixed';
  toast.style.bottom = '20px';
  toast.style.left = '50%';
  toast.style.transform = 'translateX(-50%)';
  toast.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  toast.style.color = 'white';
  toast.style.padding = '10px 20px';
  toast.style.borderRadius = '5px';
  toast.style.zIndex = '1000';
  toast.style.transition = 'opacity 0.3s ease';
  
  // 淡出效果
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 300);
  }, 2000);
}

// ---------- 本地保存 ----------
function persist(){
  localStorage.setItem('bots_v1', JSON.stringify(bots));
  localStorage.setItem('messages_v1', JSON.stringify(messages));
}

// ---------- 初始化 ----------
function init(){
  renderBots(); renderSuggestions(); loadTheme(); loadSpeechLang();

  const storedSelected = null;
  if(storedSelected){ selectedBot = storedSelected; enterChatView(); } else { exitChatView(); }

  contentScroll.addEventListener('click', ()=>{ if(!selectedBot) centerInput.focus(); });

  // ensure recognition object exists lazily when user first clicks mic
  // handle page unload to stop recognition
  window.addEventListener('beforeunload', ()=>{ if(recognition) try{ recognition.abort(); }catch(e){} });
}

init();
