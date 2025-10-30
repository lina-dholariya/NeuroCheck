(() => {
  function el(id){ return document.getElementById(id); }
  const STORAGE_KEY = 'nc_chat_history';

  async function sendMessage(text){
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    return res.json();
  }

  function toBulletPoints(text){
    // Convert simple newline / numbered lists into bullet points for better readability
    if(!text) return text;
    const lines = text.split(/\n+/).map(l => l.trim()).filter(Boolean);
    if(lines.length <= 1) return text;
    const ul = document.createElement('ul');
    ul.style.margin = '0';
    ul.style.paddingLeft = '18px';
    for(const line of lines){
      const li = document.createElement('li');
      li.textContent = line.replace(/^[-*\d+.\)\s]+/, '');
      ul.appendChild(li);
    }
    return ul;
  }

  function persistMessage(role, text){
    try{
      const raw = localStorage.getItem(STORAGE_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      arr.push({ role, text, t: Date.now() });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(arr));
    }catch(e){ /* ignore quota/JSON errors */ }
  }

  function appendMessage(container, role, text){
    const item = document.createElement('div');
    item.className = `chat-msg chat-${role}`;
    // For bot messages, try bullet formatting when multiple lines present
    if(role === 'bot'){
      const bullets = toBulletPoints(text);
      if(typeof bullets !== 'string'){
        item.appendChild(bullets);
      } else {
        item.textContent = text;
      }
    } else {
      item.textContent = text;
    }
    container.appendChild(item);
    container.scrollTop = container.scrollHeight;
    // Save after rendering
    persistMessage(role, text);
  }

  function init(){
    const box = el('chat-box');
    const input = el('chat-input');
    const btn = el('chat-send');
    const widget = el('chat-widget');
    const openBtn = el('chat-open');
    const closeBtn = el('chat-close');
    if(!box || !input || !btn) return;

    // Hydrate from localStorage
    try{
      const raw = localStorage.getItem(STORAGE_KEY);
      if(raw){
        const arr = JSON.parse(raw);
        for(const m of arr){
          if(m && typeof m.text === 'string' && (m.role === 'user' || m.role === 'bot')){
            appendMessage(box, m.role, m.text);
          }
        }
      }
    }catch(e){ /* ignore */ }

    // Toggle fullscreen chat
    function goToChat(){
      window.location.href = '/chat';
    }
    if(openBtn) openBtn.addEventListener('click', (e) => { e.stopPropagation(); goToChat(); });
    if(closeBtn) closeBtn.addEventListener('click', (e) => { e.stopPropagation(); widget.classList.remove('chat-fullscreen'); });
    // Navigate to /chat when clicking anywhere on the widget (excluding inputs/buttons)
    widget.addEventListener('click', (e) => {
      const t = e.target;
      if(['INPUT','BUTTON','UL','LI'].includes(t.tagName)) return;
      goToChat();
    });

    btn.addEventListener('click', async () => {
      const text = input.value.trim();
      if(!text) return;
      appendMessage(box, 'user', text);
      input.value = '';
      try{
        const json = await sendMessage(text);
        if(json && json.success){
          const data = json.data || {};
          appendMessage(box, 'bot', data.response || '');
        } else {
          appendMessage(box, 'bot', 'Sorry, there was a problem.');
        }
      } catch(e){
        appendMessage(box, 'bot', 'Network error.');
      }
    });

    input.addEventListener('keydown', (e) => {
      if(e.key === 'Enter'){
        btn.click();
      }
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();


