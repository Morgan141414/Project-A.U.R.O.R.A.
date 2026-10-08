const $ = (selector) => document.querySelector(selector);
const messages = $('#messages');

function addMessage(text, kind) {
  const row = document.createElement('div'); row.className = `message ${kind}`;
  const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
  row.appendChild(bubble); messages.appendChild(row); messages.scrollTop = messages.scrollHeight;
}
function core(working) { $('#core').classList.toggle('thinking', working); $('#core-mode').textContent = working ? 'THINKING' : 'STANDBY'; $('#core-caption-status').textContent = working ? 'Processing local neural pathways' : 'Awaiting your instruction'; }
async function send() {
  const box = $('#message'), text = box.value.trim(); if (!text) return;
  addMessage(text, 'user'); box.value = ''; core(true);
  try { const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message:text})}); const data = await response.json(); addMessage(response.ok ? data.answer : `CORE // ${data.detail || 'Request failed'}`, 'ai'); }
  catch (error) { addMessage('Connection to the local core failed. Start the server with: python app.py', 'ai'); }
  finally { core(false); refresh(); }
}
async function learn() {
  const topic = $('#topic').value.trim(), materials = $('#materials').value.trim();
  if (materials.length < 100) { $('#train-state').textContent = 'ERROR // ADD MORE SOURCE MATERIAL'; return; }
  $('#learn').disabled = true; core(true); $('#train-state').textContent = 'QUEUED // BUILDING DATASET';
  try { const response = await fetch('/api/train', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({topic,materials,epochs:1})}); const data = await response.json(); if (!response.ok) { $('#train-state').textContent = `ERROR // ${data.detail || 'TRAINING REJECTED'}`; $('#learn').disabled = false; core(false); return; } poll(); }
  catch (error) { $('#train-state').textContent = 'ERROR // SERVER OFFLINE'; $('#learn').disabled = false; core(false); }
}
async function poll() {
  try { const data = await (await fetch('/api/status')).json(), state = data.training; $('#train-state').textContent = state.active ? `${state.progress} // ${state.topic}` : `${state.progress || 'READY'} // CHECKPOINT ${data.checkpoint ? 'READY' : 'MISSING'}`; if (state.active) setTimeout(poll, 1500); else { $('#learn').disabled = false; core(false); refresh(); } }
  catch (error) { $('#train-state').textContent = 'ERROR // STATUS UNAVAILABLE'; $('#learn').disabled = false; core(false); }
}
async function refresh() {
  try { const status = await (await fetch('/api/status')).json(); $('#model-status').textContent = status.checkpoint ? 'STATUS: OK' : 'NO CHECKPOINT'; const metrics = document.querySelectorAll('.metric b'); if (metrics.length >= 2) { metrics[0].textContent = `CPU ${Math.round(status.vitals.cpu_percent)}%`; metrics[1].textContent = `RAM ${Math.round(status.vitals.memory_percent)}%`; } const data = await (await fetch('/api/logs')).json(), target = $('#logs'); target.replaceChildren(); data.logs.slice(0,7).forEach((entry) => { const line = document.createElement('div'); line.textContent = `>> ${entry.message} [${entry.time}]`; target.appendChild(line); }); }
  catch (error) { $('#model-status').textContent = 'OFFLINE'; }
}
$('#send').onclick = send; $('#learn').onclick = learn; $('#new-session').onclick = () => { messages.replaceChildren(); addMessage('New reasoning session initialized. What shall we learn today?', 'ai'); }; $('#message').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } }); setInterval(() => { $('#clock').textContent = new Date().toLocaleTimeString('en-GB'); }, 1000); refresh();
document.querySelectorAll('.history-item').forEach((item) => { item.onclick = () => { document.querySelectorAll('.history-item').forEach((entry) => entry.classList.remove('active')); item.classList.add('active'); }; });
setInterval(refresh, 5000);
