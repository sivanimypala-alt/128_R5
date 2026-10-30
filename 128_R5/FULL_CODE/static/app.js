const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n || 0).toLocaleString('en-IN');
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const channelNames = {online_review:'Online reviews', in_app:'In-app comments', staff_shift_note:'Staff shift notes', call_centre_ticket:'Call-centre tickets'};
const severityPill = s => `<span class="pill ${esc(String(s||'low').toLowerCase())}">${esc(String(s||'low').toUpperCase())}</span>`;
let currentView = 'overview';
async function getJSON(url, options={}) { const r = await fetch(url, options); if (!r.ok) throw new Error(`Request failed (${r.status})`); return r.json(); }
function showView(view) {
  currentView = view;
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `view-${view}`));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  const labels={overview:'Overview',feedback:'Feedback inbox',issues:'Issue intelligence',emerging:'Emerging radar',brief:'Regional brief',ask:'Ask StorePulse'};
  $('crumb-current').textContent=labels[view]||view;
  if(view==='feedback') loadFeedback(); if(view==='issues') loadIssues(); if(view==='emerging') loadEmerging(); if(view==='brief') loadBrief();
}
document.querySelectorAll('.nav-item').forEach(b=>b.addEventListener('click',()=>showView(b.dataset.view)));
document.querySelectorAll('[data-goto]').forEach(b=>b.addEventListener('click',()=>showView(b.dataset.goto)));

async function loadOverview() {
  try {
    const d=await getJSON('/api/overview');
    if(!d.ready){$('pipeline-status').textContent='Data files not found';$('data-status').textContent='Demo not loaded';document.querySelectorAll('#kpi-total,#kpi-actionable,#kpi-risk,#kpi-stores').forEach(e=>e.textContent='—');return;}
    $('pipeline-status').textContent=`${fmt(d.total)} records loaded`; $('data-status').textContent='Dataset connected';
    document.querySelector('.status-light').style.background='#24b58c';
    $('kpi-total').textContent=fmt(d.total); $('kpi-actionable').textContent=fmt(d.actionable); $('kpi-risk').textContent=fmt(d.high_risk); $('kpi-stores').textContent=fmt(d.stores); $('nav-feedback-count').textContent=fmt(d.total); $('radar-count').textContent=fmt(d.emerging_count);
    $('trend-caption').textContent=`${d.week_count} weeks · latest ${d.current_week}`;
    const trend=d.trend||[], max=Math.max(1,...trend.map(x=>x.count));
    $('trend-chart').innerHTML=trend.length?trend.map(x=>`<div class="bar-column"><span class="bar-value">${fmt(x.count)}</span><div class="bar" style="height:${Math.max(3,x.count/max*75)}%" title="${fmt(x.count)} records"></div><span class="bar-label">${esc(x.week)}</span></div>`).join(''):'<div class="empty-state">No weekly data available.</div>';
    const issues=d.top_issues||[], maxIssue=Math.max(1,...issues.map(x=>x.count));
    $('top-issues').innerHTML=issues.length?issues.map(x=>`<div class="top-issue-row"><div class="top-issue-name"><span>${esc(x.name)}</span><span>${fmt(x.count)}</span></div><div class="progress-track" style="grid-column:1/3"><div class="progress-fill" style="width:${x.count/maxIssue*100}%"></div></div></div>`).join(''):'<div class="empty-state">No issue categories available.</div>';
    const channels=Object.entries(d.channels||{}).sort((a,b)=>b[1]-a[1]), maxC=Math.max(1,...channels.map(x=>x[1]));
    $('channel-list').innerHTML=channels.map(([name,count])=>`<div class="channel-row"><i class="channel-dot"></i><span class="channel-name">${esc(channelNames[name]||name)}</span><strong class="channel-value">${fmt(count)}</strong><div class="channel-track"><span style="width:${count/maxC*100}%"></span></div></div>`).join('');
  } catch(e){ $('pipeline-status').textContent='API unavailable'; $('data-status').textContent='Connection error'; console.error(e); }
}
async function loadFeedback(){
  const q=$('feedback-search').value.trim(),channel=$('channel-filter').value,severity=$('severity-filter').value,region=$('region-filter').value;
  $('feedback-body').innerHTML='<tr><td colspan="5" class="empty-state">Loading records…</td></tr>';
  try { const d=await getJSON(`/api/feedback?q=${encodeURIComponent(q)}&channel=${encodeURIComponent(channel)}&severity=${severity}&region=${region}&limit=100`); $('feedback-total').textContent=`Showing ${fmt(d.items.length)} records matching the filters`;
    $('feedback-body').innerHTML=d.items.length?d.items.map(x=>`<tr><td><span class="feedback-main">${esc(x.id)} <span class="pill" style="background:#f5f6fa;color:#747d8e">${esc(channelNames[x.channel]||x.channel)}</span></span><span class="feedback-text">${esc(x.text)}</span><span class="feedback-date">${esc(x.date)}</span></td><td><span class="feedback-main">${esc(x.store)}</span><span class="feedback-date">${esc(x.region)}</span></td><td><span class="pill owner">${esc(x.issue)}</span></td><td>${severityPill(x.severity)}</td><td>${x.actionable?'<span class="action-yes">✓ Yes</span>':'<span class="action-no">— No</span>'}</td></tr>`).join(''):'<tr><td colspan="5" class="empty-state">No matching feedback.</td></tr>';
  }catch(e){$('feedback-body').innerHTML='<tr><td colspan="5" class="empty-state">Could not load feedback. Check the server console.</td></tr>';}
}
$('apply-filters').addEventListener('click',loadFeedback); $('feedback-search').addEventListener('keydown',e=>{if(e.key==='Enter')loadFeedback()});
async function loadIssues(){
 try{const d=await getJSON('/api/issues');const items=d.items||[];$('issue-grid').innerHTML=items.length?items.map((x,i)=>`<article class="issue-card"><div class="issue-card-top"><div><span class="issue-id">${esc(x.id)} · PRIORITY ${String(i+1).padStart(2,'0')}</span><h3>${esc(x.name)}</h3></div>${severityPill(x.severity)}</div><div class="issue-meta"><span class="meta-chip">${fmt(x.count)} records</span><span class="meta-chip">${fmt(x.affected_stores)} affected stores</span><span class="meta-chip">${fmt(x.high_risk)} high-risk</span><span class="meta-chip">${esc(x.trend==='up'?'↗ Rising':x.trend==='down'?'↘ Falling':'→ Flat')}</span></div><div class="issue-evidence">“${esc(x.evidence)}”</div><div class="issue-card-footer"><span>Owner · <strong>${esc(x.owner)}</strong></span><span>${fmt(x.actionable)} actionable</span></div></article>`).join(''):'<div class="empty-state">No issue signals available.</div>';}catch(e){$('issue-grid').innerHTML='<div class="empty-state">Could not load issue data.</div>';}
}
async function loadEmerging(){
 try{const d=await getJSON('/api/emerging');const items=d.items||[];$('emerging-total').textContent=`${items.length} flags`;$('emerging-body').innerHTML=items.length?items.map(x=>`<tr><td><span class="feedback-main">${esc(x.as_of_week_start)}</span></td><td><span class="feedback-main">${esc(x.store_name)}</span><span class="feedback-date">${esc(x.region)} · ${esc(x.store_id)}</span></td><td><span class="feedback-main">${esc(x.issue_name)}</span><span class="pill owner">${esc(x.issue_type_id)}</span></td><td><span class="confidence">${Math.round(Number(x.confidence||0)*100)}%</span><div class="feedback-date">Candidate score, not calibrated probability</div></td><td><span class="mini-evidence">${esc(String(x.evidence_feedback_ids||'').split(';').join(' · '))}</span><span class="feedback-date">${x.evidence_count||0} evidence records</span></td></tr>`).join(''):'<tr><td colspan="5" class="empty-state">No flags passed the current detector thresholds. This is preferable to inventing unsupported alerts.</td></tr>';}catch(e){$('emerging-body').innerHTML='<tr><td colspan="5" class="empty-state">Could not load emerging flags.</td></tr>';}
}
async function loadBrief(){
 const region=$('brief-region').value;
 try{const d=await getJSON(`/api/brief?region=${encodeURIComponent(region)}`);$('brief-header').innerHTML=`<div class="brief-eyebrow">REGIONAL ACTION BRIEF · ${esc(d.week||'DATA PERIOD')}</div><h2>${esc(d.region)} region</h2><p>${esc(d.summary||'Regional summary')}</p><div class="brief-stats"><div class="brief-stat"><strong>${(d.items||[]).length}</strong><span>TOP ISSUE CATEGORIES</span></div><div class="brief-stat"><strong>${fmt((d.items||[]).reduce((s,x)=>s+x.count,0))}</strong><span>RECORDS IN TOP 5</span></div><div class="brief-stat"><strong>Evidence-led</strong><span>RECOMMENDATION STANDARD</span></div></div>`;
 $('brief-list').innerHTML=(d.items||[]).length?d.items.map((x,i)=>`<article class="brief-item"><div class="brief-rank">${i+1}</div><div><h3>${esc(x.name)} <span class="pill owner">${esc(x.id)}</span></h3><p>${fmt(x.count)} records this week · ${fmt(x.affected_stores)} affected stores · ${esc(x.trend)} vs previous week · owner ${esc(x.owner)}</p><div class="item-bottom">${severityPill(x.severity)}</div><p class="evidence">Evidence: “${esc(x.evidence)}”</p></div></article>`).join(''):'<div class="panel empty-state">No regional issue records for this selection.</div>';
 }catch(e){$('brief-header').innerHTML='<div class="empty-state">Could not generate regional brief.</div>';}
}
$('brief-region').addEventListener('change',loadBrief);
document.querySelectorAll('[data-question]').forEach(b=>b.addEventListener('click',()=>{$('ask-question').value=b.dataset.question;$('ask-question').focus()}));
$('ask-form').addEventListener('submit',async e=>{e.preventDefault();const question=$('ask-question').value.trim();if(!question)return;const box=$('ask-answer');box.classList.remove('hidden');box.textContent='Checking the available evidence…';const btn=$('ask-form').querySelector('button[type="submit"]');btn.disabled=true;try{const d=await getJSON('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});box.textContent=(d.answer||'No answer returned')+(d.mode?`\n\nSource mode: ${d.mode}`:'');}catch(e){box.textContent='Could not answer this question. Check the backend server and try again.';}finally{btn.disabled=false;}});
$('refresh-btn').addEventListener('click',()=>{loadOverview();if(currentView!=='overview')showView(currentView)});
loadOverview();
