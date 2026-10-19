'use strict';
(() => {
  const base = document.querySelector('meta[name="v2w-base"]')?.content || '.';
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const path = (p) => /^https?:\/\//.test(p) ? p : `${base}/${p.replace(/^\//, '')}`;
  const state = { board: null, archive: null, snapshot: 'current', search: '', provider: 'All', detail: false, metricView: 'key', sort: 'score', sortDir: 'desc', chart: 'score', news: null, analysis: null, cases: [], caseId: 'tape', caseA: 'opus5', caseB: 'gpt6astra' };
  const paperExtra = [
    {key:'t_ape',label:'Translation APE',short:'T-APE',unit:'cm',direction:'down',digits:2,group:'Dynamics',description:'Translation APE from the published-paper results.'},
    {key:'r_ape',label:'Rotation APE',short:'R-APE',unit:'deg',direction:'down',digits:2,group:'Dynamics',description:'Rotation APE from the published-paper results.'},
    {key:'t_rpe',label:'Translation RPE',short:'T-RPE',unit:'cm',direction:'down',digits:2,group:'Dynamics',description:'Translation RPE from the published-paper results.'}
  ];
  const compactKeys = ['score','success','progress','scene_cd','build'];
  const labels = {score:'Overall score',success:'Task success',build:'Build rate'};
  const getRows = () => state.snapshot === 'current' ? state.board.models : state.archive.models;
  const getMetrics = () => state.snapshot === 'current' ? state.board.metrics : state.archive.fields.map(k => state.board.metrics.find(m => m.key === k) || paperExtra.find(m => m.key === k));
  const metric = (key) => [...state.board.metrics,...paperExtra].find(m => m.key === key);
  function format(v, key, withUnit = false) {
    if (v == null || !Number.isFinite(Number(v))) return '—';
    const m = metric(key);
    const n = Number(v).toFixed(m?.digits ?? 2);
    return withUnit && m?.unit === '%' ? `${n}%` : n;
  }
  async function fetchJSON(p) {
    const embedded=$('#v2w-inline-data');
    if(embedded){const data=JSON.parse(embedded.textContent);if(Object.prototype.hasOwnProperty.call(data,p))return data[p];}
    const response = await fetch(path(p));
    if (!response.ok) throw new Error(`Could not load ${p}`);
    return response.json();
  }
  function toast(message) {
    const t = $('#toast'); if (!t) return;
    t.textContent = message; t.classList.add('visible');
    clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove('visible'), 2600);
  }
  async function copy(text, message = 'Copied to clipboard') {
    try { await navigator.clipboard.writeText(text); toast(message); }
    catch {
      const area = document.createElement('textarea'); area.value = text; area.style.position = 'fixed'; area.style.opacity = '0'; document.body.append(area); area.select();
      const ok = document.execCommand('copy'); area.remove(); toast(ok ? message : 'Copy unavailable in this browser');
    }
  }
  function download(content, name, type = 'text/plain') {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const a = document.createElement('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function sortedRows() {
    const direction = state.sortDir === 'desc' ? -1 : 1;
    return [...getRows()].sort((a,b) => (a.reference ? -1 : b.reference ? 1 : direction * (a.metrics[state.sort] - b.metrics[state.sort])));
  }
  function competitionRanks(rows, key) {
    let rank = 0, previous;
    return new Map(rows.map((row, index) => {
      const value = row.metrics[key];
      if (index === 0 || value !== previous) rank = index + 1;
      previous = value;
      return [row.id, rank];
    }));
  }
  function filteredRows() {
    return sortedRows().filter(row => {
      const providerMatch = state.provider === 'All' || (state.provider === 'Others' ? !['Anthropic','OpenAI','Google','Meta','Reference'].includes(row.provider) : row.provider === state.provider);
      return providerMatch && `${row.name} ${row.provider} ${row.model_id || ''}`.toLowerCase().includes(state.search.toLowerCase());
    });
  }
  function renderProviders() {
    const providers = ['All','Anthropic','OpenAI','Google','Meta','Others'];
    $('#provider-filters').innerHTML = providers.map(p => `<button type="button" data-provider="${esc(p)}" class="${p === state.provider ? 'active' : ''}" aria-pressed="${p === state.provider}">${esc(p)}</button>`).join('');
    $$('#provider-filters button').forEach(b => b.addEventListener('click', () => {state.provider = b.dataset.provider; renderProviders(); renderTable();}));
  }
  // The leaderboard keeps the height of its full, unfiltered list: filtering or searching leaves
  // empty room at the foot of the section instead of pulling the next section up.
  let boardFullHeight=0;
  function holdBoardHeight(){
    const sec=$('#leaderboard');if(!sec||!sec.getBoundingClientRect)return;
    const full=state.provider==='All'&&!state.search;
    if(full){sec.style.minHeight='';boardFullHeight=sec.getBoundingClientRect().height;}
    if(boardFullHeight)sec.style.minHeight=`${Math.ceil(boardFullHeight)}px`;
  }
  globalThis.addEventListener?.('resize',()=>{if(state.provider==='All'&&!state.search){holdBoardHeight();}else{boardFullHeight=0;const sec=$('#leaderboard');if(sec)sec.style.minHeight='';}});
  function renderTable() {
    const available = getMetrics();
    // Metric views: key metrics, one evaluation dimension (V2WScore stays first as the anchor), or all.
    const view = state.metricView;
    const columns = view === 'all' ? available
      : view === 'key' ? compactKeys.map(k => available.find(m => m.key === k)).filter(Boolean)
      : available.filter(m => m.key === 'score' || m.group === view);
    $('#leaderboard-table').classList.toggle('all-metrics',view === 'all');
    $('#table-head').innerHTML = `<tr><th scope="col">#</th><th scope="col">Model / configuration</th>${columns.map(m => `<th scope="col" aria-sort="${state.sort===m.key ? (state.sortDir==='desc'?'descending':'ascending') : 'none'}"><button type="button" data-sort="${esc(m.key)}" class="${state.sort===m.key?'sorted':''}" title="${esc(m.description)}"><span>${esc(m.short)} ${m.direction==='up'?'↑':'↓'}</span><small>${esc(m.unit)}</small></button></th>`).join('')}</tr>`;
    const globalRanks = competitionRanks(sortedRows().filter(r => !r.reference), state.sort);
    const best = Object.fromEntries(columns.map(m => [m.key,(m.direction==='up'?Math.max:Math.min)(...getRows().filter(r=>!r.reference).map(r=>r.metrics[m.key]))]));
    const rows = filteredRows();
    $('#table-body').innerHTML = rows.length ? rows.map(r => `<tr class="${r.featured?'featured ':''}${r.reference?'reference':''}"><td class="${r.reference?'ref-rank':''}"><span class="table-rank rank-${r.reference?'ref':globalRanks.get(r.id)}">${r.reference?'REF':String(globalRanks.get(r.id)).padStart(2,'0')}</span></td><td><button type="button" class="model-name" data-model="${esc(r.id)}"><span>${esc(r.name)}${r.featured?'<span class="inline-tag">NEW LEADER</span>':''}</span><small>${esc(r.provider)}${r.run_label?` · ${esc(r.run_label)}`:''}</small></button></td>${columns.map(m => {
      const value = r.metrics[m.key];
      const title = r.coverage?.[m.key] != null ? `n=${r.coverage[m.key]} / ${state.board.instances}; open model details for source information` : m.description;
      if(m.key==='score') return `<td title="${esc(title)}"><div class="score-cell"><span class="score-track" aria-hidden="true"><i style="width:${Math.max(0,Math.min(100,value))}%"></i></span><span>${format(value,m.key)}</span></div></td>`;
      return `<td title="${esc(title)}"><span class="number ${!r.reference && value===best[m.key]?'best':''}">${format(value,m.key,m.unit==='%')}</span></td>`;
    }).join('')}</tr>`).join('') : `<tr><td class="empty-table" colspan="${columns.length+2}">No models match this filter. Try another search or provider.</td></tr>`;
    $$('#table-head button[data-sort]').forEach(b=>b.addEventListener('click',()=>{
      if(state.sort===b.dataset.sort)state.sortDir=state.sortDir==='desc'?'asc':'desc';
      else{state.sort=b.dataset.sort;state.sortDir=metric(state.sort).direction==='up'?'desc':'asc';}
      renderTable();
    }));
    $$('#table-body button[data-model]').forEach(b=>b.addEventListener('click',()=>openModel(b.dataset.model)));
    $('#board-summary').textContent = `${rows.filter(r=>!r.reference).length} configurations${state.snapshot==='paper'?' + reference':''}`;
    renderChart();holdBoardHeight();
  }
  // Model logos for the podium (assets/logos, from @lobehub/icons-static-svg, MIT); the letter
  // mark stays as the fallback for providers without one.
  const providerLogos = {Anthropic:'claude-color',OpenAI:'openai',Google:'gemini-color',Meta:'meta-mono','Moonshot AI':'kimi-color',Moonshot:'kimi-color',DeepSeek:'deepseek-color',Alibaba:'qwen-color','Z.ai':'zhipu-color',Zhipu:'zhipu-color'};
  function modelLogo(row) {
    const file = providerLogos[row.provider];
    return file ? `<img src="${esc(path(`assets/logos/${file}.svg`))}" alt="" width="56" height="56" decoding="async">` : esc(modelMark(row));
  }
  function modelMark(row) {
    const marks = {Anthropic:'A',OpenAI:'O',Google:'G',Moonshot:'K',DeepSeek:'D',Alibaba:'Q',Zhipu:'Z',Reference:'R'};
    return marks[row.provider] || row.name.replace(/^(Claude |GPT-)/,'').slice(0,1);
  }
  function renderChart() {
    // the ranking board always shows every model; provider and search filters apply to the table only
    const rows = getRows().sort((a,b)=>(a.reference?-1:b.reference?1:b.metrics[state.chart]-a.metrics[state.chart]));
    const contenders = rows.filter(r=>!r.reference);
    const globalRanks = competitionRanks([...getRows()].filter(r=>!r.reference).sort((a,b)=>b.metrics[state.chart]-a.metrics[state.chart]), state.chart);
    const leader = contenders[0];
    const isFiltered = false;
    const colors = {Anthropic:'#dfb899',OpenAI:'#8acbb3',Google:'#95b8f5',Moonshot:'#c1a7ec',DeepSeek:'#93b6ec',Alibaba:'#edb478',Zhipu:'#b5abea'};
    $('#chart-title').textContent = labels[state.chart];
    if($('#ranking-context'))$('#ranking-context').textContent = `${isFiltered?'Filtered models':'All models'} · higher is better`;
    if($('#rank-chart'))$('#rank-chart').setAttribute('aria-label',`${labels[state.chart]} by model on a zero to one hundred scale. ${state.snapshot==='current'?'Supplied snapshot.':'Published paper results.'}`);
    // Ranking board in the manner of 2048: every model is a tile on a four-column board; the
    // higher the value, the deeper the tile. #1 takes a 2x2 tile, #2 and #3 a 2x1 tile, the rest
    // one cell each; empty cells show as wells. Switching "Rank by" slides tiles to their new
    // places (FLIP), and a tile that changes rank pops briefly.
    const unit=state.chart==='score'?'points':'pp';
    const board=$('#rank-podium');
    const before=new Map(),prevRank=new Map();
    (board.querySelectorAll?board.querySelectorAll('[data-rank-model]'):[]).forEach(t=>{if(t.getBoundingClientRect){before.set(t.dataset.rankModel,t.getBoundingClientRect());prevRank.set(t.dataset.rankModel,t.dataset.rank);}});
    const vals=contenders.map(r=>r.metrics[state.chart]),hi=Math.max(...vals),lo=Math.min(...vals);
    const slot=i=>i===0?[1,1,2,2]:i===1?[3,1,2,1]:i===2?[3,2,2,1]:[1+(i-3)%4,3+Math.floor((i-3)/4),1,1];
    const rowsUsed=contenders.length?Math.max(...contenders.map((_,i)=>{const q=slot(i);return q[1]+q[3]-1;})):0;
    const tile=(r,i)=>{
      const v=r.metrics[state.chart],t=hi>lo?(v-lo)/(hi-lo):1,[c,ro,w,h]=slot(i),rank=globalRanks.get(r.id);
      const note=i===0?(contenders[1]?`${format(v-contenders[1].metrics[state.chart],state.chart)} ${unit} ahead`:'Highest in view'):`${format(leader.metrics[state.chart]-v,state.chart)} ${unit} behind`;
      return `<button type="button" class="podium-card rank-tile ${i===0?'podium-leader tile-lead':i<3?'tile-wide':'tile-small'} ${t>0.62?'is-deep':''}" data-rank-model="${esc(r.id)}" data-rank="${rank}" style="--c:${c};--r:${ro};--w:${w};--h:${h};--t:${t.toFixed(3)}" aria-label="Rank ${rank}, ${esc(r.name)}, ${format(v,state.chart,true)}. Open details.">
        <span class="tile-top"><span class="tile-rank">${String(rank).padStart(2,'0')}</span>${r.featured?'<span class="podium-new">New</span>':''}</span>
        <span class="tile-id"><span class="provider-mark">${modelLogo(r)}</span><span><span class="podium-provider">${esc(r.provider)}</span><strong class="podium-name">${esc(r.name)}</strong></span></span>
        <span class="podium-score">${format(v,state.chart)}<small>${state.chart==='score'?'/ 100':'%'}</small></span>
        ${i<3?`<span class="podium-note">${note}</span>`:''}
      </button>`;
    };
    const wells=Array.from({length:rowsUsed*4},(_,k)=>`<span class="tile-well" aria-hidden="true" style="--c:${1+k%4};--r:${1+Math.floor(k/4)}"></span>`).join('');
    board.innerHTML = contenders.length?wells+contenders.map(tile).join(''):'<p class="rank-empty">No models match the current filters.</p>';
    board.style&&board.style.setProperty&&board.style.setProperty('--rows',rowsUsed);
    const reduce=globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if(!reduce&&board.querySelectorAll)board.querySelectorAll('[data-rank-model]').forEach(el=>{
      if(!el.animate||!el.getBoundingClientRect)return;
      const id=el.dataset.rankModel,old=before.get(id),now=el.getBoundingClientRect();
      if(!old){el.animate([{transform:'scale(.6)',opacity:0},{transform:'scale(1)',opacity:1}],{duration:260,easing:'cubic-bezier(.2,.8,.2,1)'});return;}
      const dx=old.left-now.left,dy=old.top-now.top;  // slide only; size changes settle with the pop, so text never stretches
      if(dx||dy)el.animate([{transform:`translate(${dx}px,${dy}px)`},{transform:'none'}],{duration:420,easing:'cubic-bezier(.3,.7,.2,1)'});
      if(prevRank.get(id)&&prevRank.get(id)!==el.dataset.rank)el.animate([{scale:1},{scale:1.06},{scale:1}],{duration:220,delay:400,easing:'ease-out'});
    });
    if($('#rank-chart'))$('#rank-chart').innerHTML = rows.length ? rows.map((r,i)=>`<button type="button" class="chart-row ${r.id===leader?.id?'ladder-leader':''} ${r.reference?'reference':''}" data-rank-model="${esc(r.id)}" style="--row-color:${colors[r.provider]||'#b9acd6'};--row-order:${i}"><span class="chart-rank">${r.reference?'REF':String(globalRanks.get(r.id)).padStart(2,'0')}</span><span class="chart-name">${esc(r.name)}${r.featured?'<span class="ladder-new">NEW</span>':''}</span><span class="chart-lane"><span class="chart-bar" style="width:${Math.max(0,Math.min(100,r.metrics[state.chart]))}%" aria-hidden="true"><i></i></span></span><span class="chart-value">${format(r.metrics[state.chart],state.chart,state.chart!=='score')}</span><span class="chart-open" aria-hidden="true">↗</span></button>`).join('') : '<p class="ranking-empty">No models match this filter. Try another search or provider.</p>';
    $$('[data-rank-model]').forEach(b=>b.addEventListener('click',()=>openModel(b.dataset.rankModel)));
  }
  function openModel(id) {
    const row=getRows().find(r=>r.id===id);if(!row)return;
    const config=row.configuration?Object.entries(row.configuration).map(([k,v])=>`${k}: ${v??'not supplied'}`).join(' · '):'Full agent settings and a complete-manifest scoring receipt were not supplied.';
    $('#model-dialog-content').innerHTML=`<div class="dialog-head"><span class="eyebrow">${esc(row.provider)} · ${state.snapshot==='current'?'RESULT SNAPSHOT':'PUBLISHED PAPER'}</span><h2>${esc(row.name)}</h2><p>${row.featured?'Leading overall score in this result set.':row.reference?'Built with source information and human assistance; a separate reference row.':'Component metrics help explain what the aggregate score captures.'}</p></div><table class="dialog-metrics"><thead><tr><th>Metric</th><th>Value</th><th>Coverage</th></tr></thead><tbody>${getMetrics().map(m=>`<tr><td>${esc(m.label)}</td><td>${format(row.metrics[m.key],m.key,m.unit==='%')} ${!['%','0–1','0–100'].includes(m.unit)?esc(m.unit):''}</td><td>${row.coverage?.[m.key]!=null?`n=${row.coverage[m.key]} / ${state.board.instances}`:'Not supplied'}</td></tr>`).join('')}</tbody></table><p class="dialog-source">${state.snapshot==='current'?`Recorded: ${esc(row.source?.date||state.board.snapshot_date)}. Source: ${esc(row.source?.name||'supplied evaluation table')}. ${row.run_label?`Run label: ${esc(row.run_label)}. `:''}${esc(config)}`:`Source: <a href="${esc(state.archive.source_url)}" target="_blank" rel="noopener">published Video2World project page</a>. This archive is separate from the newer supplied snapshot.`}</p><div class="dialog-actions">${row.featured?`<a class="button primary" href="${path(`updates/${row.id}.html`)}">Read model update ↗</a>`:''}<button class="button secondary" id="download-model-json" type="button">Download row ↓</button></div>`;
    $('#download-model-json').addEventListener('click',()=>download(JSON.stringify({snapshot:state.snapshot==='current'?state.board.snapshot_id:state.archive.snapshot_id,...row},null,2),`${row.id}-${state.snapshot}.json`,'application/json'));
    const dialog=$('#model-dialog');if(typeof dialog.showModal==='function')dialog.showModal();else dialog.setAttribute('open','');
    history.replaceState(null,'',`${location.pathname}${location.search}#model=${encodeURIComponent(id)}`);
  }
  function setSnapshot(snapshot) {
    state.snapshot=snapshot;state.sort='score';state.sortDir='desc';
    ['current','paper'].forEach(s=>{const b=$(`#snapshot-${s}`);b.classList.toggle('active',s===snapshot);b.setAttribute('aria-pressed',String(s===snapshot));});
    $('#snapshot-date').textContent=snapshot==='current'?lastUpdated():'Archive · published project page';
    if($('#snapshot-note'))$('#snapshot-note').textContent=snapshot==='current'?'Recorded results and per-metric coverage. Select a model for counts and source details.':'Published-paper values are shown separately. The newer snapshot uses different reported values and diagnostic labels.';
    renderTable();
  }
  function exportCSV() {
    const columns=getMetrics(),rows=getRows();
    const quote=v=>`"${String(v??'').replace(/"/g,'""')}"`;
    const header=['snapshot_id','model','provider',...columns.flatMap(m=>[m.key,`n_${m.key}`])];
    const lines=[header,...rows.map(r=>[state.snapshot==='current'?state.board.snapshot_id:state.archive.snapshot_id,r.name,r.provider,...columns.flatMap(m=>[r.metrics[m.key],r.coverage?.[m.key]??''])])];
    download(lines.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n',`video2world-${state.snapshot}-results.csv`,'text/csv;charset=utf-8');toast('Results exported with snapshot and coverage');
  }
  function caseMedia(c,id){return id==='har'?c.reference:c.models.find(m=>m.id===id);}
  function caseOptions(c, selected) {
    return [c.reference,...c.models.filter(m=>m.video)].map(m=>`<option value="${esc(m.id)}" ${m.id===selected?'selected':''}>${esc(m.name)}</option>`).join('');
  }
  function renderCaseColumn(m, isSource=false) {
    return `<div class="case-video-column"><div class="case-col-top"><span class="col-identity">${esc(m.name)}</span><span class="outcome ${isSource?'source':m.success?'':'failure'}">${isSource?'OBSERVED':m.success?'✓ Success':m.progress!=null?`× Progress ${Number(m.progress).toFixed(2)}`:'× Failed'}</span></div><div class="case-video-frame"><video class="case-video" src="${esc(path(m.video))}" poster="${esc(path(m.poster))}" muted playsinline loop preload="metadata" controls aria-label="${esc(m.name)} case video"></video></div><div class="case-video-caption">${esc(isSource?'The original demonstration provided to the coding agent.':m.description)}${!isSource?`<span>${[m.progress!=null?`Progress ${Number(m.progress).toFixed(2)}`:'',m.final?esc(m.final):''].filter(Boolean).join(' · ')}</span>`:''}</div></div>`;
  }
  function renderCase() {
    const c=state.cases.find(c=>c.id===state.caseId)||state.cases[0];if(!c)return;
    state.caseId=c.id;
    if(!caseMedia(c,state.caseA)?.video)state.caseA='opus5';
    if(!caseMedia(c,state.caseB)?.video)state.caseB='gpt6astra';
    $('#case-tabs').innerHTML=state.cases.map(c=>`<button type="button" role="tab" id="tab-${esc(c.id)}" aria-controls="case-explorer" aria-selected="${c.id===state.caseId}" class="${c.id===state.caseId?'active':''}" data-case="${esc(c.id)}">${esc(c.title)}</button>`).join('');
    $('#case-explorer').setAttribute('role','tabpanel');$('#case-explorer').setAttribute('aria-labelledby',`tab-${c.id}`);
    $('#case-explorer').innerHTML=`<div class="case-summary"><div><h3>${esc(c.title)}</h3><p>${esc(c.description)}</p></div><div class="case-summary-controls"><label>Agent A<select id="case-model-a">${caseOptions(c,state.caseA)}</select></label><label>Agent B<select id="case-model-b">${caseOptions(c,state.caseB)}</select></label></div></div><div class="case-videos">${renderCaseColumn(c.source,true)}${renderCaseColumn(caseMedia(c,state.caseA))}${renderCaseColumn(caseMedia(c,state.caseB))}</div><div class="case-controls"><div><button type="button" id="case-play">Play together ▷</button><button type="button" id="case-pause">Pause Ⅱ</button><button type="button" id="case-reset">Restart ↺</button></div><span class="sync-note">Concurrent playback; clips are not temporally aligned.</span></div>`;
    $$('#case-tabs button').forEach(b=>b.addEventListener('click',()=>{pauseCases();state.caseId=b.dataset.case;renderCase();history.replaceState(null,'',`${location.pathname}${location.search}#case=${c.id===state.caseId?c.id:state.caseId}`);}));
    ['a','b'].forEach(which=>$(`#case-model-${which}`).addEventListener('change',e=>{pauseCases();state[which==='a'?'caseA':'caseB']=e.target.value;renderCase();}));
    $('#case-play').addEventListener('click',()=>{$$('.case-video').forEach(v=>v.play().catch(()=>{}));});
    $('#case-pause').addEventListener('click',pauseCases);
    $('#case-reset').addEventListener('click',()=>{$$('.case-video').forEach(v=>{v.pause();v.currentTime=0;});});
    bindVideoFallbacks();
  }
  function pauseCases(){$$('.case-video').forEach(v=>v.pause());}
  function bindVideoFallbacks() {
    $$('video').forEach(v=>{if(v.dataset.boundError)return;v.dataset.boundError='1';v.addEventListener('error',()=>{v.setAttribute('title','Video unavailable. The verified poster is shown.');});});
  }
  function citation(){return '@article{tang2026video2world,\n  title={Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos},\n  author={Tang, Jinzhou and Zhang, Zijun and Yang, Jing and others},\n  journal={arXiv preprint arXiv:2610.04432},\n  year={2026}\n}';}
  function commonEvents(){
    $('#cite-button')?.addEventListener('click',()=>copy(citation(),'Citation copied'));
    $('.news-brief-close')?.addEventListener('click',closeBrief);
    globalThis.addEventListener?.('hashchange',()=>{if((location.hash||'').startsWith('#news='))openBrief(decodeURIComponent(location.hash.slice(6)));});
    $('#news-brief')?.addEventListener('click',e=>{const d=$('#news-brief');if(e.target===d&&d.getBoundingClientRect){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeBrief();}});
    $('#news-brief')?.addEventListener('close',()=>{if($('#news-brief').open)return;briefRequest++;$$('#news-brief video').forEach(v=>v.pause());if((location.hash||'').startsWith('#news='))history.replaceState(null,'',`${location.pathname}${location.search}`);});
    $('.dialog-close')?.addEventListener('click',()=>{const d=$('#model-dialog');if(d.close)d.close();else d.removeAttribute('open');});
    $('#model-dialog')?.addEventListener('close',()=>{if(location.hash.startsWith('#model='))history.replaceState(null,'',`${location.pathname}${location.search}`);});
    $('#model-dialog')?.addEventListener('click',e=>{if(e.target===$('#model-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)$$('video').forEach(v=>v.pause());});
    bindVideoFallbacks();
  }
  // Homepage: one source video and four coding agents. A missing rollout or task judgment
  // stays explicitly unavailable; it is never rendered as a failed task.
  async function initVerifiedCases(){
    const root=$('#opus-case-explorer');if(!root)return;
    const collection=await fetchJSON('data/frontier-cases.json');
    const list=collection.cases||[];
    if(!list.length){const sec=$('#cases');if(sec)sec.hidden=true;return;}
    const modelIds=[...new Set(collection.model_ids||['gpt-6-1-sol','fable-5-1','meta-muse','opus-5-5'])];
    const modelRow=id=>state.board.models.find(m=>m.id===id)||state.board.pending_models.find(m=>m.id===id);
    const num=i=>String(i+1).padStart(2,'0');
    const metrics=output=>Object.entries({...((typeof output.progress==='number')?{progress:output.progress}:{}),...output.metrics}).filter(([key,value])=>metric(key)&&typeof value==='number'&&Number.isFinite(value)).map(([key,value])=>{
      const m=metric(key),unit=['%','0–1','0–100'].includes(m.unit)?'':` ${m.unit}`;
      return `<div class="cs-metric"><span>${esc(m.label)} ${m.direction==='up'?'↑':'↓'}</span><strong>${format(value,key,m.unit==='%')}${esc(unit)}</strong></div>`;
    }).join('');
    const column=(c,id)=>{
      const output=c.models?.[id]||{},row=modelRow(id),name=row?.name||output.name||id;
      const testing=output.status==='testing';
      const hasMedia=!testing&&output.status!=='awaiting_evidence'&&typeof output.video==='string'&&output.video.trim()!=='';
      const judged=hasMedia&&typeof output.success==='boolean';
      const status=judged?(output.success?'success':'failed'):testing?'testing':hasMedia?'unjudged':'awaiting_evidence';
      const label=judged?(output.success?'Success':'Failed'):testing?'Coming soon':hasMedia?'Outcome pending':'Awaiting evidence';
      const message=testing?'Testing is in progress. Verified case evidence will appear here.':hasMedia?'A rollout is available; the formal task judgment has not been supplied.':'No verified rollout is available for this model on this source video.';
      const media=hasMedia?`<div class="cs-frame"><video data-case-video="${esc(id)}" src="${esc(path(output.video))}"${output.poster?` poster="${esc(path(output.poster))}"`:''} muted playsinline loop preload="metadata" controls aria-label="${esc(name)} simulation for ${esc(c.title)}"></video></div>`:`<div class="cs-placeholder"><span class="cs-placeholder-title">${label}</span><p>${message}</p></div>`;
      const measured=hasMedia?metrics(output):'';
      const evidenceNote=hasMedia?(judged?output.description:message):'';
      return `<article class="cs-run ${judged?(output.success?'is-win':'is-loss'):'is-pending'}" data-model-case="${esc(id)}" data-case-status="${status}"><div class="cs-run-top"><h4 class="cs-run-name">${esc(name)}</h4><span class="cs-outcome" aria-label="${esc(name)}: ${label}">${judged?`<span aria-hidden="true">${output.success?'✓':'✕'}</span> `:''}${label}</span></div>${media}${evidenceNote?`<p class="cs-evidence-note">${esc(evidenceNote)}</p>`:''}${measured?`<div class="cs-metrics">${measured}</div>`:''}</article>`;
    };
    const lede=collection.selection?.note||'Compare four coding agents on the same source video, with their verified rollouts and task judgments.';
    const head=`<div class="cs-head"><span class="eyebrow section-index">${esc($('#opus-case-explorer .section-index')?.textContent||'Case studies')}</span><h2 id="cases-title">${esc($('#cases-title')?.textContent||'Same video. Different worlds.')}</h2><p class="cs-lede">${esc(lede)}</p></div>`;
    const select=i=>{
      const c=list[i];
      $$('#opus-case-explorer video').forEach(v=>v.pause());
      $$('#opus-case-explorer .cs-thumb').forEach((t,j)=>{t.classList.toggle('active',j===i);t.setAttribute('aria-selected',String(j===i));});
      $('#opus-case-explorer .cs-detail').setAttribute('aria-labelledby',`frontier-case-${i}`);
      $('#opus-case-explorer .cs-detail').innerHTML=`<div class="cs-detail-head"><div><span class="eyebrow">Case ${num(i)} / ${esc(c.dataset||c.family)}</span><h3 id="frontier-case-${i}">${esc(c.title)}</h3></div><a class="cs-source" href="${esc(path(c.source.video))}" target="_blank" rel="noopener">View source <span aria-hidden="true">↗</span></a></div><div class="cs-playback"><div><button type="button" data-case-play>Play available simulations ▷</button><button type="button" data-case-reset>Restart ↺</button></div><span class="sync-note">Same input video; rollouts are not time-aligned.</span></div><div class="cs-runs cs-frontier-runs">${modelIds.map(id=>column(c,id)).join('')}</div>`;
      const vids=$$('#opus-case-explorer .cs-detail video');
      const playing=()=>vids.some(v=>!v.paused);
      const sync=()=>root.classList.toggle('is-playing',playing());
      vids.forEach(v=>{v.addEventListener('play',sync);v.addEventListener('pause',sync);});
      const play=root.querySelector('[data-case-play]'),reset=root.querySelector('[data-case-reset]');
      play.disabled=reset.disabled=vids.length===0;
      play.addEventListener('click',()=>vids.forEach(v=>v.play().catch(()=>{})));
      reset.addEventListener('click',()=>vids.forEach(v=>{v.pause();v.currentTime=0;}));
      root.classList.remove('is-playing');
      bindVideoFallbacks();
    };
    root.innerHTML=`${head}<div class="cs-gallery"><div class="cs-library-head"><span class="eyebrow">Source video library</span><a class="cs-all" href="#updates">All model reports <span aria-hidden="true">↗</span></a></div><div class="cs-library" style="--case-columns:${Math.min(5,list.length)}" role="tablist" aria-label="Choose a source video">${list.map((c,i)=>`<button class="cs-thumb" id="frontier-tab-${i}" type="button" role="tab" aria-selected="false" aria-controls="frontier-case-panel" data-case-thumb="${i}" aria-label="Case ${num(i)}: ${esc(c.title)}"><span class="cs-thumb-media"><img src="${esc(path(c.source.poster))}" alt="" loading="lazy"><span class="cs-thumb-play" aria-hidden="true"></span></span><span class="cs-thumb-label"><span>${esc(c.dataset||c.family)}</span>${esc(c.title)}</span></button>`).join('')}</div></div><div class="cs-detail" id="frontier-case-panel" role="tabpanel"></div>`;
    $$('#opus-case-explorer .cs-thumb').forEach((t,i)=>t.addEventListener('click',()=>select(i)));
    select(0);
  }
  async function initHome(){
    // first screen = title + Model News: the news section takes the height left under the title
    // Title + Model News form the first screen: their content is centred in the height under
    // the navigation (equal room above the title and below the news), so the next section starts
    // below the fold and nothing is crammed on tall screens.
    const intro=$('.leaderboard-intro'),news=$('#updates'),rootStyle=document.documentElement?.style;
    const fitIntro=()=>{
      if(!intro||!news||!rootStyle?.setProperty||!globalThis.innerHeight)return;
      rootStyle.setProperty('--first-pad','0px');
      const nav=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-height'))||96;
      const free=globalThis.innerHeight-nav-intro.offsetHeight-news.offsetHeight;
      rootStyle.setProperty('--first-pad',`${Math.max(0,Math.floor(free/2))}px`);
    };
    globalThis.addEventListener?.('resize',fitIntro);
    renderProviders();renderTable();if($('#case-tabs'))renderCase();renderLatest();fitIntro();document.fonts?.ready?.then(fitIntro);await initVerifiedCases();
    $('#model-search').addEventListener('input',e=>{state.search=e.target.value;renderTable();});
    $('#snapshot-current')?.addEventListener('click',()=>setSnapshot('current'));$('#snapshot-paper')?.addEventListener('click',()=>setSnapshot('paper'));
    const viewNotes = {key:'',Geometry:'Does the initialized scene match the observed one? Scene, object shape and size.',Dynamics:'When it runs, does the object move as in the video? Terminal translation and rotation.',Functionality:'Does the task actually get done? Task success and stage progress.',all:'Every reported metric, including fixed-reference diagnostics and build rate.'};
    $$('[data-metric-view]').forEach(b=>b.addEventListener('click',()=>{
      state.metricView=b.dataset.metricView;state.detail=state.metricView==='all';
      $$('[data-metric-view]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});
      if($('#metric-view-note'))$('#metric-view-note').textContent=viewNotes[state.metricView]||'';
      // keep the sort on a visible column; otherwise fall back to V2WScore
      const visible=state.metricView==='all'||state.metricView==='key'?null:getMetrics().filter(m=>m.key==='score'||m.group===state.metricView).map(m=>m.key);
      if(state.metricView==='key'&&!compactKeys.includes(state.sort)||visible&&!visible.includes(state.sort)){state.sort='score';state.sortDir='desc';}
      renderTable();
    }));
    $('#export-csv').addEventListener('click',exportCSV);
    $$('[data-chart]').forEach(b=>b.addEventListener('click',()=>{state.chart=b.dataset.chart;$$('[data-chart]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});renderChart();}));
    let heroPaused=false;$('#hero-play')?.addEventListener('click',()=>{heroPaused=!heroPaused;$$('.hero-video-grid video').forEach(v=>heroPaused?v.pause():v.play().catch(()=>{}));$('#hero-play').textContent=heroPaused?'▷':'Ⅱ';$('#hero-play').setAttribute('aria-label',heroPaused?'Play hero videos':'Pause hero videos');});
    if($('#hero-play')&&matchMedia('(prefers-reduced-motion: reduce)').matches){heroPaused=true;$$('.hero-video-grid video').forEach(v=>v.pause());$('#hero-play').textContent='▷';$('#hero-play').setAttribute('aria-label','Play hero videos');}
    if($('#case-tabs')&&location.hash.startsWith('#case=')){const params=new URLSearchParams(location.hash.slice(1));state.caseId=params.get('case')||state.caseId;state.caseA=params.get('a')||state.caseA;state.caseB=params.get('b')||state.caseB;renderCase();$('#cases').scrollIntoView();}
    if(location.hash.startsWith('#model='))openModel(decodeURIComponent(location.hash.slice(7)));
  }
  // Date the results data was last updated (leaderboard.json snapshot_date), e.g. "Last updated · 09 Oct 2026".
  function lastUpdated(){return `Last updated · ${new Date(`${state.board.snapshot_date}T00:00:00Z`).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'})}`;}
  function renderLatest(){
    const best=[...state.board.models].sort((a,b)=>b.metrics.score-a.metrics.score)[0];
    const highlight=$('.release-highlight');
    if(highlight){
      $('#release-title').textContent=best.name;
      highlight.querySelectorAll('a').forEach(a=>{a.setAttribute('href',path(`updates/${best.id}.html`));a.setAttribute('aria-label',`Read ${best.name} results`);});
      const values=highlight.querySelectorAll('.metric-value');
      values[0].textContent=format(best.metrics.score,'score');values[1].innerHTML=`${format(best.metrics.success,'success')}<span>%</span>`;
      values[2].innerHTML=`${format(best.metrics.build,'build')}<span>%</span>`;
      highlight.querySelectorAll('.metric-name')[2].textContent='Build rate';
    }
    const count=$$('.benchmark-stats dt')[2];if(count)count.textContent=state.board.models.length;
    $('#snapshot-date').textContent=lastUpdated();
    renderNews();renderCampaign();
  }
  // ===== Score composition (2026-10-18) =====
  // V2WScore = mean(F, G, D) x 100 (leaderboard.json score_components, 0-1 each).
  const MIX=[['F','Functionality'],['G','Geometry'],['D','Dynamics']];
  const mixOf=r=>r?.score_components||null;
  // three thin bars for a ranking tile
  function mixMini(r){const c=mixOf(r);if(!c)return '';return `<span class="tile-mix" aria-label="Functionality ${c.F.toFixed(2)}, geometry ${c.G.toFixed(2)}, dynamics ${c.D.toFixed(2)}">${MIX.map(([k])=>`<span><em>${k}</em><i><b style="width:${(c[k]*100).toFixed(1)}%"></b></i></span>`).join('')}</span>`;}
  // grouped bars: one group per component, one bar per model; the first model is the subject
  function mixCompare(rows){
    rows=rows.filter(r=>mixOf(r));if(!rows.length)return '';
    return `<div class="mix-legend">${rows.map((r,i)=>`<button type="button" class="mix-key" data-mix="${esc(r.id)}" style="--k:${i}"><i></i>${esc(r.name)}<small>${format(r.metrics.score,'score')}</small></button>`).join('')}</div><div class="mix-grid">${MIX.map(([k,l])=>`<div class="mix-group"><span class="mix-label">${l}</span>${rows.map((r,i)=>`<span class="mix-row" data-mix="${esc(r.id)}" style="--k:${i}"><i style="--w:${(mixOf(r)[k]*100).toFixed(1)}%"></i><b>${mixOf(r)[k].toFixed(2)}</b></span>`).join('')}</div>`).join('')}</div>`;
  }
  function bindMix(root){
    if(!root?.querySelectorAll)return;
    root.querySelectorAll('[data-mix]').forEach(el=>{const id=el.dataset.mix;const on=v=>root.querySelectorAll('[data-mix]').forEach(x=>x.classList.toggle('is-dim',v&&x.dataset.mix!==id));el.addEventListener('pointerenter',()=>on(true));el.addEventListener('pointerleave',()=>on(false));el.addEventListener('focus',()=>on(true));el.addEventListener('blur',()=>on(false));});
    if('IntersectionObserver' in globalThis){const io=new IntersectionObserver(es=>{if(es[0].isIntersecting){root.classList.add('is-drawn');io.disconnect();}},{threshold:.25});io.observe(root);}else root.classList.add('is-drawn');
  }
  function renderScoreMix(root,ids){
    if(!root)return;const rows=ids.map(id=>state.board.models.find(m=>m.id===id)).filter(Boolean);
    root.innerHTML=mixCompare(rows);bindMix(root);
  }
  // Model news (2026-10-18): each published model result is a news card with a headline, dek,
  // model logo and three numbers. "report" entries open their standalone page; "brief"
  // entries open a dialog with story points, a comparison and the model's own cases.
  const fmtDay=d=>new Date(`${d}T00:00:00Z`).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'});
  function newsTarget(entry){return entry.detail==='brief'?`#news=${entry.id}`:path(entry.href);}
  function newsRank(id){const order=[...state.board.models].sort((a,b)=>b.metrics.score-a.metrics.score);return [competitionRanks(order,'score').get(id)||0,order.length];}
  function newsCard(entry){
    const row=state.board.models.find(m=>m.id===entry.model_id),name=row?.name||entry.title,[rank,total]=row?newsRank(row.id):[0,0];
    const brief=entry.detail==='brief';
    const stats=row?`<span class="nc-stats"><span><b>${format(row.metrics.score,'score')}</b><small>V2WScore</small></span><span><b>${format(row.metrics.success,'success',true)}</b><small>Task success</small></span><span><b>${format(row.metrics.build,'build',true)}</b><small>Build rate</small></span></span>`:'';
    return `<a class="news-card" href="${esc(newsTarget(entry))}" data-news-id="${esc(entry.id)}" data-model="${esc(entry.model_id||'')}" data-detail="${esc(entry.detail||'report')}"${brief?' aria-haspopup="dialog"':''}><span class="nc-cover">${entry.cover?`<img src="${esc(path(entry.cover))}" alt="${esc(entry.cover_alt||'')}" loading="lazy" decoding="async">`:''}${rank?`<span class="nc-rank">#${rank} <small>of ${total}</small></span>`:''}</span><span class="nc-body"><span class="nc-kicker"><span>Model news</span><time datetime="${esc(entry.date)}">${esc(fmtDay(entry.date))}</time></span><span class="nc-model">${esc(name)}</span><strong class="nc-headline">${esc(entry.headline||entry.title)}</strong><span class="nc-dek">${esc(entry.dek||entry.summary)}</span>${stats}<span class="nc-cta">${brief?'Open the brief <i aria-hidden="true">+</i>':'Read the full report <i aria-hidden="true">↗</i>'}</span></span></a>`;
  }
  function pendingCard(m){
    return `<div class="news-card is-pending" data-pending="${esc(m.id)}"><span class="nc-cover"><span class="nc-pending-mark">${esc(m.provider||'')}</span></span><span class="nc-body"><span class="nc-kicker"><span>In evaluation</span><span>Coming soon</span></span><span class="nc-model">${esc(m.name)}</span><strong class="nc-headline">Results are being verified</strong><span class="nc-dek">The news will follow once the evaluation is complete on the same 222-instance protocol.</span></span></div>`;
  }
  function bindNewsCards(root){
    root.querySelectorAll('[data-detail="brief"][data-news-id]').forEach(c=>{if(c.dataset.bound)return;c.dataset.bound='1';c.addEventListener('click',e=>{e?.preventDefault?.();openBrief(c.dataset.newsId);});});
  }
  function renderNews(){
    const published=state.news.entries.filter(n=>n.status==='published').sort((a,b)=>b.date.localeCompare(a.date)||((a.detail==='report')?-1:(b.detail==='report')?1:0));
    const models=published.filter(e=>e.kind==='model_result');
    const pending=state.board.pending_models||[];
    const html=models.map(newsCard).join('')+pending.map(pendingCard).join('');
    const home=$('#latest-model-news');
    if(home){home.innerHTML=newsFront(models,pending);bindNewsCards(home);}
    const feed=$('#news-feed');
    if(feed){feed.innerHTML=html;bindNewsCards(feed);}
    const summary=$('#nav-news-summary');
    if(summary)summary.innerHTML=`<span class="news-nav-kicker">MODEL EVALUATION UPDATES</span>${models.slice(0,4).map(entry=>`<a class="news-nav-report" href="${esc(entry.detail==='brief'?path(`index.html#news=${entry.id}`):path(entry.href))}">${esc(state.board.models.find(m=>m.id===entry.model_id)?.name||entry.title)} <span aria-hidden="true">↗</span></a>`).join('')}`;
    if((location.hash||'').startsWith('#news='))openBrief(decodeURIComponent(location.hash.slice(6)));
  }
  // The homepage keeps the headline ticker and compact model-logo cards; execution videos
  // remain in the case studies and model reports.
  function newsFront(models,pending){
    const rowOf=e=>state.board.models.find(m=>m.id===e.model_id);
    const tick=[...models.map(e=>{const r=rowOf(e);return `<span class="nh-tick"><b>${esc(e.headline||e.title)}</b>${r?` <em>${format(r.metrics.score,'score')} V2WScore · ${format(r.metrics.success,'success',true)} task success</em>`:''}</span>`;}),...pending.map(m=>`<span class="nh-tick"><b>${esc(m.name)}</b> In evaluation <em>Coming soon</em></span>`)].join('<i class="nh-dot" aria-hidden="true">◆</i>');
    const item=e=>{const r=rowOf(e),brief=e.detail==='brief';return `<li><a class="nh-item" href="${esc(newsTarget(e))}" data-news-id="${esc(e.id)}" data-detail="${esc(e.detail||'report')}"${brief?' aria-haspopup="dialog"':''}><span class="nh-story-brand"><span class="nh-model-logo" aria-hidden="true">${r?modelLogo(r):''}</span><span class="nh-story-identity"><span class="nh-item-kicker">${esc(r?.name||e.title)}</span><time datetime="${esc(e.date)}">${esc(fmtDay(e.date))}</time></span>${r?`<span class="nh-item-rank">#${newsRank(r.id)[0]}</span>`:''}</span><strong>${esc(e.headline||e.title)}</strong><span class="nh-item-dek">${esc(e.dek||e.summary)}</span>${r?`<span class="nc-stats"><span><b>${format(r.metrics.score,'score')}</b><small>V2WScore</small></span><span><b>${format(r.metrics.success,'success',true)}</b><small>Task success</small></span><span><b>${format(r.metrics.build,'build',true)}</b><small>Build rate</small></span></span>`:''}<span class="nh-item-cta">${brief?'Open brief <i aria-hidden="true">+</i>':'Read report <i aria-hidden="true">↗</i>'}</span></a></li>`;};
    const pend=m=>`<li><div class="nh-item is-pending" data-pending="${esc(m.id)}"><span class="nh-story-brand"><span class="nh-model-logo" aria-hidden="true">${modelLogo(m)}</span><span class="nh-story-identity"><span class="nh-item-kicker">${esc(m.name)}</span><span>Coming soon</span></span></span><strong>Results are being verified</strong><span class="nh-item-dek">The evaluation is in progress. A model report will follow once the results are verified.</span><span class="nh-pending-label">In evaluation</span></div></li>`;
    return `<div class="nh-ticker" aria-label="Latest headlines"><span class="nh-ticker-tag">Latest</span><div class="nh-ticker-track"><div class="nh-ticker-move"><span class="nh-ticker-set">${tick}<i class="nh-dot" aria-hidden="true">◆</i></span><span class="nh-ticker-set" aria-hidden="true">${tick}<i class="nh-dot" aria-hidden="true">◆</i></span></div></div></div><ol class="nh-list">${models.map(item).join('')}${pending.map(pend).join('')}</ol>`;
  }
  // Brief dialog for a news entry without a standalone page.
  let briefRequest=0;
  async function openBrief(id){
    const entry=state.news.entries.find(e=>e.id===id&&e.detail==='brief'),dialog=$('#news-brief');if(!entry||!dialog)return;
    const request=++briefRequest;
    const row=state.board.models.find(m=>m.id===entry.model_id);if(!row)return;
    const [rank,total]=newsRank(row.id),M=k=>metric(k),byId=i=>state.board.models.find(m=>m.id===i);
    const val=(r,k)=>{const m=M(k);const u=['%','0–1','0–100'].includes(m.unit)?(m.unit==='%'?'%':''):` ${m.unit}`;return `${format(r.metrics[k],k)}${u}`;};
    // comparisons stay inside the model's own line (its earlier versions); the funnel's reference
    // is the median of all evaluated models, not another named model
    const cmp=entry.compare,line=cmp?(cmp.models||[cmp.before,row.id]).map(byId).filter(Boolean):[];
    const better=(k,a,b)=>{const m=M(k);return m.direction==='up'?a>b:a<b;};
    const table=line.length>1?`<div class="nb-block"><h3>${esc(cmp.title)}</h3><table class="nb-compare"><thead><tr><th scope="col">Metric</th>${line.map(r=>`<th scope="col">${esc(r.name)}</th>`).join('')}</tr></thead><tbody>${cmp.metrics.map(k=>`<tr><td>${esc(M(k).short||M(k).label)} <span aria-hidden="true">${M(k).direction==='up'?'↑':'↓'}</span></td>${line.map((r,i)=>{if(r.id!==row.id)return `<td>${val(r,k)}</td>`;const p=line[i-1];return `<td class="${!p||r.metrics[k]===p.metrics[k]?'':better(k,r.metrics[k],p.metrics[k])?'is-up':'is-down'}"><b>${val(r,k)}</b></td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`:'';
    const median=k=>{const v=state.board.models.map(m=>m.metrics[k]).sort((a,b)=>a-b),n=v.length;return n%2?v[(n-1)/2]:(v[n/2-1]+v[n/2])/2;};
    const funnel=entry.funnel?(()=>{const steps=[['build','World builds',v=>v],['progress','Stage progress',v=>v*100],['success','Task success',v=>v]];const ref={metrics:Object.fromEntries(steps.map(([k])=>[k,median(k)]))};return `<div class="nb-block"><h3>${esc(entry.funnel.title)}</h3><div class="nb-funnel">${steps.map(([k,l,f])=>`<div class="nb-step"><span class="nb-step-label">${l}</span><span class="nb-step-bars"><i class="is-a" style="width:${Math.max(1,f(row.metrics[k])).toFixed(1)}%"></i><i class="is-b" style="width:${Math.max(1,f(ref.metrics[k])).toFixed(1)}%"></i></span><span class="nb-step-val"><b>${val(row,k)}</b><small>median ${val(ref,k)}</small></span></div>`).join('')}</div><div class="af2-legend"><span class="is-a">${esc(row.name)}</span><span class="is-b">Median of all ${state.board.models.length} models</span></div></div>`;})():'';
    const points=(entry.points||[]).length?`<div class="nb-points">${entry.points.map(p=>`<div class="nb-point"><span>${esc(p.label)}</span><p>${esc(p.text)}</p></div>`).join('')}</div>`:'';
    const mixBlock=mixOf(row)?`<div class="nb-block nb-mix"><h3>Where the score comes from</h3>${mixCompare([row,...line.filter(r=>r.id!==row.id).reverse()])}</div>`:'';
    $('#news-brief-content').innerHTML=`<header class="nb-head">${entry.cover?`<img class="nb-cover" src="${esc(path(entry.cover))}" alt="${esc(entry.cover_alt||'')}">`:''}<div><span class="nc-kicker"><span>Model news</span><time datetime="${esc(entry.date)}">${esc(fmtDay(entry.date))}</time></span><h2 id="news-brief-title">${esc(entry.headline)}</h2><p>${esc(entry.dek)}</p><div class="nb-stats"><span><b>#${rank}</b><small>of ${total} models</small></span><span><b>${format(row.metrics.score,'score')}</b><small>V2WScore</small></span><span><b>${format(row.metrics.success,'success',true)}</b><small>Task success</small></span><span><b>${format(row.metrics.build,'build',true)}</b><small>Build rate</small></span></div></div></header>${points}${mixBlock}${funnel}${table}<div class="nb-block nb-cases" id="news-brief-cases"></div><footer class="nb-foot"><span>Snapshot as of ${esc(fmtDay(entry.date))} · 222 instances</span><a class="text-link" href="${esc(path('index.html#leaderboard'))}" data-brief-board>See the full ranking ↗</a></footer>`;
    bindMix($('#news-brief-content .nb-mix'));
    if(dialog.showModal&&!Object.hasOwn(dialog.attrs||{},'open')&&!dialog.open)dialog.showModal();else if(!dialog.showModal)dialog.setAttribute('open','');
    if(history.replaceState&&location.hash!==`#news=${id}`)history.replaceState(null,'',`${location.pathname}${location.search}#news=${id}`);
    $('[data-brief-board]')?.addEventListener('click',()=>{dialog.close?.();});
    if(entry.cases){
      const col=await fetchJSON(entry.cases);if(request!==briefRequest)return;const list=[...(col.cases||[])].sort((a,b)=>(b.model?.success?1:0)-(a.model?.success?1:0));
      const box=$('#news-brief-cases');if(!box||!list.length){if(box)box.hidden=true;return;}
      box.innerHTML=`<h3>${esc(row.name)} in action</h3><p class="nb-note">Real rollouts on shared source videos; successes first. Formal task judgments.</p><div class="nb-tabs" role="tablist">${list.map((c,i)=>`<button type="button" role="tab" data-brief-case="${i}" aria-selected="false">${c.model?.success?'<i class="ok" aria-hidden="true">✓</i>':'<i class="no" aria-hidden="true">✕</i>'}${esc(c.title)}</button>`).join('')}</div><div class="nb-case" id="news-brief-case"></div>`;
      const show=i=>{const c=list[i],m=c.model;box.querySelectorAll('[data-brief-case]').forEach(b=>{const on=+b.dataset.briefCase===i;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on));});box.querySelectorAll('video').forEach(v=>v.pause());
        $('#news-brief-case').innerHTML=`<div class="nb-clip"><span class="nb-clip-top">Source video <em>${esc(c.dataset||'')}</em></span><video src="${esc(path(c.source.video))}" poster="${esc(path(c.source.poster))}" muted playsinline loop controls preload="metadata"></video></div><div class="nb-clip ${m.success?'is-win':'is-loss'}"><span class="nb-clip-top">${esc(row.name)} <em class="nb-verdict">${m.success?'✓ Task success':'✕ Task failed'}</em></span><video src="${esc(path(m.video))}" poster="${esc(path(m.poster))}" muted playsinline loop controls preload="metadata"></video></div>`;bindVideoFallbacks();};
      box.querySelectorAll('[data-brief-case]').forEach(b=>b.addEventListener('click',()=>show(+b.dataset.briefCase)));show(0);
    }
  }
  function closeBrief(){briefRequest++;const d=$('#news-brief');if(!d)return;d.querySelectorAll('video').forEach(v=>v.pause());if(d.close)d.close();else d.removeAttribute('open');}
  function renderCampaign(){
    const campaign=state.news.campaign;
    const title=$('#campaign-title');if(!title)return;
    title.textContent=campaign.title;
    const rows=campaign.model_ids.map(id=>state.board.models.find(m=>m.id===id)||state.board.pending_models.find(m=>m.id===id)).filter(Boolean);
    const complete=rows.filter(r=>r.metrics!=null).length;
    $('#campaign-status').textContent=`${complete} of ${campaign.model_ids.length} results available`;
    const roundup=state.news.entries.find(n=>n.kind==='roundup'&&n.status==='published'&&n.href===campaign.roundup_href);
    const message=$('#campaign-message');if(message)message.textContent=roundup?'The joint comparison is published. Read it alongside the individual model reports.':complete===campaign.model_ids.length?'All four result rows are available. The joint article will follow after protocol and evidence review.':'Claude, Gemini, Muse and GPT: the four-model comparison will follow once all results are verified on a compatible protocol.';
    const publication=$('#campaign-publication');if(publication)publication.innerHTML=roundup?`<a class="text-link" href="${esc(path(roundup.href))}">Read the joint analysis ↗</a>`:esc(campaign.publication_note||'Joint article in preparation.');
  }

  // Model update pages: one verified case at a time, chosen with pill buttons (the homepage shows a
  // single featured case and links here). #case-N selects a case directly.
  function renderCaseSwitcher(root,collection,modelName,kicker){
    const list=collection.cases||[];if(!root||!list.length)return;
    const sel=collection.selection;
    const intro=sel?`<p class="case-selection">${esc(`Of the ${sel.common_cases} cases evaluated for both models, ${sel.matching_cases} are completed by ${modelName} and failed by ${sel.baseline_model} on the same source video. These ${sel.selected_cases} cover different task families. Outcomes are the formal task-success judgments.`)}</p>`:'';
    root.innerHTML=`<div class="section-heading"><div><span class="eyebrow section-index">${esc(kicker)}</span><h2>Inspect the new model's execution.</h2></div></div>${collection.intro?`<p class="case-lead">${esc(collection.intro)}</p>`:''}${intro}<div class="case-tabs" role="tablist" aria-label="Choose a case">${list.map((c,i)=>`<button type="button" role="tab" data-case-index="${i}" aria-selected="false">${esc(c.title)}</button>`).join('')}</div><div class="case-explorer" id="model-case-view" role="tabpanel"></div>`;
    const show=i=>{
      const c=list[i];
      root.querySelectorAll('[data-case-index]').forEach(b=>{const on=+b.dataset.caseIndex===i;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on));});
      root.querySelectorAll('video').forEach(v=>v.pause());
      $('#model-case-view').innerHTML=`<div class="case-summary"><div><span class="eyebrow section-index">${String(i+1).padStart(2,'0')} / ${String(list.length).padStart(2,'0')}</span><h3>${esc(c.title)}</h3><p>${esc(c.description)}</p></div><span class="outline-tag">${esc(c.dataset||c.family)}</span></div><div class="case-videos">${renderCaseColumn({...c.source,name:'Source video'},true)}${renderCaseColumn({...c.model,name:modelName})}${c.baseline?renderCaseColumn(c.baseline):''}</div>${c.analysis?`<p class="case-analysis">${esc(c.analysis)}</p>`:''}<div class="case-controls"><div><button type="button" data-case-play>Play together ▷</button><button type="button" data-case-reset>Restart ↺</button></div><span class="sync-note">Clips play side by side; they are not time-aligned.</span></div>`;
      $('[data-case-play]').addEventListener('click',()=>$$('#model-case-view video').forEach(v=>v.play().catch(()=>{})));
      $('[data-case-reset]').addEventListener('click',()=>$$('#model-case-view video').forEach(v=>{v.pause();v.currentTime=0;}));
      bindVideoFallbacks();
    };
    root.querySelectorAll('[data-case-index]').forEach(b=>b.addEventListener('click',()=>{const i=+b.dataset.caseIndex;show(i);history.replaceState(null,'',`${location.pathname}${location.search}#case-${i+1}`);}));
    const m=/^#case-(\d+)$/.exec(location.hash||'');const start=m?Math.min(list.length,Math.max(1,+m[1]))-1:0;
    show(start);
    if(m||location.hash==='#model-update-cases')root.scrollIntoView?.();
  }
  // Numbers inside page copy come from leaderboard.json:
  //   data-v="model:metric"            the formatted value (with % for percentage metrics)
  //   data-diff="modelA:modelB:metric" A − B, signed (pp for percentages)
  //   data-ratio="modelA:modelB:metric" A ÷ B, as "2.56×"
  //   #success-bars[data-models]       the task-success bar chart, in the listed order
  function bindNumbers(){
    const row=id=>state.board.models.find(m=>m.id===id);
    const val=(id,k)=>row(id)?.metrics?.[k];
    $$('[data-v]').forEach(el=>{const [id,k]=el.dataset.v.split(':');const v=val(id,k);if(v!=null)el.textContent=format(v,k,metric(k)?.unit==='%');});
    $$('[data-diff]').forEach(el=>{const [a,b,k]=el.dataset.diff.split(':');const d=val(a,k)-val(b,k);if(Number.isFinite(d))el.textContent=`${d>0?'+':''}${d.toFixed(metric(k)?.unit==='%'?1:2)}${metric(k)?.unit==='%'?' pp':''}`;});
    $$('[data-ratio]').forEach(el=>{const [a,b,k]=el.dataset.ratio.split(':');const r=val(a,k)/val(b,k);if(Number.isFinite(r))el.textContent=`${r.toFixed(2)}×`;});
    const bars=$('#success-bars');
    if(bars){
      const ids=bars.dataset.models.split(':'),vals=ids.map(id=>val(id,'success')),top=Math.ceil(Math.max(...vals)/5)*5+5;
      const lead=ids[vals.indexOf(Math.max(...vals))];
      bars.querySelector('.result-bars').innerHTML=ids.map((id,i)=>`<div class="result-bar-row${id===lead?' result-winner':''}"><span>${esc(row(id).name.replace(/^Claude /,''))}</span><span class="result-bar-lane"><i class="result-bar" style="width:${(vals[i]/top*100).toFixed(1)}%"></i></span><b>${format(vals[i],'success',true)}</b></div>`).join('')+`<div class="result-bar-scale"><span>0%</span><span>${top}%</span></div>`;
      bars.setAttribute('aria-label',`Task success: ${ids.map((id,i)=>`${row(id).name}, ${format(vals[i],'success',true)}`).join('; ')}.`);
    }
  }
  // Model pages: number the visible top-level sections in order ("01 / …"), as on the main site.
  function numberSections(){
    let n=0;
    $$('.update-page > section[data-rail]').forEach(sec=>{
      if(sec.hidden)return;
      const eb=sec.querySelector('.section-index');if(!eb)return;
      const base=eb.dataset.label||eb.textContent.replace(/^\d+\s*\/\s*/,'').trim();
      eb.dataset.label=base;eb.textContent=`${String(++n).padStart(2,'0')} / ${base}`;
    });
  }
  async function initUpdate(){
    bindNumbers();
    const opus=state.board.models.find(m=>m.id==='opus-5-5');
    $('#model-results-body').innerHTML=['opus-5-5','fable-5-1','gpt-6-astra','opus-5'].map(id=>state.board.models.find(m=>m.id===id)).map(r=>`<tr class="${r.featured?'featured':''}"><td>${esc(r.name)}</td><td>${format(r.metrics.score,'score')}</td><td>${format(r.metrics.success,'success',true)}</td><td>${format(r.metrics.progress,'progress')}</td></tr>`).join('');
    $('#share-result').addEventListener('click',()=>copy(location.href.split('#')[0],'Result link copied'));
    $('#download-result-card').addEventListener('click',()=>{const a=document.createElement('a');a.href=path('assets/opus-5-5-social.png');a.download='video2world-opus-5-5.png';a.click();});
    $('#download-snapshot').addEventListener('click',()=>download(JSON.stringify(state.board,null,2),'video2world-v2-snapshot.json','application/json'));
    const collection=await fetchJSON('data/opus-5-5-cases.json');
    renderMetricCoverage(opus);
    renderScoreMix($('#score-mix'),['opus-5-5','fable-5-1','opus-5']);
    renderModelAnalysis(opus.id,collection);
    if($('#model-cases-status'))$('#model-cases-status').hidden=collection.cases?.length>0;
    if(collection.cases?.length){
      $('#model-update-cases').hidden=false;
      renderCaseSwitcher($('#model-update-cases'),collection,opus.name,'EXECUTION CASES');
      const ch=$('#model-update-cases h2');if(ch)ch.textContent='Execution, side by side.';
      bindVideoFallbacks();
    }
    // Remaining failures (both models fail on the same video): shown once verified clips are supplied.
    if(collection.failure_cases?.length&&$('#failure-cases')){
      $('#failure-cases').hidden=false;
      renderCaseSwitcher($('#failure-cases'),{cases:collection.failure_cases,intro:collection.failure_intro},opus.name,'A REMAINING FAILURE');
      const h=$('#failure-cases h2');if(h)h.textContent='Where Opus 5.5 still fails.';
    }
  }
  function renderMetricCoverage(row){
    const target=$('#generic-metric-coverage');if(!target)return;
    // Breakdown variant (Opus page): metrics already shown in the comparison table are left out,
    // coverage reads "199 / 222", and a note summarises coverage of the headline metrics.
    const skip=(target.dataset.exclude||'').split(',').filter(Boolean),breakdown=target.dataset.style==='breakdown',N=state.board.instances;
    const cov=k=>row.coverage?.[k]!=null?`${breakdown?'':'n='}${row.coverage[k]} / ${N}`:'Not supplied';
    const list=state.board.metrics.filter(m=>!skip.includes(m.key));
    if(breakdown)list.sort((a,b)=>(b.key==='build')-(a.key==='build'));  // build rate leads the breakdown
    const unitOf=m=>['%','0–1','0–100'].includes(m.unit)?'':breakdown&&m.unit==='deg'?'°':` ${esc(m.unit)}`;
    target.innerHTML=list.map(m=>`<tr><td>${esc(m.label)}</td><td>${format(row.metrics[m.key],m.key,m.unit==='%')}${unitOf(m)}</td><td>${cov(m.key)}</td></tr>`).join('');
    const note=$('#coverage-note');
    if(note){const c=k=>row.coverage?.[k];note.textContent=`Coverage varies by metric. Task success: ${c('success')??'—'}/${N}; stage progress: ${c('progress')??'—'}/${N}. ${c('score')==null?'V2W Score coverage was not provided.':`V2W Score: ${c('score')}/${N}.`}`;}
  }
  // Corner glyph for a finding card: a small dot pattern, different per card.
  function analysisGlyph(i){
    const P=[[0,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1],[1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1],[0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1]][i%4];
    return `<svg class="analysis-glyph" viewBox="0 0 16 16" aria-hidden="true">${P.map((on,k)=>on?`<circle cx="${2+(k%4)*4}" cy="${2+Math.floor(k/4)*4}" r="1.2"/>`:'').join('')}</svg>`;
  }
  // Finding figure: each metric as a large value with the baseline beside it ("vs. 20.2% · Fable
  // 5.1"). Bars only where the two values differ clearly (>= 20% relative), and then both models
  // get a bar (this model filled, the baseline outlined); near-equal values are called out instead.
  function analysisFigure(row,keys,baselineId='fable-5-1'){
    const ref=state.board.models.find(m=>m.id===baselineId&&m.id!==row.id);
    const refName=ref?ref.name.replace(/^Claude /,''):'';
    const items=keys.map(k=>{
      const m=metric(k);if(!m||row.metrics[k]==null)return '';
      const a=row.metrics[k],b=ref?.metrics?.[k],pct=m.unit==='%';
      const unit=!['%','0–1','0–100'].includes(m.unit)?esc(m.unit):'';
      const rel=b!=null?Math.abs(a-b)/(Math.max(Math.abs(a),Math.abs(b))||1):0;
      const val=v=>`${format(v,k,pct)}${unit?` ${unit}`:''}`;
      const bars=b!=null&&rel>=0.2?(()=>{const max=Math.max(a,b)||1;return `<div class="af2-bars"><i class="is-a" style="width:${(a/max*100).toFixed(1)}%"></i><i class="is-b" style="width:${(b/max*100).toFixed(1)}%"></i><div class="af2-legend"><span class="is-a">${esc(row.name.replace(/^Claude /,''))}</span><span class="is-b">${esc(refName)}</span></div></div>`;})():'';
      const note=b==null?'':a===b?'Equal':rel<0.03?'Nearly identical':'';
      // fixed rows per cell (label / value / baseline / bars or note / coverage) so cells line up;
      // the sample count sits at the foot of its own cell, never next to a neighbour's label
      return `<div class="af2-item"><div class="af2-head">${esc(m.short||m.label)} <span aria-hidden="true">${m.direction==='up'?'↑':'↓'}</span></div><div class="af2-value">${format(a,k,pct)}${unit?`<small>${unit}</small>`:''}</div>${b!=null?`<div class="af2-ref">vs. ${val(b)} · ${esc(refName)}</div>`:''}<div class="af2-extra">${bars}${note?`<div class="af2-note">${note}</div>`:''}</div>${row.coverage?.[k]!=null?`<div class="af2-cov"><span>n = ${row.coverage[k]} / ${state.board.instances}</span>${ref?.coverage?.[k]!=null?`<span>${esc(refName)}: n = ${ref.coverage[k]} / ${state.board.instances}</span>`:''}</div>`:''}</div>`;
    }).join('');
    return items?`<div class="analysis-figure af2">${items}</div>`:'';
  }
  function renderModelAnalysis(id,collection={}){
    const target=$('#model-analysis'),report=state.analysis.models[id];if(!target||!report)return;
    const row=state.board.models.find(m=>m.id===id),findings=row?report.findings||[]:[];
    target.hidden=false;
    target.innerHTML=`<div class="section-heading"><div><span class="eyebrow section-index">${row?'CAPABILITIES & LIMITS':'THE UPCOMING REPORT'}</span><h2>${row?'What the evidence shows.':'What we will examine.'}</h2><p>${esc(report.intro)}</p></div></div>${findings.length?(()=>{
      // Editorial layout: findings grouped (Improvements / Limitations); each row is a vertical label,
      // a hairline, then the text with its metric figure; label side alternates; no numbering.
      const groups=[];findings.forEach((f,fi)=>{const g=f.group||(fi<2?'Improvements':'Limitations & trade-offs');let G=groups.find(x=>x.name===g);if(!G)groups.push(G={name:g,items:[]});G.items.push(f);});
      let k=0;
      // Groups whose findings carry a stamp (findings[].stamp) show it on the tickets instead of a
      // divider heading; the group name stays available to screen readers.
      const stamped=g=>G_STAMPED.has(g);
      const G_STAMPED=new Set(groups.filter(G=>G.items.every(f=>f.stamp)).map(G=>G.name));
      return `<div class="findings-editorial">${groups.map(G=>`<div class="fe-group ${stamped(G.name)?'is-stamped':''}">${stamped(G.name)?`<h3 class="sr-only">${esc(G.name)}</h3>`:`<div class="fe-group-head"><span class="fe-rule"></span><h3 class="fe-group-title">${esc(G.name)}</h3><span class="fe-rule"></span></div>`}${G.items.map(f=>`<article class="fe-row ${k++%2?'is-flip':''}"><div class="fe-ticket"><span class="fe-tag analysis-label">${esc(f.label||f.title)}</span><span class="fe-line" aria-hidden="true"></span><div class="fe-body"><div class="fe-text"><h4>${esc(f.title)}</h4><p>${esc(f.body)}</p><div class="fe-foot">${f.stamp?`<span class="fe-stamp tone-${esc(f.stamp_tone||'improvement')}" aria-hidden="true">${esc(f.stamp)}</span>`:''}<div class="analysis-links">${(f.case_ids||[]).map(caseId=>{const i=(collection.cases||[]).findIndex(c=>c.id===caseId);return i<0?'':`<a class="analysis-case-link" href="#case-${i+1}" data-analysis-case="${i}">Inspect ${esc(collection.cases[i].title)} ↘</a>`;}).join('')}</div></div></div><div class="fe-figure">${analysisFigure(row,f.metric_keys||[],f.baseline_model_id)}</div></div></div></article>`).join('')}</div>`).join('')}</div>`;
    })():''}${report.evidence_priorities?.length?`<ul class="pending-evidence">${report.evidence_priorities.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}${report.limitations?.length?`<aside class="analysis-limits"><h3>${row?'Evaluation Notes':'Results pending'}</h3><ul>${report.limitations.map(p=>`<li>${esc(p)}</li>`).join('')}</ul></aside>`:''}`;
    // All tickets share the height of the tallest one (desktop layout only).
    const tickets=$$('#model-analysis .fe-ticket');
    const equalize=()=>{if(!tickets.length||!tickets[0].getBoundingClientRect)return;tickets.forEach(t=>{t.style.minHeight='';});if((globalThis.innerWidth||0)<=760)return;const h=Math.max(...tickets.map(t=>t.offsetHeight||0));/* offsetHeight ignores the scroll-flow scale */tickets.forEach(t=>{t.style.minHeight=`${Math.ceil(h)}px`;});};
    equalize();globalThis.addEventListener?.('resize',equalize);document.fonts?.ready?.then(equalize);
    $$('[data-analysis-case]').forEach(a=>a.addEventListener('click',()=>{
      const button=$$('#model-update-cases [data-case-index]')[Number(a.dataset.analysisCase)];if(button)button.click();
      $('#model-update-cases')?.scrollIntoView({behavior:'smooth',block:'start'});
    }));
  }
  async function initGenericUpdate(){
    const id=document.body.dataset.model;
    const row=state.board.models.find(m=>m.id===id),pending=state.board.pending_models.find(m=>m.id===id);
    if(!row&&!pending)throw new Error('Model update has no matching result or testing record');
    $('#generic-model-title').textContent=(row||pending).name;
    $('#generic-results').hidden=!row;
    $('#generic-pending').hidden=!!row;
    $$('[data-result-only]').forEach(el=>{el.hidden=!row;});
    document.body.dataset.modelStatus=row?'results':'testing';
    if(!row){
      $('#generic-model-intro').textContent=state.analysis.models[id]?.intro||`${pending.name} is being tested. Results and case evidence will be reported after verification.`;
      $('#generic-date').textContent='Status · Testing — results pending';
      renderModelAnalysis(id);
      return;
    }
    const update=state.board.updates.find(u=>u.model_id===id)||{};
    $('#generic-model-intro').textContent=`${row.name} records ${format(row.metrics.score,'score')} V2WScore and ${format(row.metrics.success,'success',true)} task success in this result set. Inspect the component metrics and supporting evidence below.`;
    $('#generic-score').innerHTML=`${format(row.metrics.score,'score')}<small>/ 100</small>`;
    $('#generic-success').textContent=format(row.metrics.success,'success',true);$('#generic-build').textContent=format(row.metrics.build,'build',true);
    $('#generic-date').textContent=`${update.date_label||'Snapshot as of'} · ${update.date||state.board.snapshot_date}`;
    const baselines=(update.compare_ids||[...state.board.models].filter(m=>m.id!==id).sort((a,b)=>b.metrics.score-a.metrics.score).slice(0,2).map(m=>m.id)).map(i=>state.board.models.find(m=>m.id===i)).filter(Boolean);
    $('#model-results-body').innerHTML=[row,...baselines].map(r=>`<tr class="${r.id===id?'featured':''}"><td>${esc(r.name)}</td><td>${format(r.metrics.score,'score')}</td><td>${format(r.metrics.success,'success',true)}</td><td>${format(r.metrics.progress,'progress')}</td></tr>`).join('');
    renderMetricCoverage(row);
    $('#share-result').addEventListener('click',()=>copy(location.href.split('#')[0],'Result link copied'));
    $('#download-result-card').addEventListener('click',()=>{const a=document.createElement('a');a.href=path(`assets/${id}-social.png`);a.download=`video2world-${id}.png`;a.click();});
    $('#download-snapshot').addEventListener('click',()=>download(JSON.stringify({snapshot_id:state.board.snapshot_id,...row},null,2),`${id}-result.json`,'application/json'));
    const collection=await fetchJSON(`data/${id}-cases.json`);
    renderModelAnalysis(id,collection);
    if(collection.cases?.length){
      $('#model-cases-status').hidden=true;$('#model-update-cases').hidden=false;
      renderCaseSwitcher($('#model-update-cases'),collection,row.name,'VERIFIED CASES');
      bindVideoFallbacks();
    }
  }
  async function main(){
    try {
      [state.board,state.archive,state.cases,state.news,state.analysis]=await Promise.all([fetchJSON('data/leaderboard.json'),fetchJSON('data/paper-results.json'),fetchJSON('data/cases.json'),fetchJSON('data/news.json'),fetchJSON('data/model-analysis.json')]);
      commonEvents();
      if(document.body.dataset.page==='home')await initHome();else if(document.body.dataset.page==='opus'){await initUpdate();renderCampaign();}else if(document.body.dataset.page==='news'){renderNews();renderCampaign();}else if(document.body.dataset.page==='model'){await initGenericUpdate();renderNews();}
      numberSections();
      document.body.dataset.ready='true';
    } catch(error){
      console.error(error);const target=$('#table-body')||$('#model-results-body');if(target)target.innerHTML='<tr><td colspan="7" class="error-note">Results could not load. Please reload the page or download the supplied snapshot.</td></tr>';toast('Some data could not load. Please reload.');
      document.body.dataset.ready='error';
    }
  }
  main();
})();
