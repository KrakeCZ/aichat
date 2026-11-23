/* Git timeline view. No external dependencies or executing repository code. */
'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let documentHistory = null;
let index = 0;
const safeText = x => String(x ?? '');
function el(tag,attrs={},txt){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(txt!==undefined)n.textContent=safeText(txt);return n;}
function isConfirmed(e){return e.confidence === 'resolved_static' || e.confidence === 'exact';}
function isDisplayed(e){return !$('certainty').checked || isConfirmed(e);}
function edgeText(e){return `${e.source} → ${e.target} [${e.type}] (${e.confidence})`;}
function proofView(e){
 const ev=(e.evidence||[])[0];
 let msg=`${edgeText(e)}\nAnalyzátor: ${e.analyzer || 'nezjištěn'}`;
 if(ev)msg+=`\nDůkaz: ${ev.path}:${ev.line}\n${ev.snippet||''}`;
 const area=$('inspect');area.replaceChildren(document.createTextNode(msg));
 if(documentHistory.repo && ev && ev.path && Number(ev.line)>0){
  const a=document.createElement('a');const sha=documentHistory.commits[index].sha;
  a.href=`https://github.com/${documentHistory.repo}/blob/${sha}/${ev.path.split('/').map(encodeURIComponent).join('/')}#L${Number(ev.line)}`;
  a.textContent='Otevřít důkaz v této Git revizi';a.rel='noopener noreferrer';a.target='_blank';area.append(document.createElement('hr'),a);
 }
}
function selected(){return documentHistory?.commits[index];}
function showText(id,s){$(id).textContent=safeText(s);}
function drawHistory(){
 const svg=$('historyChart');svg.replaceChildren();const commits=documentHistory.commits;
 if(!commits.length)return;
 const x=i=>43+(i*Math.max(0,805/(commits.length-1||1))), ymax=Math.max(1,...commits.map(c=>Math.max(c.snapshot.files,c.snapshot.edges)));
 const y=v=>149-(v/ymax)*117;
 for(const [field,color] of [['files','#66d2b1'],['edges','#8bbaf7']]){
  let pts=commits.map((c,i)=>`${x(i)},${y(c.snapshot[field])}`).join(' ');
  svg.append(el('polyline',{points:pts,stroke:color,fill:'none','stroke-width':2.5}));
  commits.forEach((c,i)=>{const dot=el('circle',{cx:x(i),cy:y(c.snapshot[field]),r:i===index?6:3,fill:color,tabindex:0});dot.addEventListener('click',()=>setIndex(i));svg.append(dot);});
 }
 svg.append(el('text',{x:16,y:23,fill:'#b5c9e0'},'Počet'));
 svg.append(el('text',{x:15,y:165,fill:'#66d2b1'},'Soubory'));
 svg.append(el('text',{x:104,y:165,fill:'#8bbaf7'},'Vztahy'));
 svg.append(el('text',{x:850,y:165,fill:'#b5c9e0','text-anchor':'end'},`${commits.length} snímků`));
 svg.append(el('line',{x1:x(index),x2:x(index),y1:27,y2:150,stroke:'#f6ca78','stroke-dasharray':'3 4'}));
}
function renderChangedFiles(){
 const target=$('changedFiles');target.replaceChildren();const c=selected();
 if(index===0){target.textContent='Referenční snímek: bez porovnání s předchozím vybraným commitem.';return;}
 const statuses=c.git_changes||[];
 if(!statuses.length){target.textContent='Git v těchto snímcích nehlásí změněné soubory.';return;}
 const added=document.createElement('div');added.className='sub';added.textContent=`Změny Git: ${statuses.length} · +${c.delta.added_files.length} / −${c.delta.removed_files.length} cest v grafu`;target.append(added);
 statuses.forEach(ch=>{
  const item=document.createElement('button');item.className='item';item.style.textAlign='left';
  item.textContent=`${ch.status} · ${ch.old?ch.old+' → ':''}${ch.path}`;
  item.onclick=()=>{showText('inspect',`${ch.status} · ${ch.old ? ch.old+' → ':''}${ch.path}\nGit změna názvu/obsahu neprokazuje změnu chování.\nCommit: ${c.sha}`);$('fileFilter').value=ch.path;drawGraph();};target.append(item);
 });
}
function renderChangedEdges(){
 const box=$('changedEdges');box.replaceChildren();const d=selected().delta;
 if(index===0){box.textContent='Toto je výchozí stav vybrané historie. Pro skutečnou změnu vyber další commit.';return;}
 const items=[...d.added_edges.map(e=>({e,tag:'+'})),...d.removed_edges.map(e=>({e,tag:'−'})),...(d.changed_evidence||[]).map(x=>({e:x.after,tag:'~'}))].filter(o=>isDisplayed(o.e));
 if(!items.length){box.textContent='Žádné změny vztahů ve vybraném filtru.';return;}
 items.forEach(({e,tag})=>{const btn=document.createElement('button');btn.className='item';btn.style.textAlign='left';
  const span=document.createElement('span');span.className=tag==='+'?'plus':tag==='−'?'minus':'muted';span.textContent=tag==='~'?'~ změna důkazu · ':tag+' ';
  btn.append(span,document.createTextNode(edgeText(e)));btn.onclick=()=>proofView(e);box.append(btn);
 });
}
function drawGraph(){
 const svg=$('graphSvg');svg.replaceChildren();const c=selected();if(!c)return;
 const mode=$('graphMode').value;
 const items=(mode==='delta'&&index>0)
  ? [...c.delta.added_edges.map(e=>({e,status:'added'})),...c.delta.removed_edges.map(e=>({e,status:'removed'}))]
  : c.edges.map(e=>({e,status:'present'}));
 const filter=$('fileFilter').value.trim().toLowerCase();
 const filtered=items.filter(({e})=>isDisplayed(e)&&(!filter||e.source.toLowerCase().includes(filter)||e.target.toLowerCase().includes(filter)));
 const limit=130;const display=filtered.slice(0,limit);
 const files=Array.from(new Set(display.flatMap(({e})=>[e.source,e.target])));
 const cx=447,cy=243, rx=344,ry=186;
 const positions=new Map(files.map((f,i)=>[f,{x:cx+rx*Math.cos((i/files.length)*Math.PI*2-Math.PI/2),y:cy+ry*Math.sin((i/files.length)*Math.PI*2-Math.PI/2)}]));
 for(const {e,status} of display){const from=positions.get(e.source),to=positions.get(e.target);if(!from||!to)continue;
  const stroke=status==='added'?'#5fe0b2':status==='removed'?'#ff9a9a':'#75acde';
  const line=el('line',{x1:from.x,y1:from.y,x2:to.x,y2:to.y,stroke,'stroke-width':1.8,'stroke-opacity':.65,
                      'stroke-dasharray':isConfirmed(e)?'':'4 4',style:'cursor:pointer'});
  line.addEventListener('click',()=>proofView(e));svg.append(line);
 }
 for(const [f,p] of positions){const dot=el('circle',{cx:p.x,cy:p.y,r:7,fill:'#f3c77a',stroke:'#ecf7ff','stroke-width':1});
  dot.addEventListener('click',()=>{showText('inspect',f+'\nUzlová identita je cesta k souboru ve vybraném commitu.');$('fileFilter').value=f;drawGraph();});svg.append(dot);
  const lab=el('text',{x:p.x+(p.x>=cx?12:-12),y:p.y+3,'text-anchor':p.x>=cx?'start':'end'},f.split('/').slice(-1)[0]);svg.append(lab);
 }
 showText('graphStatus',`${filtered.length} viditelných vztahů (${display.length} vykreslených), ${files.length} souborů. ${filtered.length>limit?'Náhled je omezen na '+limit+' hran; úplný graf je v JSON.':''}`);
 if(!filtered.length)svg.append(el('text',{x:450,y:250,fill:'#b5c9e0','text-anchor':'middle'},index===0&&mode==='delta'?'Vyber následující commit.':'Žádné vztahy pro tento filtr.'));
}
function setIndex(i){
 index=Math.max(0,Math.min(documentHistory.commits.length-1,Number(i)||0));
 $('commitRange').value=String(index);$('commitSelect').value=String(index);
 const c=selected();showText('commitTitle',c.title);showText('commitMeta',`${c.sha.slice(0,12)} · ${c.time_utc} · ${c.author} · ${index+1}/${documentHistory.commits.length}`);
 showText('countFiles',c.snapshot.files);showText('countEdges',c.snapshot.edges);
 showText('countAdded',index?c.delta.added_edges.length:'—');showText('countRemoved',index?c.delta.removed_edges.length:'—');showText('countChanged',index?(c.delta.changed_evidence||[]).length:'—');
 const warning=c.snapshot.extraction.complete?'':' · Některé velké soubory nebyly indexovány; graf je neúplný.';
 showText('loadStatus',`Analyzovaná Git revize ${c.sha.slice(0,12)}${warning}. ${documentHistory.sample_stride>1?`Vzorek každého ${documentHistory.sample_stride}. commitu; mezilehlé změny mohou chybět. `:""}Výsledky jsou statické, runtime vazby mohou chybět.`);
 $('loadStatus').className=c.snapshot.extraction.complete?'notice':'warning';
 renderChangedFiles();renderChangedEdges();drawHistory();drawGraph();
}
async function init(){
 try{
  const response=await fetch('./git-timeline.json',{cache:'no-store'});
  if(!response.ok)throw new Error('git-timeline.json není v tomto výstupu. Spusť atlas.py history pro Git repozitář.');
  documentHistory=await response.json();
  if(!documentHistory.commits?.length)throw new Error('Historie neobsahuje žádné analyzované commity.');
  const list=$('commitSelect');documentHistory.commits.forEach((c,i)=>{const op=document.createElement('option');op.value=String(i);op.textContent=`${i+1}. ${c.sha.slice(0,8)} · ${c.title.slice(0,58)}`;list.append(op);});
  $('commitRange').max=String(documentHistory.commits.length-1);
  $('commitRange').addEventListener('input',ev=>setIndex(ev.target.value));
  list.addEventListener('change',ev=>setIndex(ev.target.value));
  for(const id of ['graphMode','certainty'])$(id).addEventListener('change',()=>{renderChangedEdges();drawGraph();});
  $('fileFilter').addEventListener('input',drawGraph);$('clearFilter').onclick=()=>{$('fileFilter').value='';drawGraph();};
  setIndex(documentHistory.commits.length-1);
 }catch(err){showText('loadStatus',`Historii nelze načíst: ${err.message}`);$('loadStatus').className='warning';}
}
init();
