/* Inside File — standalone progressive enhancement, no project-specific defaults.
   No telemetry, no source execution. Source text always uses textContent. */
(()=>{'use strict';
const $=id=>document.getElementById(id), ns='http://www.w3.org/2000/svg';
let index, graph, path='', sourceText='', listPage=0, matchesShown=false, requestSequence=0;
const TYPE_COLORS={file:'#5dbbfc',class:'#ad92ff',function:'#5ed2b9',async_function:'#5ed2b9',method:'#85d6ad',variable:'#f6c17e',local_variable:'#f6c17e',field:'#f6c17e',parameter:'#edc27c'};
const text=(tag,txt,cls)=>{const n=document.createElement(tag);n.textContent=String(txt);if(cls)n.className=cls;return n};
function svgNode(tag,attrs){const s=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>s.setAttribute(k,String(v)));return s}
function download(name,data,kind='application/json'){const blob=new Blob([data],{type:kind}),uri=URL.createObjectURL(blob),a=document.createElement('a');a.href=uri;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(uri),500)}
function status(t){$('graphStatus').textContent=t}
function pathUrl(p){return p.split('/').map(encodeURIComponent).join('/')}
function sourceUrl(p){const mode=graph?.source_mode||index?.source_mode||'local';
 if(mode==='bundled')return './sources/'+pathUrl(p);
 if(mode==='local'||mode==='git_snapshot')return './__source__/'+pathUrl(p);
 return null;}
function showEvidence(node){
 const panel=$('nodeDetail');panel.replaceChildren();panel.append(text('strong',node.kind+' · '+(node.qualname||node.name)));
 panel.append(text('div',node.path+':'+(node.line||'?')+(node.end_line&&node.end_line!==node.line?'–'+node.end_line:''),'sub'));
 panel.append(text('div','Stav: '+(node.confidence||'neuvedeno'),'sub'));
 if(node.path!==path){const b=text('button','Otevřít definici v '+node.path);b.onclick=()=>location.href='?file='+encodeURIComponent(node.path);panel.append(b)}
 const relations=(graph?.edges||[]).filter(e=>e.source===node.id||e.target===node.id);
 for(const e of relations.slice(0,32)){
  const other=(graph.nodes||[]).find(x=>x.id===(e.source===node.id?e.target:e.source));
  const box=text('div',e.relation+' → '+(other?.qualname||other?.name||e.target));
  if(e.evidence?.line){const b=text('button','Důkaz '+e.evidence.path+':'+e.evidence.line,'relationBtn');b.onclick=()=>jumpToLine(e.evidence.line);box.append(b)}
  panel.append(box);
 }
 if(node.path===path)jumpToLine(node.line);
}
function showEdgeEvidence(edge){
 const panel=$('nodeDetail');panel.replaceChildren();
 const nodeById=new Map((graph.nodes||[]).map(n=>[n.id,n]));
 const from=nodeById.get(edge.source),to=nodeById.get(edge.target);
 panel.append(text('strong','Vazba: '+edge.relation));
 panel.append(text('div',(from?.qualname||from?.name||edge.source)+' → '+(to?.qualname||to?.name||edge.target),'boldlink'));
 panel.append(text('div',(from?.path||edge.source_path||'?')+':'+(from?.line||'?')+' → '+(to?.path||edge.target_path||'?')+':'+(to?.line||'?'),'sub'));
 if(to?.path && to.path!==path && index?.files?.[to.path]){
  const b=text('button','Otevřít cílový soubor: '+to.path);
  b.onclick=()=>location.href='?file='+encodeURIComponent(to.path);
  panel.append(b);
 }
 panel.append(text('div','Důvěryhodnost: '+edge.resolution+' · '+edge.analyzer,'sub'));
 if(/candidate|lexical|heuristic/.test(edge.resolution||''))panel.append(text('div','⚠ Kandidátní vazba: statická analýza neprokazuje runtime cíl, dynamické načítání ani přepsání funkcí.','warning'));
 const ev=edge.evidence||{};
 if(ev.path){panel.append(text('div','Zdroj důkazu: '+ev.path+':'+(ev.line||'?'),'sub'));
  if(ev.snippet)panel.append(text('pre',ev.snippet));
  if(ev.path===path && ev.line){const b=text('button','Otevřít důkaz na řádku '+ev.line);b.onclick=()=>jumpToLine(ev.line);panel.append(b);jumpToLine(ev.line)}
 }
}
function jumpToLine(line){const pre=$('sourceText');if(!line||!sourceText)return;let height=parseFloat(getComputedStyle(pre).lineHeight);if(!Number.isFinite(height))height=19;pre.scrollTop=Math.max(0,(line-4)*height);$('sourceStatus').textContent='Řádek '+line+' · '+sourceText.split('\n').length+' řádků';}
function renderNodes(){const host=$('nodeList'),q=$('nodeSearch').value.toLocaleLowerCase().trim();host.replaceChildren();
 let nodes=(graph.nodes||[]).filter(x=>x.kind!=='file');if(q)nodes=nodes.filter(x=>(x.qualname||x.name).toLocaleLowerCase().includes(q));
 nodes.slice(0,(listPage+1)*90).forEach(n=>{const b=text('button',n.kind+' · '+(n.qualname||n.name)+' :L'+n.line);b.onclick=()=>showEvidence(n);host.append(b)});
 $('moreSymbols').hidden=nodes.length<=(listPage+1)*90;
}
function renderGraph(){if(!graph)return;const svg=$('graphSvg');svg.replaceChildren();const allNodes=[...(graph.nodes||[])];
 const ids=new Set(allNodes.map(n=>n.id));
 for(const e of graph.file_edges||[]) {for(const p of [e.source_path,e.target_path]) {const fid='file:'+p;if(!ids.has(fid)) {ids.add(fid);allNodes.push({id:fid,path:p,name:p.split('/').at(-1),kind:'file',line:1,external:true});}}}
 const hideVars=!$('showVariables').checked;const filter=$('relationFilter').value;
 let nodes=allNodes.filter(n=>!hideVars||!['parameter','local_variable','variable','field'].includes(n.kind));
 const edgeSet=[...(graph.edges||[]),...(graph.file_edges||[])],showEdges=edgeSet.filter(e=>filter==='all'||(filter==='file'?e.relation.startsWith('file_'):e.relation===filter));
 const special=showEdges.flatMap(e=>[e.source,e.target]);
 if(filter!=='all'){nodes=nodes.filter(n=>n.kind==='file'||special.includes(n.id))}
 // Focus is derived exclusively from names in this file's indexed symbols.
 // Retain direct neighbours to show why the selected symbol is connected.
 const focus=$('graphFocus').value.trim().toLocaleLowerCase();
 if(focus){
  const direct=new Set(nodes.filter(n=>((n.qualname||n.name||'').toLocaleLowerCase()).includes(focus)).map(n=>n.id));
  const keep=new Set(direct);
  for(const e of showEdges){if(direct.has(e.source))keep.add(e.target);if(direct.has(e.target))keep.add(e.source)}
  nodes=nodes.filter(n=>keep.has(n.id));
 }
 const total=nodes.length;if(nodes.length>140)nodes=nodes.slice(0,140);
 const selected=new Set(nodes.map(n=>n.id));const ed=showEdges.filter(e=>selected.has(e.source)&&selected.has(e.target));
 const byId=new Map(allNodes.map(x=>[x.id,x]));
 const depth=n=>{if(n.kind==='file')return 0;const parent=byId.get(n.parent);return parent?.kind==='class'?2:parent&&parent.kind!=='file'?3:1};
 const columns=[[],[],[],[]];nodes.forEach(n=>columns[Math.min(3,depth(n))].push(n));
 const height=Math.max(390,...columns.map(c=>c.length*64+80)),width=960;
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('preserveAspectRatio','xMinYMin meet');
 if($('graphScale').value==='detail'){
   // No forced fit-to-490px scaling: large graphs must remain readable.
   svg.style.width=width+'px';svg.style.height=height+'px';
 }else{svg.style.width='100%';svg.style.height='100%'}
 const defs=svgNode('defs',{}),marker=svgNode('marker',{id:'pointer',viewBox:'0 0 10 10',refX:10,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});
 marker.append(svgNode('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#88a5c6'}));defs.append(marker);svg.append(defs);
 const positions=new Map();columns.forEach((column,i)=>column.forEach((n,j)=>positions.set(n.id,{x:20+i*237,y:42+j*64})));
 ed.forEach(e=>{const a=positions.get(e.source),b=positions.get(e.target);if(!a||!b)return;
  const x1=a.x+204,y1=a.y+20,x2=b.x,y2=b.y+20;
  const same=(Math.abs(x2-x1)<10);const curve=same?`M${a.x+102} ${y1} C${a.x+220} ${y1-36},${b.x+220} ${y2+36},${b.x+102} ${y2}`:`M${x1} ${y1} C${x1+70} ${y1}, ${x2-70} ${y2}, ${x2} ${y2}`;
  const attrs={d:curve,fill:'none',stroke:e.relation==='calls'?'#62d1ae':e.relation==='inherits'?'#ffc27a':e.relation==='call_candidate'?'#e8b46b':'#617d9b','stroke-width':e.relation==='calls'?2.3:1.5,'marker-end':'url(#pointer)',opacity:.95,style:'cursor:pointer'};
  if(/candidate|lexical|heuristic/.test(e.resolution||''))attrs['stroke-dasharray']='5 5';
  const line=svgNode('path',attrs);
  const tt=svgNode('title',{});tt.textContent=e.relation+' · '+e.resolution+' · '+(e.evidence?.path||path)+':'+(e.evidence?.line||'?');line.append(tt);
  line.addEventListener('click',()=>showEdgeEvidence(e));svg.append(line);
 });
 nodes.forEach(n=>{const p=positions.get(n.id),g=svgNode('g',{class:'n',role:'button',tabindex:0,'aria-label':(n.kind+' '+(n.qualname||n.name))});
  g.append(svgNode('rect',{x:p.x,y:p.y,width:204,height:41,rx:9,fill:'#1d344d',stroke:TYPE_COLORS[n.kind]||'#628bba','stroke-width':1.5}));
  const title=svgNode('text',{x:p.x+10,y:p.y+17,fill:TYPE_COLORS[n.kind]||'#b2cae3','font-size':10});title.textContent=n.kind;g.append(title);
  const label=svgNode('text',{x:p.x+10,y:p.y+33,fill:'#edf5ff','font-size':12});let str=n.name||n.qualname||'?';label.textContent=str.length>29?str.slice(0,27)+'…':str;g.append(label);
  const tip=svgNode('title',{});tip.textContent=(n.qualname||n.name)+' · '+n.path+':'+(n.line||'?');g.append(tip);
  const select=()=>showEvidence(n);g.addEventListener('click',select);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select()}});svg.append(g);
 });
 status(`${nodes.length} / ${total} zobrazených uzlů · ${ed.length} vztahů. Kompletní index je v JSON a v seznamu symbolů.`);
}
function renderFileRelations(){const host=$('relationList');host.replaceChildren();const edges=graph.file_edges||[];
 if(!edges.length){host.append(text('p','Žádné přímé mezisouborové vazby nebyly nalezeny. To neprokazuje nezávislost.','sub'));return}
 for(const e of edges.slice(0,200)){
  const other=e.source_path===path?e.target_path:e.source_path;
  const b=text('button',e.relation+' · '+(e.source_path===path?'→ ':'← ')+other,'relationBtn');
  b.title=e.analyzer+' · '+e.resolution;
  if(index.files[other])b.onclick=()=>{location.href='?file='+encodeURIComponent(other)};
  else b.disabled=true;
  host.append(b);
 }
 if(edges.length>200)host.append(text('small',`Zobrazeno 200 / ${edges.length}, zbývající jsou v JSON.`));
}
async function loadFile(file){const request=++requestSequence;const item=index.files[file];if(!item){$('fileTitle').textContent='Soubor nenalezen v indexu';return;}
 path=file;history.replaceState(null,'','?file='+encodeURIComponent(file));$('backAtlas').href='./atlas-v2.html?file='+encodeURIComponent(file);
 const resp=await fetch('./'+item.graph,{cache:'no-store'});if(!resp.ok)throw Error('Index souboru HTTP '+resp.status);const loaded=await resp.json();if(request!==requestSequence)return;graph=loaded;graph.source_mode=index.source_mode;
 $('fileTitle').textContent=path;
 $('coverage').textContent=graph.caution+' Úroveň analýzy: '+graph.parser+'. Každý vztah má typ a důkazní řádek, nejde o runtime záznam.';
 $('stats').replaceChildren();[['Jazyk',graph.language],['Symboly',item.symbol_count],['Vnitřní vazby',item.internal_edges],['Ostatní soubory',item.file_links]].forEach(([label,val])=>{const s=text('span',label+': '+val,'chip');$('stats').append(s)});
 $('nodeSearch').value='';listPage=0;renderGraph();renderNodes();renderFileRelations();
 sourceText='';$('sourceText').textContent='Načítám zdrojový soubor…';$('sourceStatus').textContent='';
 const url=sourceUrl(path);
 if(!url){$('sourceText').textContent='Zdroj není přibalen. Pro statické GitHub Pages spusť analyzátor s --bundle-sources.';return}
 try{const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const loadedText=await r.text();if(request!==requestSequence)return;sourceText=loadedText;$('sourceText').textContent=sourceText.split('\n').map((l,i)=>String(i+1).padStart(5,' ')+' │ '+l).join('\n');$('sourceStatus').textContent=sourceText.split('\n').length+' řádků';
  if(globalThis.crypto?.subtle && item.source_sha256){
   try{const d=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(loadedText));
    const hash=[...new Uint8Array(d)].map(b=>b.toString(16).padStart(2,'0')).join('');
    if(request===requestSequence && hash!==item.source_sha256){$('sourceStatus').textContent+=' · ⚠ Obsah se od indexování mohl změnit (hash nesouhlasí).'}
   }catch(_){}
  }
 }
 catch(e){$('sourceText').textContent='Zdroj nelze načíst: '+e.message;}
}
function fileList(){const q=$('fileSearch').value.toLocaleLowerCase().trim(),match=Object.keys(index.files).filter(f=>f.toLocaleLowerCase().includes(q));const host=$('fileMatches');host.replaceChildren();
 match.slice(0,160).forEach(p=>{const b=text('button','📄 '+p);b.onclick=()=>{host.hidden=true;loadFile(p).catch(e=>status(String(e)))};host.append(b)});
 if(match.length>160)host.append(text('small',`Zobrazeno prvních 160 / ${match.length} shod. Upřesni hledání.`));host.hidden=!matchesShown;}
async function init(){try{const r=await fetch('./symbol-index.json',{cache:'no-store'});if(!r.ok)throw Error('Symbolový index HTTP '+r.status);index=await r.json();
 const wanted=new URLSearchParams(location.search).get('file');const first=wanted&&index.files[wanted]?wanted:Object.keys(index.files)[0];if(!first){$('fileTitle').textContent='Projekt neobsahuje indexované soubory.';return}
 $('viewFiles').onclick=()=>{matchesShown=!matchesShown;fileList()};$('fileSearch').oninput=()=>{matchesShown=true;fileList()};
 $('relationFilter').onchange=renderGraph;$('showVariables').onchange=renderGraph;$('graphScale').onchange=renderGraph;$('graphFocus').oninput=renderGraph;$('nodeSearch').oninput=()=>{listPage=0;renderNodes()};$('moreSymbols').onclick=()=>{listPage++;renderNodes()};
 $('saveFileGraph').onclick=()=>{if(graph)download(path.split('/').at(-1)+'.symbols.json',JSON.stringify(graph,null,2)+'\n')};
 $('saveSource').onclick=()=>{if(sourceText)download(path.split('/').at(-1),sourceText,'text/plain;charset=utf-8')};
 await loadFile(first);
 }catch(err){$('fileTitle').textContent='Index se nepodařilo načíst';status(String(err));}}
init();
})();
