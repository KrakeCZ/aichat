/* Thin V0 add-on: dynamic edge layers + full file explorer. Does not replace
   Cytoscape renderer, 3D Force Graph extension, search or original inspector. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const state={searchLimit:100};
  const labels={import:'Importy',import_alias:'Import s aliasem',type_reference:'Reference typu',
    inheritance:'Dědičnost',interface_implementation:'Implementace rozhraní',
    script_source:'Spouštění/načtení skriptu',include:'C/C++ include',
    jni_binding:'JNI vazba',manifest_component:'Android Manifest',
    gradle_dependency:'Gradle závislost',cmake_source:'CMake zdroj',
    cmake_link:'CMake link',resource_reference:'Resource',view_binding:'ViewBinding',
    same_package:'Stejný package',call_candidate:'Kandidátní volání'};
  const mk=(tag,label)=>{const e=document.createElement(tag);e.textContent=label;return e};
  const root={dirs:new Map(),files:[]};
  let paths=[];let select=()=>{};
  function constructTree(){
    root.dirs.clear();root.files=[];
    for(const id of paths){let cur=root;const parts=id.split('/');for(const p of parts.slice(0,-1)){
      if(!cur.dirs.has(p))cur.dirs.set(p,{dirs:new Map(),files:[]});cur=cur.dirs.get(p);
    }cur.files.push(id)}
  }
  function makeButton(path){
    const b=mk('button','📄 '+path.split('/').at(-1));
    b.className='result';b.title=path;b.style.padding='4px 8px';
    b.onclick=()=>select(path);return b;
  }
  function renderDir(branch,host){
    for(const [name,sub] of [...branch.dirs].sort((a,b)=>a[0].localeCompare(b[0]))){
      const detail=document.createElement('details');
      detail.style.paddingLeft='11px';
      detail.append(mk('summary','📁 '+name));host.append(detail);
      detail.addEventListener('toggle',()=>{
        if(detail.open && !detail.dataset.ready){
          const inner=mk('div','');renderDir(sub,inner);detail.append(inner);detail.dataset.ready='1';
        }
      });
    }
    for(const path of [...branch.files].sort())host.append(makeButton(path));
  }
  function renderTree(){
    const term=$('treeSearch').value.toLowerCase().trim();
    const host=$('treeResults');host.replaceChildren();
    if(!term){renderDir(root,host);return}
    const matches=paths.filter(p=>p.toLowerCase().includes(term));
    host.append(mk('small',matches.length+' souborů'));
    for(const path of matches.slice(0,125)){
      const b=makeButton(path);b.textContent='📄 '+path;host.append(b)
    }
    if(matches.length>125)host.append(mk('small','Zobrazeno prvních 125 shod. Zpřesni filtr nebo rozbal strom bez filtru.'));
  }
  function dynamicLayers(edges){
    const counts=new Map();
    for(const edge of edges)counts.set(edge.type,(counts.get(edge.type)||0)+1);
    const host=$('dynamicEdges');host.replaceChildren();
    if(!counts.size){host.append(mk('p','Nejsou nalezené žádné vztahy.'));return}
    const saved=(()=>{try{return JSON.parse(localStorage.getItem('universal-code-atlas-prefs-v1')||'{}')}catch{return {}}})();
    for(const [type,count] of [...counts].sort((a,b)=>b[1]-a[1])){
      const row=mk('label','');const c=document.createElement('input');c.type='checkbox';c.dataset.edge=type;
      c.checked=Array.isArray(saved.edgeTypes) ? saved.edgeTypes.includes(type) : type!=='call_candidate';
      row.append(c,mk('span',(labels[type]||type.replaceAll('_',' '))+' · '+count));host.append(row);
      c.addEventListener('change',()=>{
        if(typeof handleChange==='function')handleChange();
      });
    }
  }
  function exportClosure(nodeList,edgeList){
    const focus = (typeof chosen!=='undefined' && chosen) ? chosen : '';
    if(!focus){$('closureStatus').textContent='Nejdřív vyber soubor.';return}
    const useCandidates=$('allowInferred').checked;
    const out=new Map();
    for(const e of edgeList){
      if(!useCandidates && e.confidence!=='exact')continue;
      if(!out.has(e.source))out.set(e.source,[]);
      out.get(e.source).push(e);
    }
    const files=new Set([focus]);const queue=[focus];const included=[];
    while(queue.length && files.size<=5000){
      const item=queue.shift();
      for(const edge of out.get(item)||[]){
        included.push(edge);
        if(!files.has(edge.target)){files.add(edge.target);queue.push(edge.target)}
      }
    }
    const snapshot={schema:'code-atlas-closure-1',focus,
      caveat:'Statický uzávěr zahrnuje jen zjištěné směrové vazby. Runtime závislosti, oprávnění, lifecycle a dynamický kód mohou chybět.',
      nodes:nodeList.filter(n=>files.has(n.id)),edges:included,
      limits:{max_nodes:5000,truncated:queue.length>0,include_candidates:useCandidates}};
    const name='atlas-dependencies-'+focus.split('/').at(-1)+'.json';
    const url=URL.createObjectURL(new Blob([JSON.stringify(snapshot,null,2)+'\n'],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=name;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),300);
    $('closureStatus').textContent=files.size+' dosažitelných uzlů · '+included.length+' vazeb';
  }
  state.ready=function(nodes,edges,choose){
    select=choose;paths=nodes.filter(n=>n.type==='file').map(n=>n.id).sort();
    constructTree();renderTree();dynamicLayers(edges);
    $('exportClosure').onclick=()=>exportClosure(nodes,edges);
    $('treeSearch').addEventListener('input',renderTree);
    $('graphStatus').textContent += ` · ${paths.length} souborů v Exploreru`;
  };
  window.AtlasEnhancements=state;
  document.addEventListener('atlas:data-ready',()=>{
    if(typeof data!=='undefined' && data) state.ready(nodes,edges,choose);
  });
  // Cover the race where the data fetch completed before this addon loaded.
  if(typeof data!=='undefined' && data && typeof choose==='function'){
    state.ready(nodes,edges,choose);
    if(typeof renderGraph==='function') renderGraph();
  }
})();
