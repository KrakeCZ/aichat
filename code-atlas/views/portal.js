/* Universal Code Atlas portal — file tree and architecture from recorded data.
   No tracking, no analytics and no rendering from unsanitized source HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let graph = null, index = [], fileSet = new Map(), selected = '', selectedText = '', searchPage = 0;
  let architecture = null, architectureView = "auto";
  const MAX_PER_PAGE = 65;
  const text = (tag, value, cls='') => {const x=document.createElement(tag);x.textContent=String(value);if(cls)x.className=cls;return x};
  const escaped = path => path.split('/').map(encodeURIComponent).join('/');
  function sourceUrl(path) {
    const mode = graph?.metadata?.source_mode || 'local';
    if (mode === 'bundled') return './sources/' + escaped(path);
    if (mode === 'local') return './__source__/' + escaped(path);
    const repo = graph?.repository;
    if (mode === 'github' && repo && /^[\w.-]+\/[\w.-]+$/.test(repo))
      return 'https://raw.githubusercontent.com/' + repo + '/' + (graph.source_commit || 'main') + '/' + escaped(path);
    return null;
  }
  function download(name,data,type='application/json') {
    const b=new Blob([data],{type});const url=URL.createObjectURL(b);const a=document.createElement('a');
    a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),300);
  }
  function chooseFile(path) {
    selected=path;
    $('selectedName').textContent=path;
    $('atlasFileLink').href='./atlas-v2.html?file='+encodeURIComponent(path);
    $('insideFileLink').href='./inside-file.html?file='+encodeURIComponent(path);
    $('codeState').textContent='Načítám úplný soubor…';
    $('sourcePreview').textContent='';
    $('saveSource').disabled=true;
    document.querySelectorAll('.fileBtn').forEach(b=>b.setAttribute('aria-current',String(b.dataset.path===path)));
    const url=sourceUrl(path);
    if(!url){$('codeState').textContent='Zdroj není dostupný pro tento režim publikace.';return;}
    fetch(url,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.text()}).then(s=>{
      if(selected!==path)return;
      selectedText=s;
      $('sourcePreview').textContent=s.split('\n').map((l,i)=>String(i+1).padStart(5,' ')+' │ '+l).join('\n');
      $('saveSource').disabled=false;
      $('codeState').textContent=s.split('\n').length+' řádků';
    }).catch(err=>{$('codeState').textContent='Nelze načíst: '+err.message});
  }
  function fileButton(path,label) {const b=text('button','📄 '+(label||path.split('/').at(-1)),'fileBtn');b.dataset.path=path;b.onclick=()=>chooseFile(path);return b;}
  function makeTree(paths) {
    const root={dirs:new Map(),files:[]};
    paths.forEach(path=>{let current=root;const parts=path.split('/');for(const part of parts.slice(0,-1)){
      if(!current.dirs.has(part))current.dirs.set(part,{dirs:new Map(),files:[]});current=current.dirs.get(part)
    }current.files.push(path)});
    const draw=(tree,parent)=>{
      for(const [name,branch] of [...tree.dirs].sort((a,b)=>a[0].localeCompare(b[0]))) {
        const det=document.createElement('details');const summary=text('summary','📁 '+name);det.append(summary);parent.append(det);
        det.addEventListener('toggle',()=>{if(det.open && !det.dataset.populated){const content=document.createElement('div');content.className='tree';draw(branch,content);det.append(content);det.dataset.populated='1'} });
      }
      for(const path of [...tree.files].sort())parent.append(fileButton(path));
    };
    draw(root,$('fileTree'));
  }
  function searchFiles() {
    const q=$('fileSearch').value.toLocaleLowerCase().trim();const result=$('matches');
    const matched=index.filter(x=>(x.path+' '+(x.symbols||[]).join(' ')).toLocaleLowerCase().includes(q));
    if(!q){result.replaceChildren();$('more').hidden=true;return;}
    if(searchPage===0)result.replaceChildren();
    matched.slice(searchPage*MAX_PER_PAGE,(searchPage+1)*MAX_PER_PAGE).forEach(x=>result.append(fileButton(x.path,x.path)));
    $('more').hidden=(searchPage+1)*MAX_PER_PAGE>=matched.length;
    $('more').textContent=`Další výsledky (${Math.min((searchPage+1)*MAX_PER_PAGE,matched.length)} / ${matched.length})`;
  }
  function box(value,label){const e=text('div','','stat');e.append(text('b',value),text('small',label));return e}
  function drawArchitecture() {
    const view = architectureView === 'folder' ? architecture.views?.folders :
      architectureView === 'module' ? architecture.views?.build_modules : null;
    const selected = view || architecture;
    $('architectureNote').textContent=(architecture.note||'Statický pohled podle adresářů.') +
      (architectureView === 'folder' ? ' · Pohled: adresáře.' :
      architectureView === 'module' ? ' · Pohled: adresáře s build manifesty (není-li manifest, jen fyzické oblasti).' : '');
    $('folderView').setAttribute('aria-pressed',String(architectureView==='folder'));
    $('moduleView').setAttribute('aria-pressed',String(architectureView==='module'));
    const container=$('areas');container.replaceChildren();
    for(const area of selected.areas||[]){
      const c=text('div','','area');c.append(text('h3','📦 '+area.label),text('p',area.files+' souborů · '+area.inside_edges+' interních vazeb'));
      const langs=(area.languages||[]).map(l=>l.name+': '+l.count).join(' · ');c.append(text('small',langs));
      const first=area.example_files?.[0];if(first){const b=text('button','Prozkoumat '+first.split('/').at(-1));b.onclick=()=>{chooseFile(first);$('files').scrollIntoView({behavior:'smooth'})};c.append(b)}container.append(c)
    }
    const links=$('areaLinks');links.replaceChildren();
    for(const e of (selected.connections||[]).slice(0,40)){
      const c=text('div','','link');c.append(text('strong',e.source+' → '+e.target),text('small',e.count+' statických vazeb mezi soubory'));links.append(c)
    }
    if(!selected.connections?.length)links.append(text('p','Mezi vybranými oblastmi nebyly nalezeny přímé vazby. Není to důkaz, že v projektu neexistují.'));
  }
  function renderTopology(topology, capability) {
    const stats=$('topologyStats');stats.replaceChildren();
    stats.append(box(topology.confirmed_pairs,'Doložené dvojice souborů'),
                 box(topology.candidate_pairs,'Kandidátní dvojice'),
                 box(topology.confirmed_regions.length,'Oddělené oblasti potvrzeného grafu'),
                 box(topology.confirmed_cycles.length,'Nalezené cykly'));
    const regions=$('topologyRegions');regions.replaceChildren();
    for(const region of topology.confirmed_regions.filter(r=>r.size>1).slice(0,12)){
      const c=text('div','','link');c.append(text('strong',region.size+' propojených souborů'));
      const line=text('small',region.files.slice(0,5).join(' · ') + (region.size>5?' …':''));c.append(line);regions.append(c);
    }
    if(!regions.children.length)regions.append(text('p','Zatím nebyla nalezena oblast s více potvrzeně propojenými soubory.'));
    const cycles=$('topologyCycles');cycles.replaceChildren();
    for(const group of topology.confirmed_cycles.slice(0,12)){
      cycles.append(text('div',group.files.join(' ↔ '),'link'));
    }
    if(!cycles.children.length)cycles.append(text('p','V potvrzeném souborovém grafu nebyly nalezeny cykly. Nelze z toho vyvozovat, že neexistují runtime cykly.'));
    const cross=$('crossLanguages');cross.replaceChildren();
    for(const link of topology.cross_language_edges.slice(0,20)){
      const item=text('div','','link');
      item.append(text('strong',link.source+' → '+link.target));
      item.append(text('small',link.source_language+' → '+link.target_language+' · '+link.confidence+' · '+link.types.join(', ')));
      if(link.evidence?.length){item.append(text('small','Důkaz: '+link.evidence.map(e=>e.path+':'+e.line).join(' · ')))}
      cross.append(item);
    }
    if(!cross.children.length)cross.append(text('p','Nebyla nalezena žádná doložená mez jazyková vazba. To neznamená, že neexistuje.'));
    const caps=$('capabilities');caps.replaceChildren();
    for(const lang of capability.languages.filter(x=>x.files_detected>0).sort((a,b)=>b.files_detected-a.files_detected)){
      const item=text('div','','link');item.append(text('strong',lang.language+' · '+lang.files_detected+' souborů'));
      item.append(text('small',lang.method+' · '+lang.guarantee+' · skutečné parsery: '+JSON.stringify(lang.observed_symbol_parsers||{})+' · '+lang.file_edges+' grafových hran'));caps.append(item);
    }
    if(!caps.children.length)caps.append(text('p','Žádné rozpoznané soubory.'));
  }
  function exportContext() {
    if(!selected){alert('Nejprve vyber soubor.');return;}
    const related=(graph.edges||[]).filter(e=>e.source===selected||e.target===selected).slice(0,500);
    const other=new Set(related.flatMap(e=>[e.source,e.target]));
    const exportData={schema:'code-atlas-ai-context-1',project:graph.project_name||graph.metadata?.project_name,commit:graph.source_commit,
      focus:selected,nodes:(graph.nodes||[]).filter(n=>other.has(n.id)),edges:related,
      caution:'Přehled vychází ze statické analýzy a nemusí pokrývat dynamické runtime závislosti.'};
    download('atlas-context-'+selected.split('/').at(-1)+'.json',JSON.stringify(exportData,null,2)+'\n');
  }
  async function init(){
    try{
      const [g,i,a]=await Promise.all(['./graph-index.json','./source-index.json','./architecture-summary.json'].map(async url=>{
        const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error(url+': HTTP '+r.status);return r.json();
      }));graph=g;index=i.files||[];architecture=a;fileSet=new Map(index.map(x=>[x.path,x]));
      $('project').textContent=g.project_name||g.metadata?.project_name||'Projekt';
      $('stats').append(box(index.length.toLocaleString('cs-CZ'),'Indexované soubory'),box(g.edges.length.toLocaleString('cs-CZ'),'Zaznamenané vazby'),box((a.areas||[]).length,'Oblasti projektu'));
      const byLang=new Map();index.forEach(f=>byLang.set(f.language,(byLang.get(f.language)||0)+1));
      [...byLang].sort((a,b)=>b[1]-a[1]).forEach(([l,c])=>$('languages').append(text('span',l+' · '+c,'chip')));
      const eco=$('ecosystems');eco.replaceChildren();
      if(!a.detected_ecosystems?.length)eco.append(text('p','Nebyly nalezeny podporované build manifesty. Projekt může mít vlastní architekturu nebo nepodporovaný build systém.'));
      for(const item of a.detected_ecosystems||[]) {
        const el=text('div','','link');el.append(text('strong',item.ecosystem),text('small','Zdroj: '+item.path));eco.append(el);
      }
      const [topology,capability]=await Promise.all(['./dependency-topology.json','./language-capabilities.json'].map(async url=>{
        const response=await fetch(url);if(!response.ok)throw Error('HTTP '+response.status+' '+url);return response.json();
      }));
      renderTopology(topology,capability);
      // Optional build-context overlay: do not block the rest of the encyclopedia.
      try {
        const buildResponse=await fetch('./build-context.json',{cache:'no-store'});
        if(buildResponse.ok){
          const bc=await buildResponse.json();
          const stats=$('buildStats');
          if(stats)stats.append(box((bc.links||[]).length,'Build vazby'),box((bc.variants||[]).length,'Deklarace variant'),box((bc.unresolved||[]).length,'Nevyřešené případy'));
          const area=$('buildEdges');
          if(area){
            for(const item of (bc.links||[]).slice(0,40)){
              const el=text('div','','link');
              el.append(text('strong',item.from+' → '+item.to),text('small',item.kind+' · '+item.confidence+' · '+item.from+':'+item.line+' · '+item.reason));
              area.append(el);
            }
            if(!(bc.links||[]).length)area.append(text('p','Nebyly nalezeny přímo doložené build vazby. To neznamená, že neexistují.'));
          }
        }
      }catch(_error){ /* Older generated indexes may not have this layer. */ }
      try {
        const response=await fetch('./android-variant-context.json',{cache:'no-store'});
        if(response.ok){
          const v=await response.json();
          const modules=Object.values(v.modules||{}),files=Object.values(v.files||{});
          const stats=$('variantStats'),content=$('variantInfo');
          if(stats)stats.append(box(modules.length,'Android moduly'),
             box(files.filter(x=>x.possible_variants?.length).length,'Soubory s kandidátními variantami'),
             box((v.edge_annotations||[]).length,'Vazby mezi oddělenými source sety'));
          if(content){
            for(const m of modules.slice(0,20)){
              const el=text('div','','link');
              el.append(text('strong',m.build_file),text('small',
                (m.variants||[]).map(x=>x.name).join(', ') || 'Varianty nejsou jednoznačně rozpoznané'));
              content.append(el);
            }
            for(const e of (v.edge_annotations||[]).slice(0,20)){
              const el=text('div','','link');
              el.append(text('strong',e.from+' → '+e.to),
                        text('small','Bez společné kandidátní varianty · '+e.type+' · pro potvrzení je nutný AGP build model'));
              content.append(el);
            }
            if(!modules.length)content.append(text('p','Nebyl rozpoznán Android modul s doloženým pluginem.'));
          }
        }
      }catch(_error){ /* Older reports may not include variant context. */ }

      // PHP version evidence: static metadata, not a runtime vulnerability scanner.
      try {
        const response=await fetch('./php-version-report.json',{cache:'no-store'});
        if(response.ok){
          const v=await response.json();
          const stats=$('phpVersionStats');
          const records=$('phpVersionRecords');
          if(stats)stats.append(box(v.php_files_indexed||0,'PHP soubory'),
            box((v.evidence||[]).length,'Verzované deklarace'),
            box((v.review_items||[]).length,'K revizi'));
          if(records){
            const info=text('p','Provozní PHP nebylo kontrolováno. Upozornění se týkají deklarací, ne potvrzené zranitelnosti serveru.');
            records.append(info);
            for(const item of (v.evidence||[]).slice(0,60)){
              const el=text('div','','link');
              el.append(text('strong',item.path+' · '+item.kind+' · '+item.version));
              el.append(text('small',item.message+' · podpora: '+item.branch_support+
                ' · PHP2Shell: '+item.php2shell));
              records.append(el);
            }
            if(!(v.evidence||[]).length)records.append(text('p',
                'Verze PHP není v nalezené konfiguraci doložena. Starší PHP soubory lze přesto indexovat.'));
            for(const issue of (v.review_items||[]).slice(0,20)){
              const el=text('div','','link');el.append(text('strong','K revizi: '+issue.path+' · '+issue.version));
              el.append(text('small',issue.message));records.append(el);
            }
          }
        }
      }catch(_error){ /* Prior Atlas outputs may not include a PHP report. */ }
      // Stage 26: read-only evidence about declared language/build versions.
      try {
        const response=await fetch('./toolchain-version-report.json',{cache:'no-store'});
        if(response.ok){
          const r=await response.json();
          const stats=$('toolVersionStats'), records=$('toolVersionEvidence');
          const items=Array.isArray(r.evidence)?r.evidence:[];
          if(stats)stats.append(box(items.length,'Deklarace'),
            box(Object.keys(r.summary?.languages||{}).length,'Kategorie nástrojů'),
            box((r.unresolved||[]).length,'Nejasná metadata'));
          if(records){
            records.append(text('p','Běhové verze ani dostupné kompilátory nebyly kontrolovány. Hodnoty pocházejí pouze z konfigurace projektu.'));
            for(const item of items.slice(0,100)){
              const el=text('div','','link');
              el.append(text('strong',item.language+' · '+item.version+' · '+item.kind));
              el.append(text('small',item.path+(item.line_hint?' :'+item.line_hint:'')+
                ' · '+(item.explanation_cs||item.meaning)));
              records.append(el);
            }
            if(!items.length)records.append(text('p','Žádné jednoznačné deklarace verzí v dostupných metadatech. Neznamená to, že projekt nemá požadavky.'));
          }
        }
      }catch(_error){ /* Older generated Atlas outputs may not have this report. */ }
      drawArchitecture();makeTree(index.map(f=>f.path));
      const degrees=new Map();for(const e of g.edges){degrees.set(e.source,(degrees.get(e.source)||0)+1);degrees.set(e.target,(degrees.get(e.target)||0)+1)}
      const ranking=[...index].sort((a,b)=>(degrees.get(b.path)||0)-(degrees.get(a.path)||0)).slice(0,15);
      for(const f of ranking){const b=fileButton(f.path,f.path+' · '+(degrees.get(f.path)||0)+' vztahů');$('hotlist').append(b)}
      if(index[0])chooseFile(ranking[0]?.path||index[0].path);
    }catch(err){$('project').textContent='Index nelze načíst';$('architectureNote').textContent=String(err);}
  }
  $('folderView').onclick=()=>{architectureView='folder';if(architecture)drawArchitecture()};
  $('moduleView').onclick=()=>{architectureView='module';if(architecture)drawArchitecture()};
  $('fileSearch').addEventListener('input',()=>{searchPage=0;searchFiles()});
  $('more').onclick=()=>{searchPage++;searchFiles()};
  $('saveSource').onclick=()=>{if(selected)download(selected.split('/').at(-1),selectedText,'text/plain;charset=utf-8')};
  $('exportContext').onclick=exportContext;
  init();
})();
