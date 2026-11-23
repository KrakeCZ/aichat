/* Atlas inline symbol scene: project-neutral, same per-file evidence graph in 2D and 3D.
   No source execution and no remote requests; files are loaded from the generated local Atlas. */
(()=>{'use strict';
let indexPromise=null, scene=null, previousView=null, restorePending=null, last3DClick={id:'',at:0};
const $=id=>document.getElementById(id);
const palette={file:'#68baff',class:'#ab9afa',interface:'#9ba3fc',object:'#9ba3fc',function:'#55d6aa',method:'#79d2ac',constructor:'#93c7fa',parameter:'#e3bb83',local_variable:'#d4a67c',variable:'#d4a67c',field:'#e9ad69'};
const label=n=>n.qualname||n.name||n.id;
const isVariable=n=>['parameter','local_variable','variable','field'].includes(n.kind);
function escapePath(path){return path.split('/').map(encodeURIComponent).join('/');}
function status(msg){$('graphStatus').textContent=msg;}
function renderButtons(){const c=$('atlasInlineControls');c.hidden=!scene;if(!scene)return;
 $('atlasInlineTitle').textContent='🧬 '+scene.path+' · '+scene.document.parser;
 $('atlasInlineCount').textContent=scene.document.caution||'';
}
function graphElements(){if(!scene)return {nodes:[],links:[],limited:false};
 const all=scene.document.nodes||[];const byId=new Map(all.map(n=>[n.id,n]));
 const query=$('atlasInlineFilter').value.trim().toLocaleLowerCase();
 const includeVars=$('atlasInlineVariables').checked;
 const accepted=all.filter(n=>(includeVars||!isVariable(n)));
 const match=accepted.filter(n=>!query||(label(n).toLocaleLowerCase().includes(query)));
 // Keep source/evidence graph intact but prevent whole-index fit-to-view becoming unreadable.
 const keep=new Set(match.map(n=>n.id));
 if(query){for(const e of scene.document.edges||[]){if(keep.has(e.source))keep.add(e.target);if(keep.has(e.target))keep.add(e.source);}}
 let nodes=accepted.filter(n=>keep.has(n.id));
 // Limit the rendered *view*, not the source index or export.
 const total=nodes.length;nodes=nodes.slice(0,180);
 const included=new Set(nodes.map(n=>n.id));const edges=(scene.document.edges||[]).filter(e=>included.has(e.source)&&included.has(e.target));
 const links=edges.map((e,i)=>({source:e.source,target:e.target,id:'sym-link-'+i,color:e.relation==='calls'?'#62ddac':e.relation==='declares'?'#6384aa':'#dba76b',symbolEdge:e}));
 const sceneNodes=nodes.map(n=>({id:n.id,name:n.name||n.qualname||'? ',kind:n.kind,path:n.path,line:n.line,
   color:palette[n.kind]||'#a3b7dd',val:n.kind==='file'?4:1.2,atlasSymbol:n}));
 return {nodes:sceneNodes,links,limited:total>180,total,unfiltered:all.length};
}
function showNode(n){if(!n)return;const box=$('proof');box.replaceChildren();const h=document.createElement('strong');h.textContent=(n.kind||'symbol')+' '+label(n);box.append(h);
 const p=document.createElement('p');p.textContent=(n.path||scene.path)+':'+(n.line||'?')+' · '+(n.confidence||'static index');box.append(p);
 const btn=document.createElement('button');btn.textContent='Otevřít zdroj / důkazy';btn.onclick=()=>{
   const target=n.path&&indexPromise? n.path:scene.path;
   location.href='./inside-file.html?file='+encodeURIComponent(target);
 };box.append(btn);}
function explainEdge(e){const box=$('proof');box.replaceChildren();const h=document.createElement('strong');h.textContent=(e.relation||'relation')+' · '+(e.resolution||'unresolved');box.append(h);
 const ev=e.evidence||{};const p=document.createElement('p');p.textContent=(ev.path||scene.path)+':'+(ev.line||'?')+' '+(ev.snippet||'');box.append(p);
 const p2=document.createElement('p');p2.textContent='Analyzátor: '+(e.analyzer||'?')+' · Cíl: '+(e.target_path||scene.path);box.append(p2);
}
function render2D(){if(!scene)return;renderButtons();const {nodes,links,limited,total,unfiltered}=graphElements();
 if(cy){cy.destroy();cy=null;}
 if(typeof cytoscape==='undefined'){status('Chybí Cytoscape.js — vnitřní index je dostupný na stránce Inside File.');return;}
 cy=cytoscape({container:$('graph'),elements:[
   ...nodes.map(n=>({data:{id:n.id,label:n.name,color:n.color,kind:n.kind}})),
   ...links.map(l=>({data:{id:l.id,source:l.source,target:l.target,color:l.color,resolution:l.symbolEdge.resolution,stroke:/candidate|heuristic|lexical/.test(l.symbolEdge.resolution||'')?'dashed':'solid'}}))],
   layout:{name:'cose',animate:false,fit:true,padding:50,nodeRepulsion:9500,idealEdgeLength:130,numIter:240},
   minZoom:0.08,maxZoom:5,wheelSensitivity:.18,
   style:[{selector:'node',style:{'background-color':'data(color)',label:'data(label)',color:'#edf6ff','font-size':'10px','text-valign':'bottom','text-margin-y':7,width:29,height:29,'text-wrap':'wrap','text-max-width':'110px'}},
          {selector:'edge',style:{'line-color':'data(color)','target-arrow-color':'data(color)','target-arrow-shape':'triangle','curve-style':'bezier',width:1.5,'line-style':'data(stroke)'}}]
 });
 const nodeById=new Map(nodes.map(n=>[n.id,n]));const linkById=new Map(links.map(l=>[l.id,l.symbolEdge]));
 cy.on('tap','node',ev=>showNode(nodeById.get(ev.target.id())?.atlasSymbol));
 cy.on('tap','edge',ev=>explainEdge(linkById.get(ev.target.id())));
 status('Inside File 2D · '+scene.path+' · '+nodes.length+'/'+total+' zobrazených symbolů · '+links.length+' vazeb'+(limited?' · další symboly jsou v úplném indexu':''));
}
function get3DData(){const g=graphElements();return {nodes:g.nodes,links:g.links,limited:g.limited};}
async function enter(path){try{
 if(!indexPromise)indexPromise=fetch('./symbol-index.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Index HTTP '+r.status);return r.json();});
 const index=await indexPromise;if(!index.files||!Object.prototype.hasOwnProperty.call(index.files,path))throw Error('Soubor není v symbolovém indexu: '+path);
 const url='./'+index.files[path].graph;const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw Error('Symbolový graf HTTP '+response.status);
 const document=await response.json();if(document.file!==path)throw Error('Neodpovídající graf souboru');
 if(!scene){
   previousView={file:typeof chosen==='string'?chosen:'',mode:window.AtlasGraphView?.mode()||'2d',
     zoom:cy&&typeof cy.zoom==='function'?cy.zoom():null,
     pan:cy&&typeof cy.pan==='function'?cy.pan():null,
     threeD:window.AtlasGraphView?.capture()||null};
 }
 window.AtlasNodeMenu?.close();
 scene={path,document};renderButtons();window.renderGraph();
 }catch(error){status('Vnitřní graf nelze otevřít: '+error.message+' · Použij odkaz Inside File.');}}
function restoreAfterProjectRender(){
 if(scene || !restorePending)return;
 const saved=restorePending;
 if(saved.mode==='3d'){
   if(window.AtlasGraphView?.mode()!=='3d')return;
   window.AtlasGraphView?.restore(saved.threeD);
   restorePending=null;
 }else if(cy && typeof cy.nodes==='function'){
   // Avoid applying the old camera to the now-discarded symbol graph.
   if(!cy.nodes().some(n=>typeof byPath!=='undefined'&&byPath.has(n.id())))return;
   if(saved.zoom!==null&&typeof cy.zoom==='function')cy.zoom(saved.zoom);
   if(saved.pan&&typeof cy.pan==='function')cy.pan(saved.pan);
   restorePending=null;
 }
}
function leave(silent=false){
 if(!scene)return;
 const saved=previousView;
 scene=null;previousView=null;renderButtons();window.AtlasNodeMenu?.close();
 if(silent){restorePending=null;return;}
 if(saved?.file && typeof chosen==='string')chosen=saved.file;
 restorePending=saved;
 window.renderGraph();
 restoreAfterProjectRender();
}
function on3DClick(id){if(!scene)return;const n=(scene.document.nodes||[]).find(x=>x.id===id);if(!n)return;
 const now=performance.now();if(last3DClick.id===id&&now-last3DClick.at<460&&n.external&&n.path){enter(n.path);last3DClick.id='';return;}
 last3DClick={id,at:now};showNode(n);}
$('atlasInlineBack').onclick=()=>leave();$('atlasInlineFilter').oninput=()=>{if(scene)window.renderGraph();};$('atlasInlineVariables').onchange=()=>{if(scene)window.renderGraph();};
window.AtlasInlineScene={active:()=>!!scene,currentPath:()=>scene?.path||'',rootNodeId:()=>scene?'file:'+scene.path:'',render2D,get3DData,on3DClick,explainEdge,enter,leave,restoreAfterProjectRender};
window.AtlasOpenInsideFile=enter;
})();
