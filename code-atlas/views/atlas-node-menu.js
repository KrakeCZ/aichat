/* Universal Code Atlas: project-neutral long-press / right-click file node menu.
   No source-code execution; loads only the already-generated local symbol graphs. */
(()=>{'use strict';
const $=id=>document.getElementById(id);
const menu=$('atlasNodeMenu'),title=$('atlasNodeMenuTitle');
if(!menu)return;
let currentPath='',lastOpened={path:'',at:0};
function close(){menu.hidden=true;currentPath='';}
function validFile(path){return typeof path==='string' && typeof byPath!=='undefined' && byPath.has(path);}
function open(path,x,y){
 if(!validFile(path))return false;
 currentPath=path;
 lastOpened={path,at:performance.now()};
 title.textContent=path;
 menu.hidden=false;
 menu.style.left='8px';menu.style.top='8px';
 const rect=menu.getBoundingClientRect();
 const nx=Number.isFinite(x)?x:Math.max(8,window.innerWidth/2-rect.width/2);
 const ny=Number.isFinite(y)?y:Math.max(8,window.innerHeight/2-rect.height/2);
 menu.style.left=Math.max(8,Math.min(nx,window.innerWidth-rect.width-8))+'px';
 menu.style.top=Math.max(8,Math.min(ny,window.innerHeight-rect.height-8))+'px';
 $('atlasNodeMenuInspect').focus({preventScroll:true});
 return true;
}
$('atlasNodeMenuInspect').onclick=()=>{
 const path=currentPath;close();
 if(path)window.AtlasInlineScene?.enter(path);
};
$('atlasNodeMenuSource').onclick=()=>{
 const path=currentPath;close();
 if(path)window.location.assign('./inside-file.html?file='+encodeURIComponent(path));
};
$('atlasNodeMenuClose').onclick=close;
document.addEventListener('pointerdown',e=>{if(!menu.hidden&&!menu.contains(e.target))close();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!menu.hidden){e.preventDefault();close();}});
// Independent DOM pointer hold: Cytoscape's taphold is not consistently emitted
// on desktop builds or when touch gestures are interpreted as panning.
// Use current rendered geometry, never project-specific file names.
const graphElement=$('graph');
let pending=null;
function hitFile(event){
 if(!graphElement || typeof cy==='undefined' || !cy || window.AtlasInlineScene?.active())return '';
 const rect=graphElement.getBoundingClientRect();
 const x=event.clientX-rect.left,y=event.clientY-rect.top;
 if(x<0||y<0||x>rect.width||y>rect.height)return '';
 let best='',distance=Infinity;
 for(const node of cy.nodes()){
   if(!validFile(node.id()))continue;
   const at=node.renderedPosition();
   const radius=Math.max(13,(typeof node.renderedWidth==='function'?node.renderedWidth():24)/2+8);
   const d=Math.hypot(x-at.x,y-at.y);
   if(d<=radius&&d<distance){best=node.id();distance=d;}
 }
 return best;
}
function cancelPending(){if(pending?.timer)clearTimeout(pending.timer);pending=null;}
if(graphElement){
 graphElement.addEventListener('pointerdown',event=>{
   if(event.button!==0)return;
   cancelPending();const path=hitFile(event);if(!path)return;
   const press={path,x:event.clientX,y:event.clientY,timer:null};pending=press;
   press.timer=setTimeout(()=>{
     if(pending===press){open(path,press.x,press.y);pending=null;}
   },650);
 });
 graphElement.addEventListener('pointermove',event=>{
   if(pending && Math.hypot(event.clientX-pending.x,event.clientY-pending.y)>12)cancelPending();
 });
 for(const name of ['pointerup','pointercancel','pointerleave'])graphElement.addEventListener(name,cancelPending);
 graphElement.addEventListener('contextmenu',event=>{
   const path=hitFile(event);if(!path)return;
   event.preventDefault();open(path,event.clientX,event.clientY);
 });
 graphElement.addEventListener('dblclick',event=>{
   const path=hitFile(event);if(path)open(path,event.clientX,event.clientY);
 });
}
window.AtlasNodeMenu={open,close,isOpen:()=>!menu.hidden,wasRecentlyOpened:p=>lastOpened.path===p&&performance.now()-lastOpened.at<800};
})();
