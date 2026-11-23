/*
 * Universal Code Atlas — project-independent TypeScript compiler AST bridge.
 * Uses vendored TypeScript (Apache-2.0); NEVER loads user code via require/import/eval.
 * Input JSON on stdin: {files:[{path, source, hash}], file_edges:[...]}
 * Output: parser result with symbol graphs. File import targets are checked against
 * the graph's independently resolved file edges; unresolved names are not linked.
 */
'use strict';
const ts = require('./vendor/typescript/typescript.js');
const crypto = require('node:crypto');
const path = require('node:path').posix;
const fs = require('node:fs');
function hash(x) { return crypto.createHash('sha256').update(x).digest('hex'); }
function ident(p, q, line, col) { return 'symbol:' + hash(p+'\0'+q+'\0'+line+'\0'+col).slice(0,24); }
function fileId(p){ return 'file:'+p; }
function lineAt(sf, n) { return sf.getLineAndCharacterOfPosition(n.getStart(sf)); }
function nameText(n){ return n && (ts.isIdentifier(n)||ts.isPrivateIdentifier(n)) ? n.text : null; }
function exported(n){ return !!n.modifiers?.some(m=>m.kind===ts.SyntaxKind.ExportKeyword); }
function isStatic(n){ return !!n.modifiers?.some(m=>m.kind===ts.SyntaxKind.StaticKeyword); }
function scriptKind(p){return /\.tsx$/i.test(p)?ts.ScriptKind.TSX:/\.jsx$/i.test(p)?ts.ScriptKind.JSX:/\.[mc]?ts$/i.test(p)?ts.ScriptKind.TS:ts.ScriptKind.JS;}
function normalized(p){return path.normalize(p).replace(/^(\.\/)+/,'');}
let input;
try {input=JSON.parse(fs.readFileSync(0,'utf8'));}
catch(e){process.stderr.write('Invalid AST input: '+e.message+'\n');process.exit(2);}
if(!input || !Array.isArray(input.files)) process.exit(2);
const all=new Map();
const edgeInfo=new Map();
for(const e of input.file_edges||[]){
  if(e.source && e.target && e.resolution==='resolved_static'){
    const line=e.line||0,key=e.source+'\0'+line;
    if(!edgeInfo.has(key))edgeInfo.set(key,new Set());
    edgeInfo.get(key).add(e.target);
  }
}
function makeFile(f){
 const sf=ts.createSourceFile(f.path,f.source,ts.ScriptTarget.Latest,true,scriptKind(f.path));
 const parsed={path:f.path,source:f.source,hash:f.hash||hash(Buffer.from(f.source)),sf,
    errorCount:sf.parseDiagnostics.length,nodes:[],edges:[],byAst:new Map(),defs:new Map(),
    exports:new Map(),children:new Map(),scopes:[],imports:new Map(),edgeSet:new Set()};
 const file={id:fileId(f.path),path:f.path,name:path.basename(f.path),kind:'file',line:1,end_line:f.source.split(/\r?\n/).length,confidence:'verified_file'};
 parsed.file=file;parsed.nodes.push(file);
 const lines=f.source.split(/\r?\n/);
 parsed.evidence=(at)=>{
    const loc=lineAt(sf,at),line=loc.line+1;
    return {path:f.path,line,end_line:sf.getLineAndCharacterOfPosition(at.getEnd()).line+1,
      column:loc.character,source_sha256:parsed.hash,snippet:(lines[line-1]||'').trim().slice(0,200)};
 };
 parsed.addEdge=(a,b,relation,at,resolution='verified_syntax')=>{
   const ev=parsed.evidence(at),key=[a.id,b.id,relation,ev.line].join('\0');
   if(parsed.edgeSet.has(key))return;
   parsed.edgeSet.add(key);
   parsed.edges.push({source:a.id,target:b.id,source_path:a.path,target_path:b.path,
     relation,resolution,analyzer:'typescript-compiler-ast',evidence:ev});
 };
 parsed.addSym=(at,owner,name,kind,{recordTop=true,exportedName=false,staticMethod=false}={})=>{
   const loc=lineAt(sf,at),q=(owner.qualname?owner.qualname+'.':'')+name;
   const n={id:ident(f.path,q,loc.line+1,loc.character),path:f.path,name,qualname:q,kind,
     line:loc.line+1,end_line:sf.getLineAndCharacterOfPosition(at.getEnd()).line+1,
     column:loc.character,parent:owner.id,confidence:'verified_syntax'};
   if(staticMethod)n.static=true;
   parsed.nodes.push(n);parsed.byAst.set(at,n);parsed.addEdge(owner,n,'declares',at);
   if(!parsed.children.has(owner.id))parsed.children.set(owner.id,new Map());
   const own=parsed.children.get(owner.id); if(!own.has(name))own.set(name,[]);own.get(name).push(n);
   if(owner.kind==='file' && recordTop){
     if(!parsed.defs.has(name))parsed.defs.set(name,[]);parsed.defs.get(name).push(n);
     if(exportedName){if(!parsed.exports.has(name))parsed.exports.set(name,[]);parsed.exports.get(name).push(n);}
   }
   return n;
 };
 parsed.scopes.push({ast:sf,owner:file,cls:null});
 all.set(f.path,parsed);
 return parsed;
}
input.files.forEach(makeFile);
function declarations(p,container,owner,cls=null){
 for(const st of container.statements||[]){
   if(ts.isFunctionDeclaration(st) && st.name){
     const n=p.addSym(st,owner,st.name.text,'function',{exportedName:exported(st)});
     p.scopes.push({ast:st,owner:n,cls});
     for(const prm of st.parameters||[]) if(nameText(prm.name))p.addSym(prm,n,prm.name.text,'parameter',{recordTop:false});
     if(st.body)declarations(p,st.body,n,cls);
   }else if(ts.isClassDeclaration(st) && st.name){
     const n=p.addSym(st,owner,st.name.text,'class',{exportedName:exported(st)});
     p.scopes.push({ast:st,owner:n,cls:n});
     for(const member of st.members){
       const mn=nameText(member.name);
       if((ts.isMethodDeclaration(member)||ts.isConstructorDeclaration(member)||ts.isGetAccessorDeclaration(member)||ts.isSetAccessorDeclaration(member))){
         const s=p.addSym(member,n,mn||'constructor','method',{recordTop:false,staticMethod:isStatic(member)});
         p.scopes.push({ast:member,owner:s,cls:n});
         for(const prm of member.parameters||[])if(nameText(prm.name))p.addSym(prm,s,prm.name.text,'parameter',{recordTop:false});
         if(member.body)declarations(p,member.body,s,n);
       }else if(ts.isPropertyDeclaration(member)&&mn){p.addSym(member,n,mn,'field',{recordTop:false});}
     }
   }else if(ts.isInterfaceDeclaration(st) && st.name){
     p.addSym(st,owner,st.name.text,'interface',{exportedName:exported(st)});
   }else if(ts.isEnumDeclaration(st) && st.name){
     p.addSym(st,owner,st.name.text,'enum',{exportedName:exported(st)});
   }else if(ts.isTypeAliasDeclaration(st)&&st.name){
     p.addSym(st,owner,st.name.text,'type',{exportedName:exported(st)});
   }else if(ts.isVariableStatement(st)){
     for(const decl of st.declarationList.declarations){
       const name=nameText(decl.name);
       if(name){const n=p.addSym(decl,owner,name,owner.kind==='file'?'variable':'local_variable', {exportedName:exported(st)});
         if(decl.initializer)p.addEdge(owner,n,'writes',decl,'verified_syntax');}
     }
   }else if(ts.isBlock(st)){declarations(p,st,owner,cls);}
 }
}
for(const p of all.values())declarations(p,p.sf,p.file);
function unique(xs){return xs&&xs.length===1?xs[0]:null;}
function lookupChildren(p,owner,name){return unique(p.children.get(owner.id)?.get(name));}
function hasShadow(p,scope,name){
 if(scope.owner.kind==='file')return false;
 const defs=p.children.get(scope.owner.id)?.get(name)||[];
 if(defs.some(x=>['parameter','variable','local_variable','field'].includes(x.kind)))return true;
 let shadow=false;
 function walk(n){
   if(n!==scope.ast && (ts.isFunctionLike(n)||ts.isClassDeclaration(n)||ts.isArrowFunction(n)||ts.isFunctionExpression(n)))return;
   if(ts.isBinaryExpression(n)&&n.operatorToken.kind===ts.SyntaxKind.EqualsToken && ts.isIdentifier(n.left)&&n.left.text===name)shadow=true;
   ts.forEachChild(n,walk);
 }
 walk(scope.ast);
 return shadow;
}
function fileFromImport(p,st){
 const spec=st.moduleSpecifier && ts.isStringLiteralLike(st.moduleSpecifier)?st.moduleSpecifier.text:null;
 if(!spec)return null;
 const loc=lineAt(p.sf,st),line=loc.line+1;
 // A file-level import edge has a line, NOT the identity of one import
 // statement. Several imports on the same line must not borrow the target
 // of the first import. Relative paths can be independently resolved below.
 const fromEdges=Array.from(edgeInfo.get(p.path+'\0'+line)||[]).filter(q=>all.has(q));
 if(!spec.startsWith('.')){
   const sameLine=p.sf.statements.filter(stmt=>
     (ts.isImportDeclaration(stmt)||ts.isExportDeclaration(stmt)) &&
     stmt.moduleSpecifier && lineAt(p.sf,stmt).line+1===line);
   return sameLine.length===1 && sameLine[0]===st && fromEdges.length===1
     ? fromEdges[0] : null;
 }
 const base=normalized(path.join(path.dirname(p.path),spec));
 if(base==='..'||base.startsWith('../')||base.startsWith('/'))return null;
 const typedExt=/\.(?:[cm]?ts|tsx)$/i.test(base);
 const javascriptExt=/\.(?:[cm]?js|jsx)$/i.test(base);
 const stem=javascriptExt?base.replace(/\.(?:[cm]?js|jsx)$/i,''):base;
 const opts=typedExt?[base]:javascriptExt?
   [stem+'.ts',stem+'.tsx',stem+'.mts',stem+'.cts',base]:[
   base+'.ts',base+'.tsx',base+'.mts',base+'.cts',base+'.js',base+'.jsx',base+'.mjs',base+'.cjs',
   base+'/index.ts',base+'/index.tsx',base+'/index.js',base+'/index.jsx'];
 const matches=opts.filter(q=>all.has(q));
 // Do not use a mere line-level import edge when the source-relative
 // resolution is absent or ambiguous; it can describe a different import.
 return matches.length===1?matches[0]:null;
}
function collectImports(p){
 for(const st of p.sf.statements){
   if(!ts.isImportDeclaration(st)||!st.importClause||st.importClause.isTypeOnly)continue;
   const targetPath=fileFromImport(p,st);if(!targetPath)continue;
   const bindings=st.importClause.namedBindings;
   if(bindings&&ts.isNamedImports(bindings)){
     for(const b of bindings.elements){
       if(b.isTypeOnly)continue;
       p.imports.set(b.name.text,{path:targetPath,original:b.propertyName?.text||b.name.text});
     }
   }else if(bindings&&ts.isNamespaceImport(bindings)){
     p.imports.set(bindings.name.text,{path:targetPath,namespace:true});
   }
   if(st.importClause.name)p.imports.set(st.importClause.name.text,{path:targetPath,original:'default'});
 }
}
for(const p of all.values())collectImports(p);
// A declaration later overwritten by an assignment is not a stable callee.
// Conservatively regard writes from any nested scope as possible rebinding.
for(const p of all.values()){
 p.rebound=new Set();
 const names=new Set(p.defs.keys());
 function scan(n){
  if(ts.isBinaryExpression(n) && ts.isIdentifier(n.left) && names.has(n.left.text) &&
     [ts.SyntaxKind.EqualsToken,ts.SyntaxKind.PlusEqualsToken,ts.SyntaxKind.MinusEqualsToken].includes(n.operatorToken.kind))
       p.rebound.add(n.left.text);
  ts.forEachChild(n,scan);
 }
 scan(p.sf);
}
function fromImport(p,name){
 const info=p.imports.get(name); if(!info)return null;
 const target=all.get(info.path);if(!target||target.errorCount)return null;
 if(info.namespace)return {namespace:target};
 return {node:unique(target.exports.get(info.original))};
}
function ownOrImport(p,scope,name){
 if(hasShadow(p,scope,name))return null;
 const local=lookupChildren(p,scope.owner,name);
 if(local)return local;
 const top=p.rebound.has(name)?null:unique(p.defs.get(name));
 if(top&&['function','class'].includes(top.kind))return top;
 return fromImport(p,name)?.node||null;
}
function resolveCall(p,scope,expression){
 if(ts.isIdentifier(expression))return {target:ownOrImport(p,scope,expression.text),isReceiver:false};
 if(ts.isPropertyAccessExpression(expression)){
   const expr=expression.expression;
   if(expr.kind===ts.SyntaxKind.ThisKeyword && scope.cls){
     const target=lookupChildren(p,scope.cls,expression.name.text);
     return {target:target&&target.kind==='method'?target:null,isReceiver:true};
   }
   if(ts.isIdentifier(expr) && !hasShadow(p,scope,expr.text)){
     const imp=fromImport(p,expr.text);
     if(imp?.namespace)return {target:unique(imp.namespace.exports.get(expression.name.text)),isReceiver:false};
   }
 }
 return {target:null,isReceiver:false};
}
function statementsWithoutNested(n,callback){
 function visit(child){
  if(child!==n&&(ts.isFunctionLike(child)||ts.isClassDeclaration(child)||ts.isArrowFunction(child)||ts.isFunctionExpression(child)))return;
  callback(child);ts.forEachChild(child,visit);
 }
 ts.forEachChild(n,visit);
}
function traverseCalls(p){
 for(const scope of p.scopes){
   // Entire class body is scanned via each method scope, rather than attributing all to the class.
   if(scope.owner.kind==='class')continue;
   statementsWithoutNested(scope.ast,n=>{
     if(ts.isCallExpression(n)||ts.isNewExpression(n)){
       const {target,isReceiver}=resolveCall(p,scope,n.expression);
       if(!target||!['function','class','method'].includes(target.kind))return;
       if(ts.isNewExpression(n)&&target.kind!=='class')return;
       const relation=ts.isNewExpression(n)?'instantiates':'calls';
       p.addEdge(scope.owner,target,relation,n,isReceiver?'candidate_dispatch':'resolved_static');
     }
   });
 }
 // extends Class (not implements Interface at runtime); static identity only.
 for(const [at,owner] of p.byAst){
   if(owner.kind!=='class'||!ts.isClassDeclaration(at))continue;
   for(const clause of at.heritageClauses||[]){
     if(clause.token!==ts.SyntaxKind.ExtendsKeyword)continue;
     for(const t of clause.types){
       if(ts.isIdentifier(t.expression)){
         const target=ownOrImport(p,{owner:p.file,ast:p.sf},t.expression.text);
         if(target?.kind==='class')p.addEdge(owner,target,'inherits',t,'resolved_static');
       }
     }
   }
 }
}
for(const p of all.values()){
 if(!p.errorCount)traverseCalls(p);
}
const out={schema:'atlas-typescript-ast/0.1',typescript_version:ts.version,files:{}};
for(const p of all.values()){
 out.files[p.path]={nodes:p.nodes,edges:p.edges,parser:p.errorCount?'typescript_ast_recovered':
    (/\.[mc]?tsx?$/i.test(p.path)?'typescript_ast':'javascript_ast'),hash:p.hash,
    parse_errors:p.sf.parseDiagnostics.map(e=>({start:e.start,message:ts.flattenDiagnosticMessageText(e.messageText,' ').slice(0,160)})).slice(0,20)};
}
process.stdout.write(JSON.stringify(out));
