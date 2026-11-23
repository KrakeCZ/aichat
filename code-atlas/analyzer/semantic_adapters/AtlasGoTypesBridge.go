// Type resolution for AtlasGoAstBridge. All inputs come from the vetted local
// manifest. It does not run, build or download code from the scanned project.
// Types for incomplete packages are never labelled as statically resolved.
package main

import (
 "fmt"
 "go/ast"
 "go/importer"
 "go/parser"
 "go/token"
 "go/types"
 "path/filepath"
 "strings"
)

type TypedCall struct {
 Owner string `json:"owner"`
 Line int `json:"line"`
 Column int `json:"column"`
 TargetPath string `json:"target_path"`
 TargetLine int `json:"target_line"`
 TargetQual string `json:"target_qual"`
 TargetName string `json:"target_name"`
}

type parsedPackage struct { name, path string; files []*ast.File; rels []string; hasConstraints bool }

type localTypeImporter struct {
 packages map[string]*parsedPackage
 checked map[string]*types.Package
 active map[string]bool
 fset *token.FileSet
 fallback types.Importer
 calls map[string][]TypedCall
}
func (i *localTypeImporter) Import(path string) (*types.Package,error) {
 if pkg,ok:=i.checked[path];ok{return pkg,nil}
 unit:=i.packages[path]
 if unit==nil {return i.fallback.Import(path)}
 if i.active[path] {return nil,fmt.Errorf("local import cycle: %s",path)}
 if unit.hasConstraints {return nil,fmt.Errorf("build constraints not resolved: %s",path)}
 i.active[path]=true;defer delete(i.active,path)
 info:=&types.Info{Uses:map[*ast.Ident]types.Object{}, Selections:map[*ast.SelectorExpr]*types.Selection{}}
 failures:=0
 cfg:=&types.Config{Importer:i,Error:func(_ error){failures++}}
 pkg,err:=cfg.Check(path,i.fset,unit.files,info)
 if err!=nil||failures!=0||pkg==nil||!pkg.Complete() {return nil,fmt.Errorf("incomplete package %s",path)}
 i.checked[path]=pkg
 // Type-checking all files together means uses refer to the precise definition.
 // For any unresolved package, nothing from that package is published as typed.
 for _,file:=range unit.files {
  filename:=i.fset.Position(file.Pos()).Filename
  rel,ok:=i.relativePath(filename);if !ok{continue}
  for _,d:=range file.Decls {
   fn,ok:=d.(*ast.FuncDecl);if !ok||fn.Body==nil {continue}
   qual:=fn.Name.Name
   if fn.Recv!=nil&&len(fn.Recv.List)>0 {if r:=recvName(fn.Recv.List[0].Type);r!="" {qual=r+"."+qual}}
   ast.Inspect(fn.Body,func(n ast.Node)bool{
    if _,ok:=n.(*ast.FuncLit);ok{return false} // separate scope
    call,ok:=n.(*ast.CallExpr);if !ok{return true}
    expr:=call.Fun
    // Generic instantiation is still a direct invocation of the selected symbol.
    for {switch x:=expr.(type){case *ast.IndexExpr:expr=x.X;case *ast.IndexListExpr:expr=x.X;default:goto unwrapped}}
   unwrapped:
    var obj types.Object
    switch x:=expr.(type){
    case *ast.Ident:obj=info.Uses[x]
    case *ast.SelectorExpr:
     obj=info.Uses[x.Sel]
     if sel:=info.Selections[x];sel!=nil&&isDynamicSelection(sel){return true}
    }
    fun,ok:=obj.(*types.Func);if !ok||fun.Pos()==token.NoPos{return true}
    target:=i.fset.Position(fun.Pos());targetRel,ok:=i.relativePath(target.Filename)
    if !ok{return true}
    caller:=i.fset.Position(call.Pos())
    targetQual:=fun.Name()
    if sign,ok:=fun.Type().(*types.Signature);ok&&sign.Recv()!=nil {
     if r:=receiverTypeName(sign.Recv().Type());r!=""{targetQual=r+"."+fun.Name()}
    }
    i.calls[rel]=append(i.calls[rel],TypedCall{Owner:qual,Line:caller.Line,Column:caller.Column,TargetPath:targetRel,TargetLine:target.Line,TargetQual:targetQual,TargetName:fun.Name()})
    return true
   })
  }
 }
 return pkg,nil
}
func receiverTypeName(t types.Type) string {
 switch x:=t.(type){case *types.Named:return x.Obj().Name();case *types.Pointer:return receiverTypeName(x.Elem());case *types.Alias:return receiverTypeName(types.Unalias(x));default:return ""}
}
func isDynamicSelection(s *types.Selection) bool {
 recv:=s.Recv()
 if _,ok:=types.Unalias(recv).Underlying().(*types.Interface);ok{return true}
 // Type parameters have interface constraints and can dispatch dynamically.
 if _,ok:=recv.(*types.TypeParam);ok{return true}
 return false
}
func (i *localTypeImporter) relativePath(path string)(string,bool){
 clean:=filepath.Clean(path)
 rel,err:=filepath.Rel(iRoot,clean)
 if err!=nil||rel==".."||strings.HasPrefix(rel,".."+string(filepath.Separator))||filepath.IsAbs(rel){return "",false}
 rel=filepath.ToSlash(rel)
 return rel,iVetted[rel]
}
var iRoot string
var iVetted map[string]bool

func typecheckSources(root, module string, recs []File) (map[string][]TypedCall,map[string]bool) {
 out:=map[string][]TypedCall{}
 verified:=map[string]bool{}
 if module==""{return out,verified}
 fset:=token.NewFileSet()
 vetted:=map[string]bool{}
 packages:=map[string]*parsedPackage{}
 // A package is identified by its module import path + source directory.
 for _,rec:=range recs {
  if rec.ParseError!=""{continue}
  rel:=filepath.ToSlash(filepath.Clean(rec.Path))
  vetted[rel]=true
 }
 for _,rec:=range recs {
  rel:=filepath.ToSlash(filepath.Clean(rec.Path))
  if !vetted[rel]{continue}
  abs:=filepath.Join(root,filepath.FromSlash(rel))
  f,err:=parser.ParseFile(fset,abs,nil,parser.AllErrors|parser.ParseComments)
  if err!=nil{continue}
  dir:=filepath.ToSlash(filepath.Dir(rel));if dir=="."{dir=""}
  pkgpath:=module;if dir!=""{pkgpath+="/"+dir}
  key:=pkgpath+"\x00"+f.Name.Name
  u:=packages[key];if u==nil{u=&parsedPackage{name:f.Name.Name,path:pkgpath};packages[key]=u}
  u.files=append(u.files,f);u.rels=append(u.rels,rel)
  // Different platform build variants may parse but may not type-check together.
  // Refuse to issue verified edges without evaluating build constraints.
  for _,group:=range f.Comments {if group.Pos()>f.Package {continue};for _,c:=range group.List {
   t:=strings.TrimSpace(c.Text)
   if strings.HasPrefix(t,"//go:build")||strings.HasPrefix(t,"// +build"){u.hasConstraints=true}
  }}
 }
 // Only one package with a given import path may be checked. In particular,
 // *_test packages may coexist with the primary package in one directory.
 byPath:=map[string]*parsedPackage{};ambiguous:=map[string]bool{}
 for _,u:=range packages {if _,ok:=byPath[u.path];ok{ambiguous[u.path]=true};byPath[u.path]=u}
 for path:=range ambiguous{delete(byPath,path)}
 iRoot=root;iVetted=vetted
 importer:=&localTypeImporter{packages:byPath,checked:map[string]*types.Package{},active:map[string]bool{},fset:fset,fallback:importer.Default(),calls:out}
 for path:=range byPath {_,_=importer.Import(path)}
 for path:=range importer.checked { for _,rel:=range byPath[path].rels {verified[rel]=true} }
 return out,verified
}
