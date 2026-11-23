// Atlas Go AST bridge. Parses project sources but never builds or runs them.
// Only standard-library go/parser / go/ast APIs are used. Input is a JSON
// manifest listing vetted relative paths under root.
package main

import (
 "encoding/json"
 "fmt"
 "go/ast"
 "go/parser"
 "go/token"
 "os"
 "path/filepath"
 "strings"
)

type Node struct { Kind string `json:"kind"`; Name string `json:"name"`; Qual string `json:"qual"`; Parent string `json:"parent"`; Line int `json:"line"`; Column int `json:"column"`; EndLine int `json:"end_line"` }
type Call struct { Owner string `json:"owner"`; Name string `json:"name"`; Receiver string `json:"receiver"`; Line int `json:"line"`; Column int `json:"column"`; Shadow bool `json:"shadow"` }
type Import struct { Path string `json:"path"`; Alias string `json:"alias"`; Line int `json:"line"` }
type File struct { TypedCalls []TypedCall `json:"typed_calls,omitempty"`; TypeChecked bool `json:"type_checked"`; Path string `json:"path"`; Package string `json:"package"`; Nodes []Node `json:"nodes"`; Calls []Call `json:"calls"`; Imports []Import `json:"imports"`; ParseError string `json:"parse_error,omitempty"` }
type Manifest struct { Root string `json:"root"`; Files []string `json:"files"`; Module string `json:"module"` }

func position(fs *token.FileSet, n ast.Node) (int,int,int) { a:=fs.Position(n.Pos()); b:=fs.Position(n.End()); return a.Line,a.Column,b.Line }
func recvName(expr ast.Expr) string { switch x:=expr.(type) {case *ast.Ident:return x.Name; case *ast.StarExpr:return recvName(x.X);case *ast.IndexExpr:return recvName(x.X);case *ast.IndexListExpr:return recvName(x.X);case *ast.SelectorExpr:return x.Sel.Name;default:return ""} }
func addParam(fs *token.FileSet, dest *[]Node, f *ast.FieldList, owner string) {
 if f==nil{return};for _,field:=range f.List{for _,name:=range field.Names{
  l,c,e:=position(fs,name);*dest=append(*dest,Node{Kind:"parameter",Name:name.Name,Qual:owner+".param."+name.Name,Parent:owner,Line:l,Column:c,EndLine:e})
 }}
}
func shadowNames(body *ast.BlockStmt, par *ast.FieldList) map[string]bool {
 out:=map[string]bool{};if par!=nil {for _,f:=range par.List{for _,n:=range f.Names{out[n.Name]=true}}}
 if body!=nil {ast.Inspect(body,func(node ast.Node)bool{
  switch n:=node.(type){
   case *ast.FuncLit: return false // different lexical scope
   case *ast.AssignStmt: if n.Tok==token.DEFINE {for _,lhs:=range n.Lhs{if id,ok:=lhs.(*ast.Ident);ok{out[id.Name]=true}}}
   case *ast.RangeStmt: if n.Tok==token.DEFINE {for _,expr:=range []ast.Expr{n.Key,n.Value}{if id,ok:=expr.(*ast.Ident);ok{out[id.Name]=true}}}
   case *ast.ValueSpec:for _,name:=range n.Names{out[name.Name]=true}
  };return true
 })};return out
}
func parseOne(root, rel string) File {
 result:=File{Path:rel,Nodes:[]Node{},Calls:[]Call{},Imports:[]Import{}}
 clean:=filepath.Clean(rel);abs:=filepath.Join(root,clean)
 if filepath.IsAbs(clean)||clean==".."||strings.HasPrefix(clean,".."+string(filepath.Separator)) {result.ParseError="unsafe path";return result}
 fs:=token.NewFileSet(); tree,err:=parser.ParseFile(fs,abs,nil,parser.AllErrors|parser.ParseComments)
 if err!=nil {result.ParseError=err.Error();return result}
 result.Package=tree.Name.Name
 for _,spec:=range tree.Imports {imp:=Import{Line:fs.Position(spec.Pos()).Line};var s string
  if e:=json.Unmarshal([]byte(spec.Path.Value),&s);e!=nil{continue};imp.Path=s
  if spec.Name!=nil {imp.Alias=spec.Name.Name};result.Imports=append(result.Imports,imp)
 }
 for _,decl:=range tree.Decls{
  switch x:=decl.(type){
  case *ast.GenDecl:
   for _,sp:=range x.Specs{switch y:=sp.(type){
    case *ast.TypeSpec:
     k:="type";switch y.Type.(type){case *ast.StructType:k="struct";case *ast.InterfaceType:k="interface"}
     l,c,e:=position(fs,y.Name);result.Nodes=append(result.Nodes,Node{Kind:k,Name:y.Name.Name,Qual:y.Name.Name,Line:l,Column:c,EndLine:e})
    case *ast.ValueSpec:
     kind:="variable";if x.Tok==token.CONST{kind="constant"};for _,id:=range y.Names{
      l,c,e:=position(fs,id);result.Nodes=append(result.Nodes,Node{Kind:kind,Name:id.Name,Qual:id.Name,Line:l,Column:c,EndLine:e})
     }
   }}
  case *ast.FuncDecl:
   parent:="";kind:="function";if x.Recv!=nil&&len(x.Recv.List)>0 {parent=recvName(x.Recv.List[0].Type);kind="method"}
   name:=x.Name.Name; qual:=name;if parent!="" {qual=parent+"."+name}
   l,c,e:=position(fs,x);result.Nodes=append(result.Nodes,Node{Kind:kind,Name:name,Qual:qual,Parent:parent,Line:l,Column:c,EndLine:e});addParam(fs,&result.Nodes,x.Type.Params,qual)
   shadows:=shadowNames(x.Body,x.Type.Params)
   if x.Body!=nil {ast.Inspect(x.Body,func(n ast.Node)bool{
    if _,ok:=n.(*ast.FuncLit);ok{return false}
    call,ok:=n.(*ast.CallExpr);if !ok{return true}
    r:=Call{Owner:qual,Line:fs.Position(call.Pos()).Line,Column:fs.Position(call.Pos()).Column}
    switch f:=call.Fun.(type){case *ast.Ident:r.Name=f.Name;r.Shadow=shadows[f.Name]
     case *ast.SelectorExpr:r.Name=f.Sel.Name;if id,ok:=f.X.(*ast.Ident);ok{r.Receiver=id.Name;r.Shadow=shadows[id.Name]}
    }
    if r.Name!="" {result.Calls=append(result.Calls,r)};return true
   })}
  }
 }
 return result
}
func main(){if len(os.Args)!=2{fmt.Fprintln(os.Stderr,"usage: AtlasGoAstBridge <manifest.json>");os.Exit(2)}
 bytes,err:=os.ReadFile(os.Args[1]);if err!=nil{panic(err)};var m Manifest;if err=json.Unmarshal(bytes,&m);err!=nil{panic(err)}
 out:=[]File{};for _,rel:=range m.Files{out=append(out,parseOne(m.Root,rel))}
 typed,checked:=typecheckSources(m.Root,m.Module,out);for idx:=range out {out[idx].TypedCalls=typed[out[idx].Path];out[idx].TypeChecked=checked[out[idx].Path]}
 if err=json.NewEncoder(os.Stdout).Encode(out);err!=nil{panic(err)}
}
