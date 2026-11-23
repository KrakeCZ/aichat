/* Universal Code Atlas: source-only Java AST bridge, using the JDK compiler API.
 * Parses and type-checks source, with -proc:none. Never executes user code,
 * produces classes, or runs annotation processors.
 * Emits tab-separated base64-safe records consumed by java_ast.py.
 */
import com.sun.source.tree.*;
import com.sun.source.util.*;
import javax.lang.model.element.*;
import javax.lang.model.type.*;
import javax.lang.model.util.*;
import javax.tools.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.io.*;
import java.util.*;

public class AtlasJavaAstBridge {
    static String root;
    static Trees trees;
    static SourcePositions positions;
    static Set<String> allowed;
    static Types typeUtils;
    static Elements elements;
    static boolean semanticallyValid;
    static String typeDescriptor(TypeMirror t) {
        if(t==null)return null;
        switch(t.getKind()){
            case VOID:return "V"; case BOOLEAN:return "Z";
            case BYTE:return "B"; case SHORT:return "S";
            case INT:return "I"; case LONG:return "J";
            case CHAR:return "C"; case FLOAT:return "F";
            case DOUBLE:return "D";
            case ARRAY:{ String component=typeDescriptor(((ArrayType)t).getComponentType());
                         return component==null?null:"["+component; }
            case ERROR:return null;
            default:
                try {
                    TypeMirror erased=typeUtils.erasure(t);
                    if(erased.getKind()!=TypeKind.DECLARED)return null;
                    Element el=typeUtils.asElement(erased);
                    if(!(el instanceof TypeElement))return null;
                    String qualified=elements.getBinaryName((TypeElement)el).toString();
                    return "L"+qualified.replace('.','/')+";";
                } catch(Exception ex){return null;}
        }
    }
    static String methodDescriptor(ExecutableElement e){
        StringBuilder b=new StringBuilder("(");
        for(VariableElement param:e.getParameters()){
            String p=typeDescriptor(param.asType()); if(p==null)return null;
            b.append(p);
        }
        String ret=typeDescriptor(e.getReturnType()); if(ret==null)return null;
        return b.append(')').append(ret).toString();
    }
    static String enc(String x) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString((x==null?"":x).getBytes(StandardCharsets.UTF_8));
    }
    static String path(CompilationUnitTree unit) {
        if (unit == null || unit.getSourceFile() == null) return "";
        try {
            Path p = Path.of(unit.getSourceFile().toUri()).toRealPath();
            Path base = Path.of(root).toRealPath();
            if (!p.startsWith(base)) return "";
            return base.relativize(p).toString().replace('\\','/');
        } catch(Exception e) { return ""; }
    }
    static long pos(CompilationUnitTree unit, Tree node) {
        return positions.getStartPosition(unit,node);
    }
    static int line(CompilationUnitTree unit, Tree node) {
        long p=pos(unit,node);
        return p<0?0:(int)unit.getLineMap().getLineNumber(p);
    }
    static int column(CompilationUnitTree unit, Tree node) {
        long p=pos(unit,node);
        return p<0?0:(int)unit.getLineMap().getColumnNumber(p);
    }
    static int endLine(CompilationUnitTree unit,Tree node) {
        long p=positions.getEndPosition(unit,node);
        return p<0?line(unit,node):(int)unit.getLineMap().getLineNumber(p);
    }
    static void emit(String... fields) { System.out.println(String.join("\t",fields)); }
    static class Walker extends TreePathScanner<Void,Void> {
        final CompilationUnitTree unit;
        final String file;
        final boolean collectDecl;
        final boolean collectCalls;
        final Deque<String> owners = new ArrayDeque<>();
        final Deque<String> methods = new ArrayDeque<>();
        Walker(CompilationUnitTree u, boolean d, boolean c){unit=u;file=path(u);collectDecl=d;collectCalls=c;}
        String parent(){return owners.isEmpty()?"":owners.peek();}
        String qualified(String name){return parent().isEmpty()?name:parent()+"."+name;}
        void decl(Tree tree,String kind,String name,String q,String p){
            int l=line(unit,tree); if(l<1 || !collectDecl) return;
            boolean isNative=tree instanceof MethodTree && ((MethodTree)tree).getModifiers().getFlags().contains(Modifier.NATIVE);
            emit("N",enc(file),enc(kind),enc(name),enc(q),enc(p),""+l,""+column(unit,tree),""+endLine(unit,tree),enc(isNative?"true":"false"));
        }
        @Override public Void visitClass(ClassTree tree,Void z){
            String name=tree.getSimpleName().toString();
            if(name.isEmpty()) return super.visitClass(tree,z);
            String k=tree.getKind().toString().toLowerCase(Locale.ROOT);
            String q=qualified(name); decl(tree,k,name,q,parent());
            owners.push(q);
            try {return super.visitClass(tree,z);} finally {owners.pop();}
        }
        @Override public Void visitMethod(MethodTree tree,Void z){
            String name=tree.getName().toString();
            if(name.equals("<init>")) name="constructor";
            String q=qualified(name);
            decl(tree,name.equals("constructor")?"constructor":"method",name,q,parent());
            if(collectCalls && semanticallyValid &&
               tree.getModifiers().getFlags().contains(Modifier.NATIVE)){
                Element resolved=trees.getElement(getCurrentPath());
                if(resolved instanceof ExecutableElement){
                    ExecutableElement method=(ExecutableElement)resolved;
                    Element enclosing=method.getEnclosingElement();
                    if(enclosing instanceof TypeElement){
                        String owner=elements.getBinaryName((TypeElement)enclosing).toString().replace('.','/');
                        String desc=methodDescriptor(method);
                        if(desc!=null && line(unit,tree)>0){
                            emit("J",enc(file),""+line(unit,tree),""+column(unit,tree),
                                  enc(owner),enc(method.getSimpleName().toString()),enc(desc));
                        }
                    }
                }
            }
            owners.push(q); methods.push(q);
            try {return super.visitMethod(tree,z);} finally {methods.pop();owners.pop();}
        }
        @Override public Void visitVariable(VariableTree tree,Void z){
            TreePath pp=getCurrentPath().getParentPath();
            if(pp==null) return super.visitVariable(tree,z);
            Tree parentTree=pp.getLeaf();
            String kind;
            if(parentTree instanceof ClassTree)kind="field";
            else if(parentTree instanceof MethodTree)kind="parameter";
            else return super.visitVariable(tree,z);
            String name=tree.getName().toString();
            decl(tree,kind,name,qualified(name),parent());
            return super.visitVariable(tree,z);
        }
        @Override public Void visitMethodInvocation(MethodInvocationTree tree,Void z){
            if(collectCalls && !methods.isEmpty()){
                Element element=trees.getElement(getCurrentPath());
                if(element instanceof ExecutableElement){
                    TreePath target=trees.getPath(element);
                    if(target!=null && target.getLeaf() instanceof MethodTree){
                        CompilationUnitTree dest=target.getCompilationUnit();
                        String destFile=path(dest);
                        int destLine=line(dest,target.getLeaf());
                        int callLine=line(unit,tree);
                        if(!destFile.isEmpty() && allowed.contains(destFile) && destLine>0 && callLine>0){
                            Set<Modifier> mods=element.getModifiers();
                            Element enclosing=element.getEnclosingElement();
                            boolean finalClass=enclosing!=null && enclosing.getModifiers().contains(Modifier.FINAL);
                            boolean staticallyBound=mods.contains(Modifier.STATIC)||mods.contains(Modifier.PRIVATE)
                                ||mods.contains(Modifier.FINAL)||finalClass;
                            emit("C",enc(file),enc(methods.peek()),""+callLine,
                                enc(destFile),""+destLine,""+column(dest,target.getLeaf()),staticallyBound?"resolved_static":"candidate_dispatch");
                        }
                    }
                }
            }
            return super.visitMethodInvocation(tree,z);
        }
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=2)throw new IllegalArgumentException("root + manifest required");
        root=Path.of(args[0]).toRealPath().toString();
        List<String> lines=Files.readAllLines(Path.of(args[1]),StandardCharsets.UTF_8);
        allowed=new HashSet<>(); List<File> files=new ArrayList<>();
        for(String rel:lines){
            if(rel.isBlank())continue;
            Path p=Path.of(root).resolve(rel).toRealPath();
            if(p.startsWith(Path.of(root)) && Files.isRegularFile(p)){
                allowed.add(rel.replace('\\','/'));files.add(p.toFile());
            }
        }
        JavaCompiler compiler=ToolProvider.getSystemJavaCompiler();
        if(compiler==null)throw new IllegalStateException("JDK compiler unavailable");
        DiagnosticCollector<JavaFileObject> diags=new DiagnosticCollector<>();
        try(StandardJavaFileManager manager=compiler.getStandardFileManager(diags,Locale.ROOT,StandardCharsets.UTF_8)){
            Iterable<? extends JavaFileObject> sources=manager.getJavaFileObjectsFromFiles(files);
            List<String> options=Arrays.asList("-proc:none","-implicit:none","-Xlint:none");
            JavacTask task=(JavacTask)compiler.getTask(null,manager,diags,options,null,sources);
            List<CompilationUnitTree> units=new ArrayList<>();
            for(CompilationUnitTree u:task.parse())units.add(u);
            trees=Trees.instance(task);positions=trees.getSourcePositions();
            typeUtils=task.getTypes();elements=task.getElements();
            // Parse-only pass: avoid compiler-synthesized constructors in the index.
            for(CompilationUnitTree unit:units){
                if(allowed.contains(path(unit))){
                    emit("P",enc(path(unit)),enc(unit.getPackageName()==null?"":unit.getPackageName().toString()));
                    new Walker(unit,true,false).scan(unit,null);
                }
            }
            // Attribute symbols without generating class files or running processors.
            try{task.analyze();}catch(Exception e){emit("W",enc("javac_analysis_partial"),enc(e.getMessage()));}
            semanticallyValid=diags.getDiagnostics().stream().noneMatch(d -> d.getKind()==Diagnostic.Kind.ERROR);
            for(CompilationUnitTree unit:units){
                if(allowed.contains(path(unit)))new Walker(unit,false,true).scan(unit,null);
            }
            int count=0;
            for(Diagnostic<? extends JavaFileObject> d:diags.getDiagnostics()){
                if(d.getKind()==Diagnostic.Kind.ERROR && count++<100){
                    String src=d.getSource()==null?"":d.getSource().toUri().toString();
                    emit("E",enc(src),""+d.getLineNumber(),enc(d.getMessage(Locale.ROOT)));
                }
            }
        }
    }
}
