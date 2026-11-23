// Project-neutral Kotlin PSI parser. It parses files, never compiles or executes project code.
// The compiler PSI provides syntax only; a name-only target is never a proven call binding.
import com.intellij.openapi.util.Disposer
import com.intellij.psi.PsiErrorElement
import com.intellij.psi.util.PsiTreeUtil
import org.jetbrains.kotlin.cli.common.CLIConfigurationKeys
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.compiler.EnvironmentConfigFiles
import org.jetbrains.kotlin.cli.jvm.compiler.KotlinCoreEnvironment
import org.jetbrains.kotlin.config.CompilerConfiguration
import org.jetbrains.kotlin.lexer.KtTokens
import org.jetbrains.kotlin.psi.*
import java.io.File
import java.util.Base64

fun enc(s: String): String = Base64.getUrlEncoder().withoutPadding().encodeToString(s.toByteArray(Charsets.UTF_8))
fun main(args: Array<String>) {
  if(args.size != 2) error("Expected root and UTF-8 list of relative paths")
  val root=File(args[0]).canonicalFile
  val paths=File(args[1]).readLines(Charsets.UTF_8)
  val disposable=Disposer.newDisposable()
  try {
    val conf=CompilerConfiguration()
    conf.put(CLIConfigurationKeys.MESSAGE_COLLECTOR_KEY, MessageCollector.NONE)
    val env=KotlinCoreEnvironment.createForProduction(disposable,conf,EnvironmentConfigFiles.JVM_CONFIG_FILES)
    val factory=KtPsiFactory(env.project)
    for(path in paths) try {
      val source=File(root,path).canonicalFile
      if(!source.path.startsWith(root.path+File.separator) || !source.isFile) continue
      val content=source.readText(Charsets.UTF_8)
      val file=factory.createFile(source.name,content)
      val lines=mutableListOf(0)
      content.forEachIndexed { idx,ch->if(ch=='\n')lines.add(idx+1) }
      fun position(pos:Int):Pair<Int,Int> {
        val i=lines.binarySearch(pos).let { if(it<0)-it-2 else it }.coerceAtLeast(0)
        return (i+1) to (pos-lines[i])
      }
      fun record(vararg fields:String){println(fields.joinToString("\t"){enc(it)})}
      // KtPsiFactory recovers malformed source into a PSI tree containing
      // error elements. Recovered declarations are NOT valid Kotlin syntax.
      // Report an explicit rejection before any symbol/call records.
      val syntaxError=PsiTreeUtil.findChildOfType(file, PsiErrorElement::class.java)
      if(syntaxError!=null) {
        record("R",path,syntaxError.errorDescription)
        continue
      }
      val pkg=file.packageFqName.asString()
      record("P",path,pkg)
      file.importDirectives.forEach { imp ->
        val base=imp.importedFqName?.asString() ?: ""
        val fq=if(imp.isAllUnder && base.isNotBlank()) "$base.*" else base
        val alias=imp.aliasName ?: ""
        record("I",path,fq,alias,position(imp.textOffset).first.toString())
      }
      val stack=mutableListOf<String>()
      val stackKinds=mutableListOf<String>()
      val stackLocations=mutableListOf<Pair<Int,Int>>()
      fun parentReference():String {
        val parent=stack.lastOrNull() ?: return ""
        val pos=stackLocations.last()
        return "$parent\u001e${pos.first}\u001e${pos.second}"
      }
      file.accept(object:KtTreeVisitorVoid(){
        override fun visitClass(klass:KtClass) {
          val name=klass.name?:"<anonymous>"
          val parent=stack.lastOrNull()?:""
          val q=if(parent.isBlank())name else "$parent.$name"
          val (line,col)=position(klass.textOffset)
          val kind=if(klass.isInterface()) "interface" else if(klass.isEnum()) "enum" else "class"
          record("N",path,kind,name,q,parentReference(),line.toString(),col.toString(),position(klass.textRange.endOffset).first.toString(),"false")
          stack.add(q)
          stackKinds.add(kind)
          stackLocations.add(line to col)
          super.visitClass(klass)
          stackKinds.removeAt(stackKinds.lastIndex)
          stack.removeAt(stack.lastIndex)
          stackLocations.removeAt(stackLocations.lastIndex)
        }
        override fun visitObjectDeclaration(declaration:KtObjectDeclaration) {
          val name=declaration.name?:"<object>"
          val parent=stack.lastOrNull()?:""
          val q=if(parent.isBlank())name else "$parent.$name"
          val (line,col)=position(declaration.textOffset)
          record("N",path,"object",name,q,parentReference(),line.toString(),col.toString(),position(declaration.textRange.endOffset).first.toString(),"false")
          stack.add(q)
          stackKinds.add("object")
          stackLocations.add(line to col)
          super.visitObjectDeclaration(declaration)
          stackKinds.removeAt(stackKinds.lastIndex)
          stack.removeAt(stack.lastIndex)
          stackLocations.removeAt(stackLocations.lastIndex)
        }
        override fun visitNamedFunction(function:KtNamedFunction) {
          val name=function.name?:"<anonymous>"
          val parent=stack.lastOrNull()?:""
          val q=if(parent.isBlank())name else "$parent.$name"
          val (line,col)=position(function.textOffset)
          val isNative=function.hasModifier(KtTokens.EXTERNAL_KEYWORD)
          val kind=if(stackKinds.lastOrNull() in setOf("class", "interface", "enum", "object")) "method" else "function"
          record("N",path,kind,name,q,parentReference(),line.toString(),col.toString(),position(function.textRange.endOffset).first.toString(),isNative.toString())
          stack.add(q)
          stackKinds.add(kind)
          stackLocations.add(line to col)
          // Raw source types: only used when an independently compiled JVM
          // descriptor can verify a unique overload. Not semantic proof alone.
          record("T", path, q, line.toString(), col.toString(), function.valueParameters.map { it.typeReference?.text ?: "" }.joinToString("\u001f"))
          // S is syntactic signature evidence; it is NOT type-based overload binding.
          // The extra record preserves backward compatibility with existing T records.
          val visibility=if(function.hasModifier(KtTokens.PRIVATE_KEYWORD))"private" else
            if(function.hasModifier(KtTokens.INTERNAL_KEYWORD))"internal" else
            if(function.hasModifier(KtTokens.PROTECTED_KEYWORD))"protected" else "public"
          val parameters=function.valueParameters.joinToString("\u001e") { param ->
            listOf(param.name ?: "", param.hasDefaultValue().toString(),
                   param.hasModifier(KtTokens.VARARG_KEYWORD).toString()).joinToString("\u001f")
          }
          record("S", path, q, line.toString(), col.toString(), visibility, parameters)
          // A Kotlin extension declaration has a receiver type. Its simple
          // name alone cannot justify an unqualified ordinary-function edge:
          // resolution requires an explicit/implicit receiver and scope.
          function.receiverTypeReference?.let { receiverType ->
            record("X", path, q, line.toString(), col.toString(), receiverType.text)
          }
          function.valueParameters.forEach { param ->
            val pname=param.name?:return@forEach
            val (pl,pc)=position(param.textOffset)
            record("N",path,"parameter",pname,"$q.param:$pname","$q\u001e$line\u001e$col",pl.toString(),pc.toString(),position(param.textRange.endOffset).first.toString(),"false")
          }
          super.visitNamedFunction(function)
          stackKinds.removeAt(stackKinds.lastIndex)
          stack.removeAt(stack.lastIndex)
          stackLocations.removeAt(stackLocations.lastIndex)
        }
        override fun visitProperty(property:KtProperty) {
          val name=property.name ?: return super.visitProperty(property)
          val parent=stack.lastOrNull()?:""
          val q=if(parent.isBlank())name else "$parent.$name"
          val (line,col)=position(property.textOffset)
          val kind=if(stackKinds.lastOrNull() in setOf("function", "method")) "local_variable" else "property"
          record("N",path,kind,name,q,parentReference(),line.toString(),col.toString(),position(property.textRange.endOffset).first.toString(),"false")
          super.visitProperty(property)
        }
        override fun visitTypeAlias(typeAlias: KtTypeAlias) {
          val name=typeAlias.name ?: return super.visitTypeAlias(typeAlias)
          val parent=stack.lastOrNull() ?: ""
          val q=if(parent.isBlank())name else "$parent.$name"
          val (line,col)=position(typeAlias.textOffset)
          record("N",path,"typealias",name,q,parentReference(),line.toString(),col.toString(),position(typeAlias.textRange.endOffset).first.toString(),"false")
          super.visitTypeAlias(typeAlias)
        }
        override fun visitCallExpression(expression:KtCallExpression) {
          val name=expression.calleeExpression?.text?:""
          val owner=stack.lastOrNull()?:""
          if(name.isNotBlank()) {
            val receiver=(expression.parent as? KtDotQualifiedExpression)?.let { q-> if(q.selectorExpression==expression)q.receiverExpression.text else "" }?:""
            val arguments=expression.valueArguments + expression.lambdaArguments
            val named=arguments.mapNotNull { it.getArgumentName()?.asName?.asString() }
            val loc=stackLocations.lastOrNull()
            val ownerLocation=if(loc==null)"" else "${loc.first}\u001e${loc.second}"
            // Preserve the original six columns, extend with argument-shape and
            // exact owner location to distinguish same-name overloaded functions.
            record("C",path,owner,name,position(expression.textOffset).first.toString(),receiver,
                   arguments.size.toString(),named.joinToString("\u001f"),ownerLocation)
          }
          super.visitCallExpression(expression)
        }
      })
      // A file is verified only after its complete PSI traversal has finished.
      // A partial record stream must never be published as a complete AST.
      record("E",path)
    }catch(exc:Exception){ System.err.println("Kotlin PSI could not parse "+path+": "+exc.javaClass.simpleName+" "+(exc.message?:"")) }
  }finally{Disposer.dispose(disposable)}
}
