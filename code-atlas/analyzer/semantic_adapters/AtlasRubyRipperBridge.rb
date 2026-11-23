# frozen_string_literal: true
# Atlas Ruby AST bridge. Uses only Ruby's Ripper parser, never loads/evals target files.
require 'json'
require 'ripper'

module AtlasRubyAst
  module_function
  def ident(node)
    return nil unless node.is_a?(Array)
    return node if node[0].to_s.start_with?('@') && node[2].is_a?(Array)
    node.each do |child|
      found = ident(child)
      return found if found
    end
    nil
  end
  def text(node)
    return nil unless node.is_a?(Array)
    if node[0] == :string_literal
      return nil if interpolated?(node)
      values=[]
      collect_text(node, values)
      return values.join unless values.empty?
    end
    nil
  end
  def interpolated?(node)
    return false unless node.is_a?(Array)
    return true if node[0] == :string_embexpr || node[0] == :string_dvar
    node.any? { |child| interpolated?(child) }
  end
  def collect_text(node, values)
    return unless node.is_a?(Array)
    # Interpolation is not safely resolvable as a static require.
    return if node[0] == :string_embexpr || node[0] == :string_dvar
    if node[0] == :@tstring_content
      values << node[1]
    else
      node.each { |child| collect_text(child,values) }
    end
  end
  def span(tok)
    return [0,0] unless tok && tok[2].is_a?(Array)
    [tok[2][0],tok[2][1]+1]
  end
  def walk(node, out, scope='', owner='')
    return unless node.is_a?(Array)
    tag=node[0]
    case tag
    when :module, :class
      tok=ident(node[1]);return unless tok
      name=tok[1]; qual=scope.empty? ? name : scope+'::'+name
      line,col=span(tok)
      out[:nodes] << {kind:tag.to_s, name:name, qual:qual, parent:scope, line:line, column:col}
      body= tag==:module ? node[2] : node[3]
      walk(body,out,qual,owner)
      return
    when :def, :defs
      tok=tag==:def ? node[1] : node[3]
      return unless tok.is_a?(Array) && tok[0]==:@ident
      name=tok[1];qual=scope.empty? ? name : scope+'#'+name
      line,col=span(tok)
      out[:nodes] << {kind:scope.empty? ? 'function' : 'method',name:name,qual:qual,parent:scope,line:line,column:col}
      params=tag==:def ? node[2] : node[4]
      collect_params(params, out,qual)
      body=tag==:def ? node[3] : node[5]
      walk(body,out,scope,qual)
      return
    when :vcall,:fcall,:command,:command_call,:call
      tok=case tag
        when :vcall,:fcall,:command then node[1]
        when :command_call,:call then node[3]
      end
      if tok.is_a?(Array) && [:@ident,:@const,:@op].include?(tok[0])
        line,col=span(tok)
        out[:calls]<<{owner:owner,name:tok[1],line:line,column:col,receiver: tag==:command_call || tag==:call}
        if tag==:command && tok[1]=='require_relative'
          value=text_argument(node[2]);out[:requires]<<{value:value,line:line} if value
        end
      end
    end
    node.each { |child| walk(child,out,scope,owner) if child.is_a?(Array) }
  end
  def collect_params(params,out,owner)
    return unless params.is_a?(Array)
    if params[0]==:params
      # First 4 forms can represent required, optional, rest, post args;
      # recursively collect only parameter tokens, not default expressions.
      params.drop(1).each { |x| collect_params(x,out,owner) }
    elsif params[0].to_s.start_with?('@') && params[2].is_a?(Array)
      if params[0] == :@ident
        line,col=span(params)
        out[:nodes]<<{kind:'parameter',name:params[1],qual:owner+'.param.'+params[1],parent:owner,line:line,column:col}
      end
    else
      params.each { |x| collect_params(x,out,owner) if x.is_a?(Array) }
    end
  end
  def text_argument(node)
    return nil unless node.is_a?(Array)
    return text(node) if node[0]==:string_literal
    node.each do |child|
      v=text_argument(child)
      return v if v
    end
    nil
  end
end

manifest=JSON.parse(File.read(ARGV.fetch(0)))
root=manifest.fetch('root')
results=[]
manifest.fetch('files').each do |rel|
  record={path:rel,nodes:[],calls:[],requires:[]}
  begin
    source=File.read(File.join(root,rel),encoding:'UTF-8')
    tree=Ripper.sexp(source)
    if tree.nil?
      record[:parse_error]='Ripper returned nil (syntax error)'
    else
      AtlasRubyAst.walk(tree,record)
      record[:calls].uniq!{|c|[c[:owner],c[:name],c[:line],c[:column]]}
    end
  rescue StandardError=>e
    record[:parse_error]="#{e.class}: #{e.message[0,120]}"
  end
  results<<record
end
STDOUT.write(JSON.generate(results))
