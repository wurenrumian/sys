// 一个够用就好的 Markdown 渲染器。
// 之所以自己写而不是引 CDN: 这个站点要能在完全离线的机器上打开。
// 支持: 标题 / 代码块 / 行内代码 / 粗体 / 链接 / 表格 / 引用 / 列表 / 分隔线。

(function (global) {
  function esc(s) {
    return s.replace(/[&<>]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c];
    });
  }

  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }

  function splitRow(line) {
    return line.replace(/^\||\|$/g, '').split('|').map(function (c) { return c.trim(); });
  }

  function mdToHtml(md) {
    var lines = md.replace(/\r\n/g, '\n').split('\n');
    var out = [];
    var i = 0;
    var listStack = [];   // 'ul' | 'ol'

    function closeLists(toDepth) {
      while (listStack.length > toDepth) out.push('</' + listStack.pop() + '>');
    }

    while (i < lines.length) {
      var line = lines[i];

      // ---- 围栏代码块 ----
      var fence = line.match(/^```(\w*)\s*$/);
      if (fence) {
        closeLists(0);
        var buf = [];
        i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        out.push('<pre class="code"><code>' + esc(buf.join('\n')) + '</code></pre>');
        continue;
      }

      // ---- 表格 ----
      if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        closeLists(0);
        var head = splitRow(line.trim());
        i += 2;
        var rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) {
          rows.push(splitRow(lines[i].trim()));
          i++;
        }
        var t = '<div class="tablewrap"><table><thead><tr>';
        head.forEach(function (h) { t += '<th>' + inline(h) + '</th>'; });
        t += '</tr></thead><tbody>';
        rows.forEach(function (r) {
          t += '<tr>';
          r.forEach(function (c) { t += '<td>' + inline(c) + '</td>'; });
          t += '</tr>';
        });
        out.push(t + '</tbody></table></div>');
        continue;
      }

      // ---- 分隔线 ----
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        closeLists(0); out.push('<hr>'); i++; continue;
      }

      // ---- 标题 ----
      var h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        closeLists(0);
        out.push('<h' + h[1].length + '>' + inline(h[2]) + '</h' + h[1].length + '>');
        i++; continue;
      }

      // ---- 引用 ----
      if (/^\s*>\s?/.test(line)) {
        closeLists(0);
        var q = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          q.push(lines[i].replace(/^\s*>\s?/, ''));
          i++;
        }
        out.push('<blockquote>' + mdToHtml(q.join('\n')) + '</blockquote>');
        continue;
      }

      // ---- 列表 ----
      var li = line.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
      if (li) {
        var depth = Math.floor(li[1].length / 2) + 1;
        var kind = /\d/.test(li[2]) ? 'ol' : 'ul';
        while (listStack.length < depth) { out.push('<' + kind + '>'); listStack.push(kind); }
        closeLists(depth);
        out.push('<li>' + inline(li[3]) + '</li>');
        i++; continue;
      }

      // ---- 空行 / 段落 ----
      if (!line.trim()) { closeLists(0); i++; continue; }

      closeLists(0);
      var para = [];
      while (i < lines.length && lines[i].trim() &&
             !/^\s*(#{1,6}\s|>|\||```|[-*+]\s|\d+\.\s)/.test(lines[i]) &&
             !/^\s*(-{3,})\s*$/.test(lines[i])) {
        para.push(lines[i]); i++;
      }
      if (para.length) out.push('<p>' + inline(para.join(' ')) + '</p>');
      else { out.push('<p>' + inline(line) + '</p>'); i++; }
    }
    closeLists(0);
    return out.join('\n');
  }

  global.mdToHtml = mdToHtml;
})(window);
