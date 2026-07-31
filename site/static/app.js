// 站点主逻辑。
// 两种运行形态:
//   1) 由 site/serve.py 提供服务 —— 可以调参、真跑 demo、流式看输出;
//   2) 由 site/export.py 导出的单文件快照 —— window.__SNAPSHOT__ 里带着预先跑好的输出。
// 除了"能不能真跑", 两种形态的阅读体验完全一样。

var SNAP = window.__SNAPSHOT__ || null;
var CAT = null;
var RUNS = {};        // demo 文件路径 -> { text, params, status, checks }

/* ------------------------------------------------------------------ 取数据 */
function getJSON(url) {
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error(r.status + ' ' + url);
    return r.json();
  });
}
function getText(url) {
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error(r.status + ' ' + url);
    return r.text();
  });
}
function loadDoc(path) {
  if (SNAP) return Promise.resolve(SNAP.docs[path] || '（快照里没有这份文档）');
  return getText('/api/doc?path=' + encodeURIComponent(path));
}
function loadSource(path) {
  if (SNAP) return Promise.resolve(SNAP.sources[path] || '');
  return getText('/api/source?path=' + encodeURIComponent(path));
}

function allDemos() {
  var out = [];
  CAT.modules.forEach(function (m) {
    (m.demos || []).forEach(function (d) { out.push(d); });
  });
  return out;
}
function findDemo(file) {
  return allDemos().filter(function (d) { return d.file === file; })[0];
}
function moduleOf(file) {
  return CAT.modules.filter(function (m) {
    return (m.demos || []).some(function (d) { return d.file === file; });
  })[0];
}

/* ------------------------------------------------------------------ 验证点 */
function evalCheck(text, c) {
  var re;
  try { re = new RegExp(c.re, 'm'); } catch (e) { return { s: 'miss', d: '正则有误: ' + e.message }; }
  var m = text.match(re);
  if (!m) return { s: 'miss', d: '输出里没有匹配到对应的行（改过参数？或者格式变了）' };
  if (!c.cmp) return { s: 'pass', d: '命中: ' + m[0].trim().slice(0, 90) };
  var v = parseFloat(m[1]);
  if (c.cmp === 'gt_group2') {
    var v2 = parseFloat(m[2]);
    return { s: v > v2 ? 'pass' : 'fail', d: '实测 ' + v + ' vs ' + v2 };
  }
  var ok = { '<': v < c.value, '>': v > c.value, '<=': v <= c.value, '>=': v >= c.value }[c.cmp];
  return { s: ok ? 'pass' : 'fail', d: '实测 ' + v + '，要求 ' + c.cmp + ' ' + c.value };
}

function renderChecks(demo, run) {
  var checks = demo.checks || [];
  if (!checks.length) return '<p class="hint">这个 demo 没有登记验证点。</p>';
  if (!run || !run.text) {
    return '<ul class="checks">' + checks.map(function (c) {
      return '<li><span class="badge">未跑</span><span class="txt">' + esc(c.text) + '</span></li>';
    }).join('') + '</ul><p class="hint">点上面的「运行」，跑完会逐条自动比对。</p>';
  }
  var res = checks.map(function (c) { return evalCheck(run.text, c); });
  var pass = res.filter(function (r) { return r.s === 'pass'; }).length;
  var html = '<ul class="checks">' + checks.map(function (c, i) {
    var r = res[i];
    var label = { pass: '通过', fail: '不符', miss: '未匹配' }[r.s];
    return '<li class="' + r.s + '"><span class="badge">' + label + '</span>' +
      '<span><span class="txt">' + esc(c.text) + '</span>' +
      '<div class="detail">' + esc(r.d) + '</div></span></li>';
  }).join('') + '</ul>';
  html += '<div class="summary">' + pass + ' / ' + checks.length + ' 条通过。';
  if (pass < checks.length) {
    html += ' 不符或未匹配 <b>不一定是坏事</b>：如果你改过参数，说明这个结论有边界条件 —— 那正是值得记下来的东西。';
  }
  html += '</div>';
  return html;
}

/* ------------------------------------------------------------------ 小工具 */
function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}
function el(html) {
  var d = document.createElement('div');
  d.innerHTML = html;
  return d.firstElementChild;
}
function download(name, text) {
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
  a.download = name;
  a.click();
  setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
}

/* ------------------------------------------------------------------ 侧栏 */
function renderNav() {
  var hash = location.hash || '';
  var html = CAT.modules.map(function (m) {
    var act = hash === '#/m/' + m.id ? ' active' : '';
    var s = '<div class="navmod"><a href="#/m/' + m.id + '" class="' + act.trim() + '">' +
      esc(m.title) + '</a>';
    (m.demos || []).forEach(function (d) {
      var a2 = hash === '#/d/' + d.file ? ' active' : '';
      s += '<a class="navdemo' + a2 + '" href="#/d/' + d.file + '">' + esc(d.title) + '</a>';
    });
    return s + '</div>';
  }).join('');
  document.getElementById('nav').innerHTML = html;

  var e = CAT.env || {};
  document.getElementById('env').innerHTML =
    'python ' + esc(e.python || '?') + '<br>numpy ' + esc(e.numpy || '未安装') +
    '<br>torch ' + esc(e.torch || '未安装') +
    '<br>' + (SNAP ? '快照模式（不能真跑）' : '实时模式：可调参真跑');
}

/* ------------------------------------------------------------------ 总览图 */
function overviewDiagram() {
  // 颜色一律走 CSS 变量, 这样浅色/深色主题下都清楚
  var seg = [
    ['磁盘 / 对象存储', 'var(--c1)'],
    ['CPU 内存 (ETL)', 'var(--c2)'],
    ['GPU 显存', 'var(--c3)'],
    ['算子执行', 'var(--c4)'],
    ['跨卡同步', 'var(--c5)']
  ];
  var w = 152, gap = 22, y = 34, h = 46;
  var boxes = seg.map(function (s, i) {
    var x = i * (w + gap);
    return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="7" ' +
      'fill="var(--bg2)" stroke="' + s[1] + '"/>' +
      '<text x="' + (x + w / 2) + '" y="' + (y + 28) + '" text-anchor="middle" ' +
      'fill="var(--fg)" font-size="13">' + s[0] + '</text>' +
      (i < seg.length - 1 ? '<path d="M' + (x + w + 3) + ' ' + (y + h / 2) + ' h ' + (gap - 8) +
        '" stroke="var(--fg-dim)" stroke-width="1.5" marker-end="url(#ar)"/>' : '');
  }).join('');
  var spans = [
    [0, 2, '01 存储：读得动、存得起', 'var(--c2)'],
    [2, 4, '02 编译：算得快', 'var(--c3)'],
    [3, 5, '03 网络：对得齐', 'var(--c5)']
  ];
  var bars = spans.map(function (s, k) {
    var x0 = s[0] * (w + gap), x1 = s[1] * (w + gap) - gap;
    var yy = y + h + 22 + k * 26;
    return '<line x1="' + x0 + '" y1="' + yy + '" x2="' + x1 + '" y2="' + yy +
      '" stroke="' + s[3] + '" stroke-width="3"/>' +
      '<text x="' + (x0 + 8) + '" y="' + (yy - 6) + '" fill="' + s[3] + '" font-size="12">' + s[2] + '</text>';
  }).join('');
  return '<div class="diagram"><svg viewBox="0 0 ' + (5 * (w + gap)) + ' 190" width="100%" height="190">' +
    '<defs><marker id="ar" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">' +
    '<path d="M0,0 L7,3.5 L0,7 z" fill="var(--fg-dim)"/></marker></defs>' +
    '<text x="0" y="16" fill="var(--fg-dim)" font-size="12">一次训练 step 的数据流向 —— 三个方向是这条线上的三段</text>' +
    boxes + bars + '</svg></div>';
}

/* ------------------------------------------------------------------ 模块页 */
function renderModule(id) {
  var m = CAT.modules.filter(function (x) { return x.id === id; })[0];
  var main = document.getElementById('main');
  if (!m) { main.innerHTML = '<p>没有这个模块。</p>'; return; }

  var head = '<h1>' + esc(m.title) + '</h1><p class="lead">' + esc(m.hook || '') + '</p>';
  if (id === '00_overview') head += overviewDiagram();

  if ((m.demos || []).length) {
    head += '<div class="cards">' + m.demos.map(function (d) {
      return '<a class="card" href="#/d/' + d.file + '"><div class="t">' + esc(d.title) + '</div>' +
        '<div class="h">' + esc(d.hook || '') + '</div>' +
        '<div class="meta">约 ' + (d.runtime_s || '?') + 's · ' +
        (d.params || []).length + ' 个可调参数 · ' +
        (d.checks || []).length + ' 条验证点</div></a>';
    }).join('') + '</div>';
  }

  main.innerHTML = head + '<div id="doc"><p class="hint">正在加载讲解…</p></div>';
  if (m.doc) {
    loadDoc(m.doc).then(function (md) {
      document.getElementById('doc').innerHTML = mdToHtml(md);
    }).catch(function (e) {
      document.getElementById('doc').innerHTML = '<p class="hint">讲解加载失败：' + esc(e.message) + '</p>';
    });
  } else {
    document.getElementById('doc').innerHTML = '';
  }
}

/* ------------------------------------------------------------------ demo 页 */
function currentParams(demo) {
  var out = {};
  (demo.params || []).forEach(function (p) {
    var inp = document.getElementById('p_' + p.name);
    if (!inp) return;
    if (p.type === 'bool') {
      if (inp.checked !== !!p.default) out[p.name] = inp.checked ? '1' : '0';
    } else if (String(inp.value) !== String(p.default)) {
      out[p.name] = inp.value;
    }
  });
  return out;
}

function paramHTML(p) {
  var id = 'p_' + p.name;
  if (p.type === 'bool') {
    return '<div class="param" id="w_' + p.name + '"><label>' + esc(p.label) + '</label>' +
      '<div class="row"><input type="checkbox" id="' + id + '"' + (p.default ? ' checked' : '') + '>' +
      '<span class="name">' + p.name + '</span></div></div>';
  }
  var step = p.step || 1;
  return '<div class="param" id="w_' + p.name + '"><label>' + esc(p.label) + '</label>' +
    '<div class="row">' +
    '<input type="range" id="r_' + p.name + '" min="' + p.min + '" max="' + p.max +
    '" step="' + step + '" value="' + p.default + '">' +
    '<input type="number" id="' + id + '" min="' + p.min + '" max="' + p.max +
    '" step="' + step + '" value="' + p.default + '">' +
    '</div><span class="name">' + p.name + '</span></div>';
}

function cmdLine(demo, params) {
  var pre = Object.keys(params).map(function (k) { return k + '=' + params[k]; }).join(' ');
  return (pre ? pre + ' ' : '') + 'python3 ' + demo.file;
}

function renderDemo(file) {
  var demo = findDemo(file);
  var main = document.getElementById('main');
  if (!demo) { main.innerHTML = '<p>没有这个 demo。</p>'; return; }
  var mod = moduleOf(file) || {};
  var run = RUNS[file];

  main.innerHTML =
    '<p class="hint"><a href="#/m/' + mod.id + '">← ' + esc(mod.title || '') + '</a></p>' +
    '<h1>' + esc(demo.title) + '</h1>' +
    '<p class="lead">' + esc(demo.hook || '') + '</p>' +

    '<div class="panel">' +
      '<h3>参数（改完直接重跑，不改任何代码）</h3>' +
      '<div class="params">' + (demo.params || []).map(paramHTML).join('') + '</div>' +
      '<div class="btnrow">' +
        '<button id="btnRun">运行</button>' +
        '<button id="btnStop" class="ghost" disabled>停止</button>' +
        '<button id="btnReset" class="ghost">恢复默认</button>' +
        '<button id="btnCopy" class="ghost">复制等价命令</button>' +
        '<span class="status" id="status"></span>' +
      '</div>' +
      '<div class="hint" id="cmd" style="margin-top:8px"></div>' +
      '<pre class="out" id="out"></pre>' +
    '</div>' +

    '<div class="panel">' +
      '<h3>验证点</h3>' +
      '<div id="checks"></div>' +
      '<div class="btnrow"><button id="btnReport" class="ghost">导出验证报告 (.md)</button></div>' +
    '</div>' +

    '<details class="src"><summary>查看源码 · ' + esc(demo.file) + '</summary>' +
      '<pre class="code"><code id="src">加载中…</code></pre></details>';

  // ---- 参数联动 ----
  (demo.params || []).forEach(function (p) {
    if (p.type === 'bool') {
      document.getElementById('p_' + p.name).addEventListener('change', syncCmd);
      return;
    }
    var r = document.getElementById('r_' + p.name), n = document.getElementById('p_' + p.name);
    r.addEventListener('input', function () { n.value = r.value; syncCmd(); });
    n.addEventListener('input', function () { r.value = n.value; syncCmd(); });
  });

  function syncCmd() {
    var ps = currentParams(demo);
    (demo.params || []).forEach(function (p) {
      var w = document.getElementById('w_' + p.name);
      if (w) w.className = 'param' + (ps[p.name] !== undefined ? ' changed' : '');
    });
    document.getElementById('cmd').textContent = '等价命令：' + cmdLine(demo, ps);
  }
  syncCmd();

  if (run) {
    document.getElementById('out').textContent = run.text;
    document.getElementById('status').textContent = run.status || '';
  }
  document.getElementById('checks').innerHTML = renderChecks(demo, run);

  document.getElementById('btnReset').onclick = function () {
    (demo.params || []).forEach(function (p) {
      if (p.type === 'bool') { document.getElementById('p_' + p.name).checked = !!p.default; }
      else {
        document.getElementById('p_' + p.name).value = p.default;
        document.getElementById('r_' + p.name).value = p.default;
      }
    });
    syncCmd();
  };
  document.getElementById('btnCopy').onclick = function () {
    navigator.clipboard.writeText(cmdLine(demo, currentParams(demo)));
    document.getElementById('status').textContent = '已复制到剪贴板';
  };
  document.getElementById('btnReport').onclick = function () { exportReport(demo); };
  document.getElementById('btnRun').onclick = function () { startRun(demo); };

  if (SNAP) {
    var b = document.getElementById('btnRun');
    b.disabled = true;
    b.textContent = '快照模式';
    document.getElementById('status').textContent = '这是导出的静态快照，输出为预先跑好的结果；要调参真跑请运行 python3 site/serve.py';
    if (SNAP.outputs[file] && !RUNS[file]) {
      RUNS[file] = { text: SNAP.outputs[file], params: {}, status: '（快照输出）' };
      document.getElementById('out').textContent = SNAP.outputs[file];
      document.getElementById('checks').innerHTML = renderChecks(demo, RUNS[file]);
    }
  }

  loadSource(file).then(function (t) {
    document.getElementById('src').textContent = t;
  }).catch(function () {
    document.getElementById('src').textContent = '（源码加载失败）';
  });
}

/* ------------------------------------------------------------------ 运行 */
var ES = null;
function startRun(demo) {
  if (ES) { ES.close(); ES = null; }
  var params = currentParams(demo);
  var q = Object.keys(params).map(function (k) {
    return encodeURIComponent(k) + '=' + encodeURIComponent(params[k]);
  });
  q.unshift('path=' + encodeURIComponent(demo.file));

  var out = document.getElementById('out');
  var st = document.getElementById('status');
  var btn = document.getElementById('btnRun');
  var stop = document.getElementById('btnStop');
  out.textContent = '';
  btn.disabled = true;
  stop.disabled = false;
  var t0 = Date.now();
  st.className = 'status run';
  st.textContent = '运行中…（预计 ' + (demo.runtime_s || '?') + 's）';

  RUNS[demo.file] = { text: '', params: params, status: '运行中' };
  ES = new EventSource('/api/run?' + q.join('&'));

  // 用户可能在跑的过程中切走; 那时页面上的元素已经换掉了, 只更新数据不碰 DOM
  function onPage() { return location.hash === '#/d/' + demo.file; }

  ES.addEventListener('line', function (e) {
    var line = JSON.parse(e.data);
    RUNS[demo.file].text += line + '\n';
    if (!onPage()) return;
    out.textContent += line + '\n';
    out.scrollTop = out.scrollHeight;
  });
  ES.addEventListener('done', function (e) {
    var d = JSON.parse(e.data);
    ES.close(); ES = null;
    if (!onPage()) { RUNS[demo.file].status = '已完成（在后台）'; return; }
    btn.disabled = false; stop.disabled = true;
    var secs = ((Date.now() - t0) / 1000).toFixed(1);
    if (d.timeout) {
      st.className = 'status err';
      st.textContent = '超时被终止（' + secs + 's）—— 参数可能开得太大了';
    } else if (d.code !== 0) {
      st.className = 'status err';
      st.textContent = '退出码 ' + d.code + '（' + secs + 's）';
    } else {
      st.className = 'status ok';
      st.textContent = '完成，用时 ' + secs + 's';
    }
    RUNS[demo.file].status = st.textContent;
    document.getElementById('checks').innerHTML = renderChecks(demo, RUNS[demo.file]);
  });
  ES.onerror = function () {
    if (!ES) return;
    ES.close(); ES = null;
    if (!onPage()) return;
    btn.disabled = false; stop.disabled = true;
    st.className = 'status err';
    st.textContent = '连接中断（serve.py 还在跑吗？）';
  };

  stop.onclick = function () {
    if (ES) { ES.close(); ES = null; }
    btn.disabled = false; stop.disabled = true;
    st.className = 'status err';
    st.textContent = '已停止';
  };
}

/* ------------------------------------------------------------------ 报告 */
function exportReport(demo) {
  var run = RUNS[demo.file];
  var lines = ['# 验证报告 · ' + demo.title, '', '- demo: `' + demo.file + '`',
    '- 时间: ' + new Date().toLocaleString()];
  var ps = run ? run.params : currentParams(demo);
  lines.push('- 参数: ' + (Object.keys(ps).length
    ? Object.keys(ps).map(function (k) { return '`' + k + '=' + ps[k] + '`'; }).join(' ')
    : '全部默认'));
  lines.push('- 等价命令: `' + cmdLine(demo, ps) + '`', '', '## 验证点', '');
  if (!run || !run.text) {
    lines.push('（还没有运行，无法比对）');
  } else {
    (demo.checks || []).forEach(function (c) {
      var r = evalCheck(run.text, c);
      lines.push('- [' + (r.s === 'pass' ? 'x' : ' ') + '] **' +
        { pass: '通过', fail: '不符', miss: '未匹配' }[r.s] + '** — ' + c.text);
      lines.push('  - ' + r.d);
    });
    lines.push('', '## 完整输出', '', '```', run.text.trimEnd(), '```');
  }
  download('verify-' + demo.file.replace(/[\/.]/g, '_') + '.md', lines.join('\n'));
}

/* ------------------------------------------------------------------ 路由 */
function route() {
  var h = (location.hash || '').replace(/^#/, '');
  if (!h) { location.hash = '#/m/' + CAT.modules[0].id; return; }
  renderNav();
  window.scrollTo(0, 0);
  if (h.indexOf('/d/') === 0) renderDemo(h.slice(3));
  else if (h.indexOf('/m/') === 0) renderModule(h.slice(3));
  else location.hash = '#/m/' + CAT.modules[0].id;
}

function boot(cat) {
  CAT = cat;
  document.title = cat.title || document.title;
  window.addEventListener('hashchange', route);
  route();
}

if (SNAP) {
  boot(SNAP.catalog);
} else {
  getJSON('/api/catalog').then(boot).catch(function (e) {
    document.getElementById('main').innerHTML =
      '<h1>加载失败</h1><p class="hint">' + esc(e.message) +
      '</p><p>请确认是通过 <code>python3 site/serve.py</code> 打开的。</p>';
  });
}
