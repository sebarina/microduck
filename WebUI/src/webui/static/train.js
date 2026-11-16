import { $, esc, api, header, CN } from "/static/common.js";

const badge = (ok, yes, no) => `<span class="badge ${ok ? "ok" : "bad"}">${ok ? yes : no}</span>`;
const RES = { fail: ["bad", "未通过"], pending: ["warn", "待训练"], ok: ["ok", "通过"] };

function slotCard(s) {
  const fl = s.floor.map((x) => `<tr><td>地面绕行 · ${CN[x.asset.includes("sofa") ? "sofa" : "desk"]} <span class="mono muted">${esc(x.asset)}</span></td>
      <td>${badge(x.reached, "到达", x.failure === "timeout" ? "没走到" : esc(x.failure || "失败"))}</td>
      <td>${x.reached ? `${x.time_s} s，碰到家具 ${x.contact_steps} 步` : `跌倒 ${x.fell ? "是" : "否"}`}</td></tr>`).join("");
  const sf = s.surface.map((x) => `<tr><td>${esc(x.label)} <span class="mono muted">表面高 ${x.surface_top_z} m</span></td>
      <td>${badge(x.stayed_on_surface, "留在表面上", x.fell ? "跌倒" : "掉下表面")} ${x.moved ? '<span class="badge info">会动</span>' : '<span class="badge neutral">不动</span>'}</td>
      <td>前进 ${x.walk_speed_mps ?? "—"} m/s（命令 0.4），原地转 ${x.turn_rate_rps ?? "—"} rad/s（命令 0.9）</td></tr>`).join("");
  const vids = [...s.surface, ...s.floor].filter((x) => x.video_url).map((x) => `<figure>
      <video src="${x.video_url}" controls muted playsinline preload="metadata"></video>
      <figcaption>${x.label ? esc(x.label) : `地面绕行 · ${esc(x.asset)}`}</figcaption></figure>`).join("");
  const kind = s.kind === "trained" ? '<span class="badge info">用 OnDeskGen 资产训练</span>' : '<span class="badge neutral">对照：官方策略</span>';
  return `<section class="card">
    <div class="row" style="justify-content:space-between"><h3 style="margin:0">${esc(s.label)} ${kind}</h3>
      <span class="mono muted small">${esc(s.policy_file)} · sha256 ${esc(s.sha256)}</span></div>
    <p class="small" style="margin:8px 0">${esc(s.note)}</p>
    ${s.task_id ? `<p class="muted small" style="margin:0 0 8px">训练任务 <code>${esc(s.task_id)}</code>，checkpoint ${esc(s.checkpoint)}</p>` : ""}
    <div class="tablewrap"><table class="t"><tr><th>项目</th><th>结果</th><th>数据</th></tr>${sf}${fl}</table></div>
    <div class="slot-videos">${vids}</div></section>`;
}

(async function init() {
  await header("train");
  const d = await api("/api/train");
  const st = d.status;
  $("#status").innerHTML = `<div class="note ${st.level === "ok" ? "ok" : "warn"}"><b>${esc(st.title)}</b><br>${esc(st.text)}</div>`;
  $("#chain").innerHTML = d.chain.map((c, i) => `<div class="${i === d.chain.length - 1 ? "here" : ""}"><b>${esc(c.title)}</b>
      <span class="muted small">${esc(c.text)}</span></div>`).join("");
  $("#assets").innerHTML = d.assets.map((a) => `<figure style="margin:0">
      ${a.preview ? `<img src="${a.preview}" alt="${esc(a.asset)} 预览：视觉 / 视觉+碰撞 / 碰撞" style="width:100%;border:1px solid var(--line)">` : ""}
      <figcaption class="small"><b>${CN[a.category]}</b> <span class="mono muted">${esc(a.asset)}</span><br>
      <span class="muted">${a.size_m.depth_x.toFixed(2)}×${a.size_m.width_y.toFixed(2)}×${a.size_m.height_z.toFixed(2)} m · ${a.boxes} 个碰撞盒</span>
      </figcaption></figure>`).join("") || '<p class="muted">没有找到训练导出（运行 <code>ondeskgen train-export --all</code>）</p>';
  const names = { check_surfaces: "表面几何（check_surfaces.py）", validate_sofa_physics: "沙发软接触（validate_sofa_physics.py）" };
  $("#checks").innerHTML = Object.entries(d.checks).map(([k, c]) => `<details ${k === "check_surfaces" ? "open" : ""}>
      <summary>${names[k] || k} ${badge(c.passed, "通过", "未通过")} <span class="muted small">${esc(c.time)}</span></summary>
      <pre class="check">${esc(c.text)}</pre></details>`).join("") || '<p class="muted">还没运行</p>';
  $("#tasks").innerHTML = "训练任务：" + d.tasks.map((t) => `<code>${esc(t.id)}</code>（${esc(t.label)}）`).join("、");
  $("#stages").innerHTML = `<tr><th>阶段</th><th>做什么</th></tr>` +
    d.cloud_stages.map((s) => `<tr><td><code>${esc(s.cmd)}</code></td><td>${esc(s.text)}</td></tr>`).join("");
  $("#repo").innerHTML = d.training_repo ? `训练工程：<code>${esc(d.training_repo)}</code>（下面的 § 号指它的运行手册章节）。` : "";
  $("#history").innerHTML = d.history.map((h) => { const [cls, txt] = RES[h.result] || ["neutral", h.result];
    return `<li><span class="date">${esc(h.date)}</span><div><b>${esc(h.run)}</b> <span class="badge ${cls}">${txt}</span>
      <div class="small" style="margin-top:4px">${esc(h.summary)}</div>
      <div class="muted small">原因 / 下一步：${esc(h.cause)} <span class="mono">${esc(h.ref)}</span></div></div></li>`; }).join("");
  $("#slots").innerHTML = d.slots.map(slotCard).join("") || '<p class="muted">还没有策略槽结果</p>';
})();
