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

function verificationCard(v) {
  const names = {odg_sofa_concept_a: "沙发 concept-a（训练同款）", odg_sofa_text_b: "沙发 text-b",
    odg_desk_multiview_a: "书桌 multiview-a", odg_desk_drawers_a: "书桌 drawers-a",
    odg_chair_photo_a: "椅子 photo-a", odg_chair_photo_material_a: "椅子 photo-material-a"};
  const groups = Object.entries(names).map(([id, label], i) => {
    const videos = v.videos.filter(x => x.name.startsWith(id + "_")).map(x => {
      const m = x.name.match(/(ccw|cw)_(\d+\.\d+)$/);
      const caption = `${x.name.includes("_SOFT_") ? "软垫基线 · " : ""}${m?.[1] === "ccw" ? "逆时针" : "顺时针"} · ω=${m?.[1] === "ccw" ? "+" : "−"}${m?.[2]} rad/s`;
      return `<figure><video src="${esc(x.url)}" controls muted playsinline preload="none" aria-label="${esc(label + ' · ' + caption)}"></video><figcaption>${esc(caption)}</figcaption></figure>`;
    }).join("");
    return `<details ${i === 0 ? "open" : ""}><summary>${esc(label)}</summary><div class="slot-videos">${videos}</div></details>`;
  }).join("");
  const matched = v.sha256 === "54b144ffd8da36ccd60160462b6620be8cdc1b61e902205f9b43478e328df42e";
  return `<section class="card stack" id="odg-verification">
    <h3 style="margin:0">魔搭 c2_geo · 六物体绕圆行走验证 <span class="badge bad">未通过</span></h3>
    <p class="small">策略 <code>policy.onnx</code> · run 2026-11-14_23-07-20_c2_geo · 验证日期 2026-11-16<br>
      <span class="mono muted">SHA256 ${esc(v.sha256 || "本地文件缺失")}</span><br>${matched ? "" : badge(false, "", "本地策略与报告不匹配；以下为历史报告结果")}</p>
    <div class="note warn"><b>26 组：按指令绕圈 0 · 原地自旋 11 · 翻倒/跌落 12 · 偏离/走出 3</b><br>
      逆时针指令多表现为反向原地自旋；顺时针 13 组中 10 组翻倒或跌落。训练同款沙发的软垫基线也未按指令绕圈，报告判断表面差异不是主要原因。</div>
    <p class="muted small">本地 CPU + MuJoCo + BAM M6，50 Hz；六件家具 × 双方向 × 两档角速度，共 24 组，另加 2 组软垫基线。圆周出生、切向朝向，r=0.08 m，ω=±0.5 / ±0.9 rad/s，每组最长 12 秒。此处为绕圆任务验证，与上方官方策略的地面绕行和表面动作检查采用不同协议。</p>
    <p class="small"><a href="${esc(v.report_url)}" target="_blank" rel="noopener">查看完整验证报告与逐组指标</a> · <a href="${esc(v.markdown_url)}" target="_blank" rel="noopener">原始 Markdown 报告</a></p>
    ${groups}
  </section>`;
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
  const verification = d.odg_verification ? verificationCard(d.odg_verification) : "";
  const officialIndex = d.slots.findIndex(s => s.label === "official_velstand" || s.policy_file?.endsWith("velstand.onnx"));
  $("#slots").innerHTML = d.slots.map((s, i) => slotCard(s) + (i === officialIndex ? verification : "")).join("") +
    (officialIndex < 0 ? verification : "") || '<p class="muted">还没有策略槽结果</p>';
})();
