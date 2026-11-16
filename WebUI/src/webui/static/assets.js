import { $, esc, api, header, CN, fmtS } from "/static/common.js";
import { AssetPreview } from "/static/preview.js";

const CATS = ["desk", "sofa", "chair", "table", "cabinet", "goal"];
const OPTS = [
  ["concept_image", "文生图（概念图）", "文本先出一张产品图，再图生 3D；关闭则直接文生 3D", true, "text"],
  ["four_view", "单图生四视图", "一张图先生成四个视角，再多视角图生 3D（背面更完整）", false, "any"],
  ["part_split", "部件拆分", "把模型拆成部件，逐部件生成碰撞凸包", true, "any"],
  ["articulation", "铰接（URDF）", "识别抽屉、柜门等可动部件，转成 MuJoCo 滑动/转动关节（需要部件拆分）", true, "any"],
  ["material_transfer", "材质迁移", "按照片重新贴图，让颜色材质跟着照片走（照片经 Aholo 官方上传；约 4–5 分钟，多 1–2 次积分）", true, "image"],
  ["export", "多格式导出", "USDZ / OBJ / STL，一起放进资产包", true, "any"],
];
const STEP_CN = { request: "解析请求", concept_image: "Lux3D 文生图", four_view: "Lux3D 四视图", img_to_3d: "Lux3D 图生 3D",
  text_to_3d: "Lux3D 文生 3D", material_reference: "Lux3D 图 + 文改图（材质参考）", material_transfer: "Lux3D 材质迁移", part_split: "Lux3D 部件拆分", articulation: "Lux3D 铰接",
  export: "Lux3D 多格式导出", process: "仿真资产处理" };
let cat = "desk", mode = "text", status = null, preview = null;
const images = [];
const selected = new Set();

function seg() {
  $("#cat-seg").innerHTML = CATS.map((c) => `<button type="button" data-c="${c}" aria-pressed="${c === cat}">${CN[c]}</button>`).join("");
  document.querySelectorAll("#cat-seg button").forEach((b) => (b.onclick = () => { cat = b.dataset.c; seg(); renderOpts(); }));
}

function renderOpts() {
  $("#opts").innerHTML = OPTS.filter(([, , , , m]) => m === "any" || m === mode).map(([k, t, d, on]) =>
    `<label class="row small" style="align-items:flex-start;flex-wrap:nowrap"><input type="checkbox" data-o="${k}" ${on ? "checked" : ""} style="margin-top:4px">
     <span><b>${t}</b><br><span class="muted">${d}</span></span></label>`).join("");
  document.querySelectorAll("[data-o]").forEach((c) => (c.onchange = estimate));
  estimate();
}

function opts() {
  const o = {};
  document.querySelectorAll("[data-o]").forEach((c) => (o[c.dataset.o] = c.checked));
  if (mode === "image") o.concept_image = false;
  if (!("material_transfer" in o)) o.material_transfer = false;
  return o;
}

function estimate() {
  const o = opts();
  let n = 1;                                             // the 3D generation itself
  if (mode === "text" && o.concept_image) n += 1;
  if (o.four_view && (mode === "text" ? o.concept_image : images.length === 1)) n += 1;
  if (o.material_transfer && mode === "image") n += 2;      // transfer + export of the re-textured mesh
  if (o.part_split || o.articulation) n += 1;
  if (o.articulation && (cat === "desk")) n += 1;
  if (o.export) n += 1;
  $("#credit-note").textContent = `本次最多 ${n} 次 Lux3D 任务（同样的请求已做过的步骤直接复用，不花积分）。` +
    (status ? `已用 ${status.credits_used} / ${status.credit_cap}（${status.credits_source === "server" ? "Lux3D 服务端计数" : "本地归档计数"}）。` : "") +
    (status && !status.key_set ? " 没有检测到 AHOLO_API_KEY：只能复用已有结果。" : "");
}

function setMode(m) {
  mode = m;
  $("#m-text").setAttribute("aria-pressed", m === "text");
  $("#m-image").setAttribute("aria-pressed", m === "image");
  $("#text-box").hidden = m !== "text";
  $("#image-box").hidden = m !== "image";
  renderOpts();
}

function renderSlots() {
  const box = $("#slots");
  box.innerHTML = images.map((im, i) => `<button type="button" class="slot has" data-i="${i}" aria-label="移除第 ${i + 1} 张图">
      <img src="${im.url}" alt=""><span><b>第 ${i + 1} 张</b><span class="small muted">${esc(im.name)} · 点击移除</span></span></button>`).join("") +
    (images.length < 8 ? `<button type="button" class="slot" id="add-img"><span aria-hidden="true" style="font-size:22px">＋</span><span><b>添加图片</b><span class="small muted">JPEG / PNG / WebP</span></span></button>` : "");
  box.querySelectorAll("[data-i]").forEach((b) => (b.onclick = () => { images.splice(+b.dataset.i, 1); renderSlots(); estimate(); }));
  const add = $("#add-img");
  if (add) add.onclick = () => {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = "image/jpeg,image/png,image/webp"; inp.multiple = true;
    inp.onchange = () => [...inp.files].slice(0, 8 - images.length).forEach((f) => {
      const rd = new FileReader();
      rd.onload = () => { images.push({ name: f.name, type: f.type, data: String(rd.result).split(",")[1], url: rd.result }); renderSlots(); estimate(); };
      rd.readAsDataURL(f);
    });
    inp.click();
  };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function renderSteps(state, order) {
  $("#steps").innerHTML = order.map((k, i) => {
    const s = state[k] || {};
    const cls = s.status === "done" ? "done" : s.status === "start" ? "run" : s.status === "error" ? "err" : "";
    const mark = s.status === "done" ? "✓" : s.status === "error" ? "!" : String(i + 1).padStart(2, "0");
    return `<li class="${cls}"><span class="dot" aria-hidden="true">${mark}</span><div>
      <div class="stitle"><span>${STEP_CN[k] || k}</span><span class="muted small">${s.badge || ""}</span></div>
      <div class="sdetail">${esc(s.detail || "")}</div>${cls === "run" ? '<div class="bar"><i></i></div>' : ""}</div></li>`;
  }).join("");
}

function detailOf(e) {
  if (e.task_id) return `任务 ${e.task_id}${e.cached ? "（复用已有结果）" : e.seconds ? ` · ${fmtS(e.seconds)}` : ""}` +
    (e.files ? ` · ${e.files.map((f) => f.type).join(" / ")}` : "");
  if (e.stage === "process" && e.hulls) return `${e.hulls} 个碰撞凸包，${e.articulated ? "带可动关节，" : ""}主体 ${(+e.mass_kg).toFixed(1)} kg；${e.front}`;
  if (e.stage === "request" && e.prompt) return e.prompt;
  return e.error || e.why || "";
}

async function follow(name) {
  $("#run-title").textContent = `生成中 · ${name}`;
  $("#stop").hidden = false;
  $("#stop").onclick = () => api(`/api/odg/asset/${name}/stop`, { method: "POST" });
  const state = {}, order = [], logs = [];
  let next = 0;
  const t0 = Date.now();
  for (;;) {
    let r;
    try { r = await api(`/api/odg/asset/${name}/events?from=${next}`); } catch (e) { await sleep(1500); continue; }
    next = r.next;
    for (const e of r.events) {
      if (e.stage === "log") { logs.push(e.msg); continue; }
      if (e.stage === "run") continue;
      if (!order.includes(e.stage)) order.push(e.stage);
      state[e.stage] = { status: e.status === "skipped" ? "done" : e.status, detail: detailOf(e) || state[e.stage]?.detail,
                         badge: e.cached ? "复用" : e.seconds ? fmtS(e.seconds) : "" };
      if (e.status === "done" && e.files && (e.stage === "concept_image" || e.stage === "four_view")) showThumbs(name);
    }
    renderSteps(state, order);
    $("#log").textContent = logs.slice(-150).join("\n") || r.stdout_tail;
    $("#log").scrollTop = $("#log").scrollHeight;
    $("#run-sub").textContent = `已用 ${fmtS((Date.now() - t0) / 1000)}`;
    if (r.finished) break;
    await sleep(1200);
  }
  $("#stop").hidden = true;
  $("#go").disabled = false;
  const d = await api(`/api/odg/asset/${name}`).catch(() => null);
  $("#run-title").textContent = `${d?.run?.status === "done" ? "完成" : "未完成"} · ${name}`;
  await loadLibrary();
  if (d?.run?.status === "done") openDetail(name);
}

async function showThumbs(name) {
  const d = await api(`/api/odg/asset/${name}`).catch(() => null);
  if (!d) return;
  const f = d.files;
  $("#thumbs").innerHTML = (f.concept ? `<figure><img src="${f.concept}" alt="Lux3D 概念图"><figcaption>Lux3D 文生图（概念图）</figcaption></figure>` : "") +
    (f.views.length ? `<figure><div class="row" style="gap:4px;flex-wrap:nowrap">${f.views.map((v) => `<img src="${v}" alt="" style="width:25%">`).join("")}</div><figcaption>Lux3D 单图生四视图</figcaption></figure>` : "");
}

async function start() {
  const err = $("#form-err");
  err.hidden = true;
  const size = {};
  [["width", "#sw"], ["depth", "#sd"], ["height", "#sh"]].forEach(([k, id]) => { if ($(id).value) size[k] = +$(id).value; });
  const body = { category: cat, mode, text: $("#desc").value.trim(), options: opts(), confirm_credits: $("#confirm").checked,
                 size: Object.keys(size).length === 3 ? size : null,
                 images: mode === "image" ? images.map((i) => ({ type: i.type, data: i.data })) : [] };
  $("#go").disabled = true;
  try {
    const r = await api("/api/odg/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $("#thumbs").innerHTML = "";
    follow(r.asset);
  } catch (e) { err.hidden = false; err.textContent = e.message; $("#go").disabled = false; }
}

async function loadLibrary() {
  const rows = await api("/api/odg/assets").catch(() => []);
  $("#gallery").innerHTML = rows.length ? rows.map((a) => {
    return `<div class="asset">
      <div class="row" style="justify-content:space-between;flex-wrap:nowrap">
        <h3>${CN[a.category] || a.category} <span class="mono muted small">${esc(a.asset)}</span></h3>
        ${a.status === "done" ? `<label class="small row" style="flex-wrap:nowrap"><input type="checkbox" data-sel="${esc(a.asset)}" ${selected.has(a.asset) ? "checked" : ""}> 选中</label>` : `<span class="badge neutral">${esc(a.status)}</span>`}
      </div>
      ${a.concept ? `<img src="${a.concept}" alt="" style="width:100%;height:150px;object-fit:cover;border:1px solid var(--line);margin:8px 0">` : ""}
      <div class="small muted">${esc((a.text || "").slice(0, 60))}</div>
      <div class="row" style="margin-top:8px">
        ${a.articulated ? '<span class="badge info">可动关节</span>' : ""}
        <span class="badge neutral">${a.hulls ?? "—"} 凸包</span>
      </div>
      <div class="small muted" style="margin-top:8px">Lux3D：${a.features.map(esc).join("、")}</div>
      ${a.status === "done" ? `<button type="button" class="btn" style="margin-top:10px;width:100%" data-open="${esc(a.asset)}">查看详情与 3D 预览</button>` : ""}
    </div>`;
  }).join("") : `<div class="empty">还没有资产。在上面生成第一个。</div>`;
  document.querySelectorAll("[data-sel]").forEach((c) => (c.onchange = () => {
    c.checked ? selected.add(c.dataset.sel) : selected.delete(c.dataset.sel);
    $("#sel-n").textContent = `已选 ${selected.size} 个`;
    $("#pack-btn").disabled = !selected.size;
  }));
  document.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () => openDetail(b.dataset.open)));
}

async function openDetail(name, view = {}) {
  const d = await api(`/api/odg/asset/${name}`);
  const sec = $("#detail");
  sec.hidden = false;
  const a = d.asset || {}, run = d.run, ph = a.physics || {}, n = a.normalization || {};
  const sz = a.size_m || {};
  const chain = (run.tasks || []).map((t) => `<tr><td>${esc(d.feature_cn[t.feature] || t.feature)}</td><td class="mono">${t.task_id ?? "—"}</td>
    <td>${t.cached ? '<span class="badge neutral">复用</span>' : t.seconds ? fmtS(t.seconds) : ""}</td></tr>`).join("");
  const joints = (a.articulation?.joints || []);
  sec.innerHTML = `
    <div class="head" style="display:flex;gap:16px;align-items:baseline;flex-wrap:wrap"><span class="kicker">ASSET · 资产详情</span>
      <h2 style="font-size:22px">${CN[a.category] || ""} <span class="mono muted" style="font-size:15px">${esc(name)}</span></h2></div>
    <p class="what muted" style="margin:6px 0 14px">${esc(run.prompt || run.request?.text || "")}</p>
    <div class="grid2">
      <div>
        <div class="viewer" id="pv" style="height:440px"><div class="loading" id="pv-msg">${d.preview_glb ? "加载 3D 预览…" : "没有预览文件"}</div></div>
        <div class="controls">
          <label><input type="checkbox" id="pv-vis" checked> 视觉层</label>
          <label><input type="checkbox" id="pv-col"> 碰撞凸包</label>
        </div>
        <div id="pv-joints" class="stack" style="margin-top:10px">${joints.map((j, i) => `<label class="controls" style="margin:0">
          <span class="small" style="min-width:120px">${esc(j.child)}（${j.type === "slide" ? "滑动" : "转动"} ${j.range[0]}–${j.range[1]}${j.type === "slide" ? " m" : " rad"}）</span>
          <input type="range" min="${j.range[0]}" max="${j.range[1]}" step="0.005" value="0" data-j="${i}"></label>`).join("")}</div>
      </div>
      <div class="stack">
        <dl class="kv">
          <dt>尺寸</dt><dd>${(+sz.depth_x).toFixed(2)} × ${(+sz.width_y).toFixed(2)} × ${(+sz.height_z).toFixed(2)} m（深×宽×高）；${n.scale_by === "lux3d" ? "保留 Lux3D 预测的真实尺寸" : `按${esc(n.scale_by)}缩放`}</dd>
          <dt>朝向</dt><dd>${esc(n.front || "")}</dd>
          <dt>碰撞</dt><dd>CoACD 凸包 ${a.collision?.hulls ?? "—"} 个${a.collision?.fit_stats?.links ? `（${Object.entries(a.collision.fit_stats.links).map(([k, v]) => `${k} ${v.hulls}`).join("，")}）` : ""}</dd>
          <dt>质量</dt><dd>${joints.length ? `主体 ${(+ph.mass).toFixed(1)} kg + ${joints.map((j) => `${j.child} ${j.mass_kg} kg`).join(" + ")}` : `${ph.mass} kg`}（按类别估计）</dd>
          <dt>摩擦</dt><dd>${(ph.friction || []).join(" / ")}；家具–地板 μ = ${ph.floor_friction ?? "—"}</dd>
          <dt>关节</dt><dd>${joints.length ? joints.map((j) => `${esc(j.child)}：${j.type}，轴 [${j.axis.map((x) => (+x).toFixed(2)).join(", ")}]`).join("<br>") : "无"}</dd>
          ${a.articulation?.corrections?.length ? `<dt>修正</dt><dd>${a.articulation.corrections.map((c) => `${esc(c.joint)} 往回移 ${(c.moved_back_m * 100).toFixed(1)} cm`).join("；")}</dd>` : ""}
          <dt>积分</dt><dd>本资产新花 ${run.credits_spent ?? 0} 次，用时 ${fmtS(run.seconds)}</dd>
        </dl>
        <details open><summary>Lux3D 任务链</summary><table class="t">${chain}</table></details>
        ${d.files.exports.length ? `<div class="row">${d.files.exports.map((e) => `<a class="btn" href="${e.url}" download>${e.type.toUpperCase()} · ${(e.bytes / 1e6).toFixed(1)} MB</a>`).join("")}</div>` : ""}
      </div>
    </div>
    <div class="figs" style="margin-top:16px">
      ${d.files.concept ? `<figure><img src="${d.files.concept}" alt=""><figcaption>Lux3D 文生图（概念图）</figcaption></figure>` : ""}
      ${d.files.views.length ? `<figure><div class="row" style="gap:4px;flex-wrap:nowrap">${d.files.views.map((v) => `<img src="${v}" alt="" style="width:25%">`).join("")}</div><figcaption>Lux3D 四视图</figcaption></figure>` : ""}
    </div>`;
  sec.scrollIntoView({ behavior: view.instant ? "instant" : "smooth", block: "start" });
  preview?.dispose();
  preview = null;
  if (d.preview_glb) {
    preview = new AssetPreview($("#pv"));
    try {
      await preview.load(d.preview_glb, { name, articulation: a.articulation });
      $("#pv-msg")?.remove();
      $("#pv-vis").onchange = (e) => preview.setLayer("vis", e.target.checked);
      $("#pv-col").onchange = (e) => preview.setLayer("col", e.target.checked);
      document.querySelectorAll("[data-j]").forEach((r) => (r.oninput = () => preview.setJoint(+r.dataset.j, +r.value)));
      if (view.col) { $("#pv-col").checked = true; preview.setLayer("col", true); }
      // open = the range end farthest from 0 (doors open towards negative angles)
      if (view.open) document.querySelectorAll("[data-j]").forEach((r) => { r.value = Math.abs(+r.min) > Math.abs(+r.max) ? r.min : r.max; r.oninput(); });
    } catch (e) { $("#pv-msg").textContent = `预览加载失败：${e.message}`; }
  }
}

async function pack() {
  $("#pack-btn").disabled = true;
  $("#pack-result").innerHTML = `<div class="note info">打包中（复制文件、生成预览、在包目录里自检场景）…</div>`;
  try {
    const r = await api("/api/odg/pack", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ assets: [...selected] }) });
    $("#pack-result").innerHTML = `<div class="note ok">资产包 <b>${esc(r.name)}</b>：${r.assets} 个资产，${(r.bytes / 1e6).toFixed(0)} MB。
      <a href="${r.zip}" download>下载 zip</a> · <a href="${r.readme}" target="_blank" rel="noopener">README</a></div>`;
  } catch (e) { $("#pack-result").innerHTML = `<div class="note bad">${esc(e.message)}</div>`; }
  $("#pack-btn").disabled = !selected.size;
}

(async function init() {
  status = await header("assets");
  seg();
  setMode("text");
  renderSlots();
  $("#m-text").onclick = () => setMode("text");
  $("#m-image").onclick = () => setMode("image");
  $("#go").onclick = start;
  $("#pack-btn").onclick = pack;
  await loadLibrary();
  if (status?.odg_running) { $("#go").disabled = true; follow(status.odg_running); }
  // deep link: /assets#asset=<name>[&open=1][&col=1] (drawers pulled out / collision hulls shown)
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get("asset")) openDetail(q.get("asset"), { instant: true, open: q.get("open") === "1", col: q.get("col") === "1" });
})();
