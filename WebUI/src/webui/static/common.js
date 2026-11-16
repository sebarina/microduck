// Shared helpers: header, fetch, formatting.
export const $ = (sel, root = document) => root.querySelector(sel);
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const CN = { sofa: "沙发", desk: "书桌", table: "茶几", chair: "椅子", cabinet: "边柜", goal: "球门", ball: "足球" };
export const SOURCE_CN = { default: "默认库存", library: "库存匹配", cache: "缓存（之前生成）", generated: "Lux3D 现场生成",
  fallback: "降级为库存", would_generate: "预演：将生成" };
export const fmtS = (s) => (s == null ? "—" : s < 60 ? `${Number(s).toFixed(1)} s` : `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`);

export async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

export async function header(active) {
  const h = document.createElement("header");
  h.className = "top";
  h.innerHTML = `
    <div class="brand"><svg width="30" height="22" viewBox="0 0 30 22" aria-hidden="true" fill="none">
      <rect x="1" y="1" width="20" height="20" stroke="#f5a623" stroke-width="1.5"/>
      <circle cx="11" cy="12" r="5" stroke="#e6ebf2" stroke-width="1.5"/><circle cx="13.5" cy="9.5" r="1.3" fill="#e6ebf2"/>
      <circle cx="26" cy="17" r="3" fill="#00e5ff"/></svg>
      <span>OnDeskGen</span></div>
    <nav class="main" aria-label="主导航">
      <a href="/" ${["demo", "try", "process"].includes(active) ? 'aria-current="page"' : ""}>演示视频</a>
      <a href="/assets" ${active === "assets" ? 'aria-current="page"' : ""}>资产引擎</a>
      <a href="/train" ${active === "train" ? 'aria-current="page"' : ""}>仿真训练</a>
    </nav>
    <div class="credits" id="credits"><span class="label">Lux3D 积分</span><span class="meter"><i style="width:0"></i></span><span class="n">—</span></div>`;
  document.body.prepend(h);
  try {
    const s = await api("/api/status");
    const pct = Math.min(100, (100 * s.credits_used) / s.credit_cap);
    $("#credits .meter i").style.width = pct + "%";
    $("#credits .n").textContent = `${s.credits_used} / ${s.credit_cap}`;
    return s;
  } catch (e) { return null; }
}
