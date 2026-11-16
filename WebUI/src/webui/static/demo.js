import { $, esc, api, header } from "/static/common.js";

const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

(async function init() {
  await header("demo");
  const d = await api("/api/demo").catch(() => null);
  const v = $("#demo");
  if (!d?.video) {
    v.replaceWith(Object.assign(document.createElement("div"), { className: "note warn",
      textContent: "还没有演示视频：运行 uv run python scripts/make_demo_video.py 生成" }));
    return;
  }
  v.src = d.video;
  $("#vmeta").textContent = `${mmss(d.duration_s || 0)} · 1280×720 · 无声，中文字幕 · 生成于 ${d.generated}`;
  const ol = $("#chapters");
  ol.innerHTML = d.chapters.map((c, i) => `<li><button type="button" data-t="${c.t}">
      <span class="mono">${mmss(c.t)}</span><span><b>${esc(c.title)}</b>${c.kicker ? ` <span class="muted small">${esc(c.kicker)}</span>` : ""}
      <span class="muted small sub">${esc((c.lines || [])[0] || "")}</span></span></button></li>`).join("");
  ol.querySelectorAll("button").forEach((b) => (b.onclick = () => { v.currentTime = +b.dataset.t; v.play(); }));
  const mark = () => {
    let cur = 0;
    d.chapters.forEach((c, i) => { if (v.currentTime >= c.t - 0.1) cur = i; });
    ol.querySelectorAll("li").forEach((li, i) => li.classList.toggle("on", i === cur));
  };
  v.addEventListener("timeupdate", mark);
  mark();
})();
