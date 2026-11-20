// Everything that touches the page. No network or capture logic here.

export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function log(text, cls = "") {
  const p = document.createElement("p");
  p.className = cls; p.textContent = text;
  $("log").appendChild(p); $("log").scrollTop = 1e9;
}

export function status(text) { $("status").textContent = text; }

export function hint(text) {
  const h = $("hint");
  h.textContent = text; h.hidden = !text;
  if (text) setTimeout(() => { if (h.textContent === text) h.hidden = true; }, 6000);
}

export function setConnected(on) {
  $("conn").classList.toggle("on", on);
  $("conn").title = on ? "Connected" : "Not connected";
}

export function setLocale(country, currency) { $("locale").textContent = `${country} · ${currency}`; }

// AR tracking chip: hidden outside AR; green when ready, amber while starting or lost.
export function setAr(state) {
  const chip = $("arChip");
  chip.hidden = !state;
  chip.className = `chip ${state === "ready" ? "ok" : "warn"}`;
  chip.textContent = { starting: "AR starting", ready: "AR tracking", lost: "AR lost" }[state] || "AR";
}

// One phase at a time: only the buttons that make sense now are shown.
const PHASES = {
  idle:       { step: "connect", show: ["btnConnect"] },
  connected:  { step: "sweep",   show: ["btnAR", "btnCam"] },
  sweeping:   { step: "sweep",   show: ["btnStop"] },
  processing: { step: "packet",  show: [] },
  done:       { step: "packet",  show: [] },
  failed:     { step: "packet",  show: [] },
};
export function setPhase(phase, { ar = true } = {}) {
  const p = PHASES[phase];
  for (const id of ["btnConnect", "btnAR", "btnCam", "btnStop"]) {
    const shown = p.show.includes(id) && !(id === "btnAR" && !ar);
    $(id).hidden = !shown; $(id).disabled = !shown;
  }
  const order = ["connect", "sweep", "packet"];
  for (const li of $("steps").children) {
    const i = order.indexOf(li.dataset.step), cur = order.indexOf(p.step);
    li.classList.toggle("active", i === cur && phase !== "done");
    li.classList.toggle("done", i < cur || phase === "done");
  }
}

export function renderInventory(m) {
  $("nBooks").textContent = m.book_count; $("nReadable").textContent = m.readable;
  $("nIdentified").textContent = m.identified; $("nItems").textContent = m.items.length;
  $("tElapsed").textContent = m.elapsed_s;
  $("items").innerHTML = m.items.map((i) => `<li>${esc(i.label)}</li>`).join("");
  $("books").innerHTML = m.books.slice().reverse().map((b) =>
    `<li value="${b.n}">${b.title ? `<span class="t">${esc(b.title)}</span>`
      : (b.text ? `“${esc(b.text)}”` : `<span class="muted">spine (unit ${b.unit})</span>`)}</li>`).join("");
  $("emptyInv").hidden = m.books.length + m.items.length > 0;
}

export function renderPacket(m) {
  const t = m.totals, cur = t.currency || "";
  const money = (v) => (v ? `${Math.round(v).toLocaleString()} ${cur}` : "—");
  const el = $("packet");
  el.hidden = false;
  el.innerHTML = `
    <h3>Claim packet ready <span class="muted small">built in ${m.seconds} s</span></h3>
    <div class="totals">
      <div>Books<b>${t.book_count}</b>${t.books_identified} identified</div>
      <div>Books, replacement<b>${money(t.books_replacement_cost)}</b>used ${money(t.books_used_value)}</div>
      <div>Other items<b>${t.items_replacement_cost_high ? money(t.items_replacement_cost_low) : "—"}</b>${
        t.items_replacement_cost_high ? `to ${money(t.items_replacement_cost_high)}` : "none priced"}</div>
      <div>Need review<b>${t.excluded_from_totals}</b>lines not in totals</div>
    </div>
    <div class="row">
      <a class="btn primary" href="${m.report}" target="_blank" rel="noopener">Open report</a>
      ${m.pdf ? `<a class="btn" href="${m.pdf}" download="claim-${esc(m.sweep_id)}.pdf">Download PDF</a>` : ""}
      <a class="btn" href="${m.json}" download="claim_packet-${esc(m.sweep_id)}.json">Download JSON</a>
    </div>`;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Narrow screens show one panel at a time; wide screens show both and ignore the tabs.
export function initTabs() {
  const tabs = document.querySelectorAll(".tabs [role=tab]");
  const show = (name) => {
    for (const t of tabs) t.setAttribute("aria-selected", String(t.dataset.tab === name));
    for (const p of document.querySelectorAll(".panel")) p.toggleAttribute("hidden-tab", p.id !== `tab-${name}`);
  };
  for (const t of tabs) t.onclick = () => show(t.dataset.tab);
  show("inventory");
}
