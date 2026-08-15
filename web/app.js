import init, { run } from "./pkg/gv_web.js";

const error = document.querySelector("#import-error");
const importPanel = document.querySelector("#import-panel");
const viewer = document.querySelector("#viewer");

await init();

// --- Environment detection (best-effort; UA strings are imperfect) ---------

function detectOS() {
  const platform = (navigator.userAgentData?.platform || navigator.platform || "").toLowerCase();
  const ua = navigator.userAgent.toLowerCase();
  if (platform.includes("win") || ua.includes("windows")) return "windows";
  if (platform.includes("mac") || ua.includes("mac os")) return "mac";
  if (platform.includes("linux") || ua.includes("linux")) return "linux";
  return "other";
}

function detectBrowser() {
  const ua = navigator.userAgent;
  if (/Edg\//.test(ua)) return "edge"; // must come before Chrome
  if (/Firefox\//.test(ua)) return "firefox";
  if (/Chrome\//.test(ua)) return "chrome";
  return "other";
}

// A "Copy" button for a flag URL / pref — web pages cannot open or toggle
// `chrome://`/`about:config` pages, so copy-to-clipboard is the best we can do.
function copyChip(value) {
  return `<code>${value}</code> <button type="button" class="copy" data-copy="${value}">Copy</button>`;
}

// Instructions tailored to the detected OS + browser. Leads with "use/update
// Chrome or Edge"; flag-flipping is a Linux-only fallback (and may be blocked on
// managed machines), and #enable-vulkan is Linux-only (Windows uses D3D12, macOS
// Metal).
function buildWebgpuHelp() {
  const os = detectOS();
  const browser = detectBrowser();
  const flagsHost = browser === "edge" ? "edge://flags" : "chrome://flags";

  let steps = "";
  if (browser === "firefox") {
    steps = `
      <li><strong>Recommended:</strong> switch to an up-to-date <strong>Chrome or Edge</strong>
        — Firefox's WebGPU is early and much slower today.</li>
      <li>To try Firefox anyway: open <code>about:config</code> and set
        ${copyChip("dom.webgpu.enabled")} to <em>true</em>
        (best in <strong>Firefox Nightly</strong>; Linux also needs a working Vulkan driver),
        then reload.</li>`;
  } else if (os === "linux") {
    steps = `
      <li>On Linux, WebGPU uses <strong>Vulkan</strong> and may need enabling. Open
        ${copyChip(`${flagsHost}/#enable-unsafe-webgpu`)} and
        ${copyChip(`${flagsHost}/#enable-vulkan`)}, set both to <em>Enabled</em>, relaunch,
        then verify at <code>chrome://gpu</code> → “WebGPU: Hardware accelerated”.</li>
      <li><em>If flags are disabled by IT policy</em>, try launching with
        ${copyChip("--enable-unsafe-webgpu --enable-features=Vulkan")}, or use a machine
        where WebGPU is available.</li>`;
  } else {
    // Windows / macOS: WebGPU is on by default in 113+; no flags (and no Vulkan flag).
    steps = `
      <li><strong>Update to the latest Chrome or Edge (113+)</strong> — WebGPU is enabled by
        default on Windows and macOS, so no flags are needed.</li>
      <li>Still not working? Check <code>chrome://gpu</code> → “WebGPU”. A managed/corporate
        browser or an old GPU driver can disable it.</li>`;
  }

  return `
    <div class="help">
      <p><strong>WebGPU is required and isn't available here.</strong> There is no WebGL2
      fallback — this viewer draws with a WebGPU-only pipeline. Fix it and reload:</p>
      <ul>
        ${steps}
        <li>Serve over <code>http://localhost</code> or HTTPS — WebGPU is blocked on
          <code>file://</code>.</li>
      </ul>
    </div>`;
}

// Copy-button handler (event delegation on the message area).
error.addEventListener("click", async (event) => {
  const button = event.target.closest("button.copy");
  if (!button) return;
  try {
    await navigator.clipboard.writeText(button.dataset.copy);
    const original = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = original), 1500);
  } catch {
    button.textContent = "Copy failed";
  }
});

// Checks that WebGPU is present and can actually produce an adapter (Firefox in
// particular may expose navigator.gpu yet fail to create one).
async function webgpuStatus() {
  if (!navigator.gpu) {
    return { ok: false, reason: "This browser does not expose WebGPU (navigator.gpu is undefined)." };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      return {
        ok: false,
        reason: "navigator.gpu is present but no WebGPU adapter is available — usually the GPU/Vulkan backend is not enabled.",
      };
    }
    return { ok: true };
  } catch (reason) {
    return { ok: false, reason: "WebGPU adapter request failed: " + reason };
  }
}

// Warn up front, before the user even picks files, if WebGPU is missing.
webgpuStatus().then((status) => {
  if (!status.ok) {
    error.innerHTML = `<span class="err">${status.reason}</span>` + buildWebgpuHelp();
  }
});

// Hands the two file texts to the wasm viewer, which starts the wgpu/egui event
// loop on #graph-canvas. Only called once the user has supplied files, and only
// after a WebGPU check passes.
async function open(edgesText, nodesText) {
  error.innerHTML = "";

  const status = await webgpuStatus();
  if (!status.ok) {
    error.innerHTML = `<span class="err">${status.reason}</span>` + buildWebgpuHelp();
    return; // stay on the import screen — do not start the renderer
  }

  importPanel.hidden = true;
  viewer.hidden = false;

  // Let the canvas take its full-viewport CSS size before winit reads it; winit
  // then sizes the backing store (see `fit_canvas_to_window` in gv-app).
  await new Promise((resolve) => requestAnimationFrame(resolve));

  try {
    run(edgesText, nodesText);
  } catch (reason) {
    importPanel.hidden = false;
    viewer.hidden = true;
    error.innerHTML = `<span class="err">${String(reason)}</span>`;
  }
}

document.querySelector("#open-graph").addEventListener("click", async () => {
  const edges = document.querySelector("#edges-file").files[0];
  const nodes = document.querySelector("#nodes-file").files[0];
  if (!edges || !nodes) {
    error.textContent = "Choose both repo.edges and nodes.tsv.";
    return;
  }
  await open(await edges.text(), await nodes.text());
});

document.querySelector("#open-example").addEventListener("click", async () => {
  try {
    const [edgesResponse, nodesResponse] = await Promise.all([
      fetch("examples/gv.edges"),
      fetch("examples/gv.nodes.tsv"),
    ]);
    if (!edgesResponse.ok || !nodesResponse.ok) {
      throw new Error("The bundled example files are unavailable.");
    }
    await open(await edgesResponse.text(), await nodesResponse.text());
  } catch (reason) {
    error.innerHTML = `<span class="err">${String(reason)}</span>`;
  }
});

// The winit event loop owns the canvas for the page's lifetime; the simplest way
// back to the import screen is a fresh page.
document.querySelector("#reload").addEventListener("click", () => location.reload());
