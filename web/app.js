import init, { load_graph } from "./pkg/gv_web.js";

const edges = document.querySelector("#edges-file");
const nodes = document.querySelector("#nodes-file");
const error = document.querySelector("#import-error");

await init();

async function openGraph(edgesText, nodesText) {
  document.querySelector("#import-panel").hidden = true;
  document.querySelector("#viewer").hidden = false;
  try {
    load_graph(edgesText, nodesText);
  } catch (reason) {
    document.querySelector("#import-panel").hidden = false;
    document.querySelector("#viewer").hidden = true;
    error.textContent = String(reason);
  }
}

document.querySelector("#open-graph").addEventListener("click", async () => {
  if (!edges.files[0] || !nodes.files[0]) {
    error.textContent = "Choose both repo.edges and nodes.tsv.";
    return;
  }
  await openGraph(await edges.files[0].text(), await nodes.files[0].text());
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
    await openGraph(await edgesResponse.text(), await nodesResponse.text());
  } catch (reason) {
    error.textContent = String(reason);
  }
});
