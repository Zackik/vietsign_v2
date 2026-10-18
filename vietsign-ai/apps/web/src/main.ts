import { api, type ModelInfo, type PipelineInfo } from "./api";

const statusPill = element<HTMLSpanElement>("status-pill");
const pipelineList = element<HTMLDivElement>("pipeline-list");
const modelList = element<HTMLDivElement>("model-list");
const result = element<HTMLPreElement>("video-result");
const featuresPath = element<HTMLInputElement>("features-path");
const camera = element<HTMLVideoElement>("camera");
const videoFile = element<HTMLInputElement>("video-file");
const videoPreview = element<HTMLVideoElement>("video-preview");
const liveLabel = element<HTMLElement>("live-label");
const liveConfidence = element<HTMLElement>("live-confidence");
let stream: MediaStream | null = null;
let liveRunning = false;
let mediaRecorder: MediaRecorder | null = null;

document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view ?? "home"));
});

element<HTMLButtonElement>("run-continuous").addEventListener("click", async () => {
  await runAction(() => api.recognizeContinuous(featuresPath.value));
});

element<HTMLButtonElement>("run-translate").addEventListener("click", async () => {
  await runAction(() => api.recognizeAndTranslate(featuresPath.value));
});

videoFile.addEventListener("change", () => {
  const file = videoFile.files?.[0];
  videoPreview.src = file ? URL.createObjectURL(file) : "";
});

element<HTMLButtonElement>("submit-video").addEventListener("click", async () => {
  const file = videoFile.files?.[0];
  if (!file) {
    result.textContent = "Choose a video file first.";
    return;
  }
  await runAction(() => api.predictIsolatedVideo(file, file.name, 5));
});

element<HTMLButtonElement>("start-camera").addEventListener("click", async () => {
  await startCamera();
});

element<HTMLButtonElement>("start-live").addEventListener("click", async () => {
  await startCamera();
  liveRunning = true;
  liveLabel.textContent = "Listening...";
  liveConfidence.textContent = "Confidence: -";
  void recordNextClip();
});

element<HTMLButtonElement>("stop-camera").addEventListener("click", () => {
  liveRunning = false;
  mediaRecorder?.state === "recording" && mediaRecorder.stop();
  mediaRecorder = null;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  camera.srcObject = null;
  liveLabel.textContent = "Stopped";
  liveConfidence.textContent = "Confidence: -";
});

await refresh();

async function refresh(): Promise<void> {
  try {
    const [health, pipelines, models] = await Promise.all([api.health(), api.pipelines(), api.models()]);
    statusPill.textContent = String(health.status ?? "unknown");
    statusPill.className = `pill ${health.real_vsl_ready ? "ready" : "dev"}`;
    renderPipelines(pipelines.pipelines);
    renderModels(models.models);
  } catch (error) {
    statusPill.textContent = "API offline";
    statusPill.className = "pill offline";
    pipelineList.textContent = String(error);
  }
}

async function startCamera(): Promise<void> {
  if (stream) {
    return;
  }
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  camera.srcObject = stream;
}

async function recordNextClip(): Promise<void> {
  if (!stream || !liveRunning || mediaRecorder?.state === "recording") {
    return;
  }
  const mimeType = preferredMimeType();
  const chunks: Blob[] = [];
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  mediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) {
      chunks.push(event.data);
    }
  });
  mediaRecorder.addEventListener("stop", async () => {
    const blob = new Blob(chunks, { type: mimeType || "video/webm" });
    await updateLivePrediction(blob);
    if (liveRunning) {
      window.setTimeout(() => void recordNextClip(), 350);
    }
  });
  mediaRecorder.start();
  window.setTimeout(() => {
    if (mediaRecorder?.state === "recording") {
      mediaRecorder.stop();
    }
  }, 2500);
}

async function updateLivePrediction(blob: Blob): Promise<void> {
  try {
    const payload = await api.predictIsolatedVideo(blob, `webcam-${Date.now()}.webm`, 5);
    const label = String(payload.label ?? "Unknown");
    const confidence = Number(payload.confidence ?? 0);
    liveLabel.textContent = label;
    liveConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  } catch (error) {
    liveLabel.textContent = "Prediction failed";
    liveConfidence.textContent = String(error);
  }
}

function preferredMimeType(): string {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
}

async function runAction(action: () => Promise<Record<string, unknown>>): Promise<void> {
  result.textContent = "Running...";
  try {
    result.textContent = JSON.stringify(await action(), null, 2);
  } catch (error) {
    result.textContent = String(error);
  }
}

function renderPipelines(pipelines: PipelineInfo[]): void {
  pipelineList.replaceChildren(...pipelines.map((item) => card(item.name, [
    `Task: ${item.task}`,
    `Ready: ${item.ready ? "yes" : "no"}`,
    item.reason ? `Reason: ${item.reason}` : "",
    item.development_only ? "Development only" : "",
  ])));
}

function renderModels(models: ModelInfo[]): void {
  modelList.replaceChildren(...models.map((item) => card(item.name, [
    `Task: ${item.task}`,
    `Modality: ${item.modality}`,
    `Checkpoint: ${item.checkpoint ?? "not configured"}`,
    `Loaded: ${item.loaded ? "yes" : "no"}`,
    item.development_only ? "Development only" : "",
    ...item.warnings,
  ])));
}

function card(title: string, lines: string[]): HTMLElement {
  const wrapper = document.createElement("article");
  wrapper.className = "card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  wrapper.append(heading);
  lines.filter(Boolean).forEach((line) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    wrapper.append(paragraph);
  });
  return wrapper;
}

function switchView(view: string): void {
  document.querySelectorAll(".view").forEach((section) => section.classList.toggle("active", section.id === view));
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`Missing element: ${id}`);
  }
  return found as T;
}
