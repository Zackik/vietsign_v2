export type PipelineInfo = {
  name: string;
  task: string;
  required_inputs: string[];
  ready: boolean;
  reason?: string | null;
  development_only: boolean;
};

export type ModelInfo = {
  name: string;
  task: string;
  modality: string;
  checkpoint?: string | null;
  loaded: boolean;
  parameter_count?: number | null;
  device?: string | null;
  development_only: boolean;
  warnings: string[];
};

const API_BASE = "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

async function requestForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Record<string, unknown>>("/health"),
  pipelines: () => request<{ pipelines: PipelineInfo[] }>("/pipelines"),
  models: () => request<{ models: ModelInfo[] }>("/models"),
  recognizeContinuous: (featuresPath: string) =>
    request<Record<string, unknown>>("/predict/continuous", {
      method: "POST",
      body: JSON.stringify({ features_path: featuresPath }),
    }),
  recognizeAndTranslate: (featuresPath: string) =>
    request<Record<string, unknown>>("/recognize-and-translate", {
      method: "POST",
      body: JSON.stringify({ features_path: featuresPath }),
    }),
  predictIsolatedVideo: (video: Blob, filename = "clip.webm", topK = 5) => {
    const form = new FormData();
    form.append("video", video, filename);
    form.append("model", "visignlanguage_tiny_rgb");
    form.append("top_k", String(topK));
    return requestForm<Record<string, unknown>>("/predict/isolated", form);
  },
};
