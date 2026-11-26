"use client";

import { useEffect, useState, type FormEvent } from "react";
import OllamaPanel from "./OllamaPanel";
import styles from "./page.module.css";

type HealthState =
  | { status: "checking"; message: string }
  | { status: "online"; message: string }
  | { status: "offline"; message: string };

type HardwareInfo = {
  os_name: string;
  os_version: string;
  architecture: string;
  cpu_name: string;
  logical_cpus: number;
  system_ram_bytes: number | null;
  free_disk_bytes: number | null;
  disk_path: string;
  gpus: Array<{
    name: string;
    vram_bytes: number | null;
    vram_source: string | null;
  }>;
  detection_warnings: string[];
};

type CatalogModel = {
  id: string;
  name: string;
  publisher: string | null;
  task: string | null;
  license: string | null;
  downloads: number;
  likes: number;
  gated: boolean;
  tags: string[];
  source_url: string;
  estimated_repository_size_bytes: number | null;
};

type CompatibilityInfo = {
  model_id: string;
  status: string;
  reasons: string[];
  assumptions: string[];
  estimated_runtime_memory_bytes: number | null;
  estimated_download_and_install_bytes: number | null;
  available_memory_bytes: number | null;
  available_disk_bytes: number | null;
};

type ModelFileInfo = {
  path: string;
  size_bytes: number | null;
};

type DownloadInfo = {
  id: string;
  model_id: string;
  status: "queued" | "downloading" | "completed" | "cancelled" | "failed";
  files: string[];
  current_file: string | null;
  downloaded_bytes: number;
  total_bytes: number | null;
  progress: number | null;
  speed_bytes_per_second: number | null;
  eta_seconds: number | null;
  error: string | null;
};

type InstalledModel = {
  model_id: string;
  files: ModelFileInfo[];
  total_bytes: number;
};

function formatBytes(bytes: number | null): string {
  if (bytes === null) {
    return "Unavailable";
  }

  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const digits = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

export default function Home() {
  const [health, setHealth] = useState<HealthState>({
    status: "checking",
    message: "Checking backend...",
  });
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [hardwareError, setHardwareError] = useState(false);
  const [query, setQuery] = useState("");
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [compatibility, setCompatibility] = useState<
    Record<string, CompatibilityInfo>
  >({});
  const [modelFiles, setModelFiles] = useState<
    Record<string, ModelFileInfo[]>
  >({});
  const [filesLoading, setFilesLoading] = useState<Record<string, boolean>>({});
  const [selectedFiles, setSelectedFiles] = useState<Record<string, string[]>>(
    {},
  );
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({});
  const [downloads, setDownloads] = useState<Record<string, DownloadInfo>>({});
  const [installedModels, setInstalledModels] = useState<InstalledModel[]>([]);
  const [installedError, setInstalledError] = useState<string | null>(null);
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${apiBaseUrl}/health`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Health check returned ${response.status}`);
        }

        const data = (await response.json()) as {
          status?: string;
          service?: string;
        };
        setHealth({
          status: "online",
          message: `${data.service ?? "Backend"} is healthy`,
        });
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setHealth({
          status: "offline",
          message: "Backend is not reachable",
        });
      });

    return () => controller.abort();
  }, [apiBaseUrl]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBaseUrl}/downloads/installed`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Installed models returned ${response.status}`);
        }
        setInstalledModels((await response.json()) as InstalledModel[]);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setInstalledError(
          error instanceof Error
            ? error.message
            : "Installed models could not be loaded.",
        );
      });

    return () => controller.abort();
  }, [apiBaseUrl]);

  useEffect(() => {
    const activeDownloads = Object.values(downloads).filter(
      (download) =>
        download.status === "queued" || download.status === "downloading",
    );
    if (activeDownloads.length === 0) {
      return;
    }

    let stopped = false;
    const timer = setInterval(() => {
      for (const download of activeDownloads) {
        void fetch(`${apiBaseUrl}/downloads/${download.id}`)
          .then(async (response) => {
            if (!response.ok) {
              throw new Error(`Download status returned ${response.status}`);
            }
            const latest = (await response.json()) as DownloadInfo;
            if (stopped) {
              return;
            }
            setDownloads((current) => ({
              ...current,
              [latest.model_id]: latest,
            }));
            if (latest.status === "completed" && !stopped) {
              const installedResponse = await fetch(
                `${apiBaseUrl}/downloads/installed`,
              );
              if (installedResponse.ok) {
                setInstalledModels(
                  (await installedResponse.json()) as InstalledModel[],
                );
              }
            }
          })
          .catch((error: unknown) => {
            if (!stopped) {
              setFileErrors((current) => ({
                ...current,
                [download.model_id]:
                  error instanceof Error
                    ? error.message
                    : "Download progress could not be refreshed.",
              }));
            }
          });
      }
    }, 1000);

    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [downloads, apiBaseUrl]);

  async function loadModelFiles(modelId: string) {
    setFilesLoading((current) => ({ ...current, [modelId]: true }));
    setFileErrors((current) => ({ ...current, [modelId]: "" }));
    try {
      const params = new URLSearchParams({ model_id: modelId });
      const response = await fetch(`${apiBaseUrl}/models/files?${params}`);
      const data = (await response.json()) as {
        detail?: string;
        files?: ModelFileInfo[];
      };
      if (!response.ok) {
        throw new Error(data.detail ?? `File list returned ${response.status}`);
      }
      setModelFiles((current) => ({
        ...current,
        [modelId]: data.files ?? [],
      }));
      setSelectedFiles((current) => ({ ...current, [modelId]: [] }));
    } catch (error: unknown) {
      setFileErrors((current) => ({
        ...current,
        [modelId]:
          error instanceof Error
            ? error.message
            : "Model files could not be loaded.",
      }));
    } finally {
      setFilesLoading((current) => ({ ...current, [modelId]: false }));
    }
  }

  async function startModelDownload(modelId: string) {
    const files = selectedFiles[modelId] ?? [];
    if (files.length === 0) {
      return;
    }
    setFileErrors((current) => ({ ...current, [modelId]: "" }));
    try {
      const response = await fetch(`${apiBaseUrl}/downloads`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, files }),
      });
      const data = (await response.json()) as DownloadInfo & { detail?: string };
      if (!response.ok) {
        throw new Error(data.detail ?? `Download returned ${response.status}`);
      }
      setDownloads((current) => ({ ...current, [modelId]: data }));
    } catch (error: unknown) {
      setFileErrors((current) => ({
        ...current,
        [modelId]:
          error instanceof Error ? error.message : "Download could not start.",
      }));
    }
  }

  async function cancelModelDownload(download: DownloadInfo) {
    try {
      const response = await fetch(`${apiBaseUrl}/downloads/${download.id}`, {
        method: "DELETE",
      });
      const data = (await response.json()) as DownloadInfo & { detail?: string };
      if (!response.ok) {
        throw new Error(data.detail ?? `Cancellation returned ${response.status}`);
      }
      setDownloads((current) => ({ ...current, [download.model_id]: data }));
    } catch (error: unknown) {
      setFileErrors((current) => ({
        ...current,
        [download.model_id]:
          error instanceof Error
            ? error.message
            : "Download could not be cancelled.",
      }));
    }
  }

  async function removeInstalledModel(modelId: string) {
    if (
      !window.confirm(
        `Permanently remove the managed files for ${modelId}? This cannot be undone.`,
      )
    ) {
      return;
    }

    try {
      const params = new URLSearchParams({
        model_id: modelId,
        confirm: "true",
      });
      const response = await fetch(
        `${apiBaseUrl}/downloads/installed?${params}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        const data = (await response.json()) as { detail?: string };
        throw new Error(data.detail ?? `Remove returned ${response.status}`);
      }
      setInstalledModels((current) =>
        current.filter((model) => model.model_id !== modelId),
      );
    } catch (error: unknown) {
      setInstalledError(
        error instanceof Error
          ? error.message
          : "Managed model could not be removed.",
      );
    }
  }

  async function searchModels(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setModelsLoading(true);
    setModelsError(null);

    try {
      const params = new URLSearchParams({
        query,
        limit: "20",
      });
      const response = await fetch(
        `${apiBaseUrl}/models/search?${params.toString()}`,
      );
      if (!response.ok) {
        throw new Error(`Catalog search returned ${response.status}`);
      }

      const data = (await response.json()) as { models: CatalogModel[] };
      setModels(data.models);

      const estimates = await Promise.all(
        data.models.map(async (model) => {
          const estimateResponse = await fetch(
            `${apiBaseUrl}/compatibility/estimate`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_id: model.id,
                repository_size_bytes:
                  model.estimated_repository_size_bytes,
              }),
            },
          );
          if (!estimateResponse.ok) {
            return null;
          }
          return (await estimateResponse.json()) as CompatibilityInfo;
        }),
      );

      setCompatibility(
        Object.fromEntries(
          estimates
            .filter(
              (estimate): estimate is CompatibilityInfo =>
                estimate !== null,
            )
            .map((estimate) => [estimate.model_id, estimate]),
        ),
      );
    } catch (error: unknown) {
      setModels([]);
      setModelsError(
        error instanceof Error
          ? error.message
          : "The model catalog could not be loaded.",
      );
    } finally {
      setModelsLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${apiBaseUrl}/hardware`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Hardware check returned ${response.status}`);
        }
        setHardware((await response.json()) as HardwareInfo);
        setHardwareError(false);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setHardwareError(true);
      });

    return () => controller.abort();
  }, [apiBaseUrl]);

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.intro}>
          <p className={styles.eyebrow}>Local AI model manager</p>
          <h1>Find, install, and run the right AI model locally.</h1>
          <p>
            A private, local-first workspace for discovering compatible models
            and managing them on your own computer.
          </p>
        </section>

        <aside
          className={`${styles.statusCard} ${styles[health.status]}`}
          aria-live="polite"
        >
          <span className={styles.statusDot} aria-hidden="true" />
          <div>
            <strong>Backend status</strong>
            <p>{health.message}</p>
          </div>
        </aside>

        {hardware && (
          <section className={styles.hardwareCard} aria-label="Hardware summary">
            <h2>Hardware summary</h2>
            <dl className={styles.hardwareGrid}>
              <div>
                <dt>System</dt>
                <dd>{hardware.os_name} {hardware.os_version}</dd>
              </div>
              <div>
                <dt>Architecture</dt>
                <dd>{hardware.architecture}</dd>
              </div>
              <div>
                <dt>Processor</dt>
                <dd>{hardware.cpu_name}</dd>
              </div>
              <div>
                <dt>Logical CPUs</dt>
                <dd>{hardware.logical_cpus}</dd>
              </div>
              <div>
                <dt>System RAM</dt>
                <dd>{formatBytes(hardware.system_ram_bytes)}</dd>
              </div>
              <div>
                <dt>Free disk space</dt>
                <dd>{formatBytes(hardware.free_disk_bytes)}</dd>
              </div>
              <div>
                <dt>GPU</dt>
                <dd>
                  {hardware.gpus.length
                    ? hardware.gpus
                        .map((gpu) =>
                          gpu.vram_bytes
                            ? `${gpu.name} (${formatBytes(gpu.vram_bytes)} VRAM)`
                            : gpu.name,
                        )
                        .join(", ")
                    : "No GPU detected"}
                </dd>
              </div>
            </dl>
            {hardware.detection_warnings.length > 0 && (
              <ul className={styles.warnings}>
                {hardware.detection_warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
          </section>
        )}

        {hardwareError && (
          <p className={styles.hardwareError}>
            Hardware information could not be loaded.
          </p>
        )}

        <section className={styles.catalogCard} aria-labelledby="catalog-title">
          <div className={styles.catalogHeader}>
            <div>
              <p className={styles.sectionEyebrow}>Model catalog</p>
              <h2 id="catalog-title">Discover text-generation models</h2>
            </div>
          </div>

          <form className={styles.searchForm} onSubmit={searchModels}>
            <label htmlFor="model-query">Search Hugging Face</label>
            <div>
              <input
                id="model-query"
                name="query"
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Try llama, phi, or mistral"
              />
              <button type="submit" disabled={modelsLoading}>
                {modelsLoading ? "Searching..." : "Search"}
              </button>
            </div>
          </form>

          {modelsError && (
            <p className={styles.hardwareError}>{modelsError}</p>
          )}

          {models.length > 0 && (
            <div className={styles.modelList}>
              {models.map((model) => (
                <article className={styles.modelCard} key={model.id}>
                  <div>
                    <h3>{model.name}</h3>
                    <p>
                      {model.publisher ?? "Unknown publisher"}
                      {model.gated ? " · Gated model" : ""}
                    </p>
                  </div>
                  <dl>
                    <div>
                      <dt>License</dt>
                      <dd>{model.license ?? "Not listed"}</dd>
                    </div>
                    <div>
                      <dt>Repository size</dt>
                      <dd>{formatBytes(model.estimated_repository_size_bytes)}</dd>
                    </div>
                    <div>
                      <dt>Downloads</dt>
                      <dd>{model.downloads.toLocaleString()}</dd>
                    </div>
                  </dl>
                  {compatibility[model.id] && (
                    <div className={styles.compatibility}>
                      <strong>{compatibility[model.id].status}</strong>
                      <p>{compatibility[model.id].reasons[0]}</p>
                    </div>
                  )}
                  <button
                    className={styles.secondaryButton}
                    type="button"
                    onClick={() => void loadModelFiles(model.id)}
                    disabled={filesLoading[model.id]}
                  >
                    {filesLoading[model.id]
                      ? "Loading files..."
                      : modelFiles[model.id]
                        ? "Refresh file list"
                        : "Choose files"}
                  </button>
                  <a href={model.source_url} target="_blank" rel="noreferrer">
                    View on Hugging Face
                  </a>
                  {modelFiles[model.id] && (
                    <div className={styles.filePicker}>
                      {modelFiles[model.id].length === 0 ? (
                        <p>
                          No supported model weights or configuration files
                          were found.
                        </p>
                      ) : (
                        <>
                          <p>
                            Choose files to download. Repository code and
                            unsupported formats are excluded.
                          </p>
                          {modelFiles[model.id].map((file) => (
                            <label key={file.path}>
                              <input
                                type="checkbox"
                                checked={(selectedFiles[model.id] ?? []).includes(
                                  file.path,
                                )}
                                onChange={(event) => {
                                  setSelectedFiles((current) => {
                                    const previous = current[model.id] ?? [];
                                    const next = event.target.checked
                                      ? [...previous, file.path]
                                      : previous.filter(
                                          (path) => path !== file.path,
                                        );
                                    return { ...current, [model.id]: next };
                                  });
                                }}
                              />
                              <span>{file.path}</span>
                              <small>{formatBytes(file.size_bytes)}</small>
                            </label>
                          ))}
                          <button
                            className={styles.primaryButton}
                            type="button"
                            onClick={() => void startModelDownload(model.id)}
                            disabled={
                              (selectedFiles[model.id] ?? []).length === 0 ||
                              downloads[model.id]?.status === "queued" ||
                              downloads[model.id]?.status === "downloading"
                            }
                          >
                            Download selected files
                          </button>
                        </>
                      )}
                    </div>
                  )}
                  {fileErrors[model.id] && (
                    <p className={styles.downloadError}>
                      {fileErrors[model.id]}
                    </p>
                  )}
                  {downloads[model.id] && (
                    <div className={styles.downloadStatus} aria-live="polite">
                      <strong>
                        {downloads[model.id].status === "completed"
                          ? "Download complete"
                          : downloads[model.id].status === "failed"
                            ? "Download failed"
                            : downloads[model.id].status === "cancelled"
                              ? "Download cancelled; partial files are kept for resuming"
                              : `Download ${downloads[model.id].status}`}
                      </strong>
                      {downloads[model.id].status === "downloading" && (
                        <>
                          <progress
                            max="1"
                            value={downloads[model.id].progress ?? undefined}
                            aria-label={`Download progress for ${model.name}`}
                          />
                          <p>
                            {formatBytes(downloads[model.id].downloaded_bytes)}
                            {downloads[model.id].total_bytes !== null
                              ? ` of ${formatBytes(downloads[model.id].total_bytes)}`
                              : " downloaded"}
                            {downloads[model.id].speed_bytes_per_second
                              ? ` · ${formatBytes(downloads[model.id].speed_bytes_per_second)}/s`
                              : ""}
                            {downloads[model.id].eta_seconds !== null
                              ? ` · about ${downloads[model.id].eta_seconds}s remaining`
                              : ""}
                          </p>
                          {downloads[model.id].current_file && (
                            <small>{downloads[model.id].current_file}</small>
                          )}
                          <button
                            className={styles.cancelButton}
                            type="button"
                            onClick={() =>
                              void cancelModelDownload(downloads[model.id])
                            }
                          >
                            Cancel download
                          </button>
                        </>
                      )}
                      {downloads[model.id].error && (
                        <p className={styles.downloadError}>
                          {downloads[model.id].error}
                        </p>
                      )}
                    </div>
                  )}
                </article>
              ))}
            </div>
          )}
        </section>

        <section className={styles.installedCard} aria-labelledby="installed-title">
          <h2 id="installed-title">Managed model files</h2>
          <p>
            Files are stored under the configured model directory. Removing a
            model here permanently deletes its managed files.
          </p>
          {installedError && (
            <p className={styles.downloadError}>{installedError}</p>
          )}
          {installedModels.length === 0 ? (
            <p>No completed model files are installed yet.</p>
          ) : (
            <ul>
              {installedModels.map((model) => (
                <li key={model.model_id}>
                  <div>
                    <strong>{model.model_id}</strong>
                    <span>
                      {model.files.length} file(s) · {formatBytes(model.total_bytes)}
                    </span>
                  </div>
                  <button
                    className={styles.removeButton}
                    type="button"
                    onClick={() => void removeInstalledModel(model.model_id)}
                  >
                    Remove...
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <OllamaPanel
          apiBaseUrl={apiBaseUrl}
          installedModels={installedModels}
        />
      </main>
    </div>
  );
}
