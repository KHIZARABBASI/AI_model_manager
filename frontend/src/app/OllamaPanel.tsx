"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import styles from "./page.module.css";

type ManagedFile = {
  path: string;
  size_bytes: number | null;
};

type InstalledModel = {
  model_id: string;
  files: ManagedFile[];
  total_bytes: number;
};

type RuntimeModel = {
  name: string;
  size_bytes: number | null;
  format: string | null;
  running: boolean;
};

type RuntimeStatus = {
  available: boolean;
  cli_available: boolean;
  version: string | null;
  models: RuntimeModel[];
  message: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type OllamaPanelProps = {
  apiBaseUrl: string;
  installedModels: InstalledModel[];
};

function formatBytes(bytes: number | null): string {
  if (bytes === null) {
    return "Size unknown";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function importableGgufFiles(files: ManagedFile[]): ManagedFile[] {
  return files.filter((file) => {
    if (!file.path.toLowerCase().endsWith(".gguf")) {
      return false;
    }
    const shard = file.path.match(/-(\d{5})-of-(\d{5})\.gguf$/i);
    return !shard || shard[1] === "00001";
  });
}

export default function OllamaPanel({
  apiBaseUrl,
  installedModels,
}: OllamaPanelProps) {
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [importing, setImporting] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [sending, setSending] = useState(false);
  const activeModel = runtime?.models.some(
    (model) => model.name === selectedModel,
  )
    ? selectedModel
    : runtime?.models[0]?.name ?? "";

  const refreshRuntime = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch(`${apiBaseUrl}/runtime/ollama`, { signal });
      if (!response.ok) {
        throw new Error(`Ollama status returned ${response.status}`);
      }
      setRuntime((await response.json()) as RuntimeStatus);
      setRuntimeError(null);
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      setRuntime(null);
      setRuntimeError(
        error instanceof Error ? error.message : "Ollama status is unavailable.",
      );
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    const controller = new AbortController();
    const initialCheck = setTimeout(() => {
      void refreshRuntime(controller.signal);
    }, 0);
    const timer = setInterval(() => {
      void refreshRuntime(controller.signal);
    }, 5000);
    return () => {
      controller.abort();
      clearTimeout(initialCheck);
      clearInterval(timer);
    };
  }, [refreshRuntime]);

  async function importModel(modelId: string, filePath: string) {
    const importKey = `${modelId}/${filePath}`;
    setImporting(importKey);
    setActionError(null);
    setActionMessage(null);
    try {
      const response = await fetch(`${apiBaseUrl}/runtime/ollama/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, file_path: filePath }),
      });
      const data = (await response.json()) as {
        model_name?: string;
        message?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(data.detail ?? `Ollama import returned ${response.status}`);
      }
      setActionMessage(
        `${data.message ?? "Model imported."} Name: ${data.model_name ?? "unknown"}`,
      );
      await refreshRuntime();
    } catch (error: unknown) {
      setActionError(
        error instanceof Error ? error.message : "Ollama import failed.",
      );
    } finally {
      setImporting(null);
    }
  }

  async function runModelAction(action: "load" | "stop") {
    if (!activeModel) {
      return;
    }
    setActionError(null);
    setActionMessage(null);
    try {
      const response = await fetch(`${apiBaseUrl}/runtime/ollama/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: activeModel }),
      });
      const data = (await response.json()) as {
        message?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(data.detail ?? `Ollama ${action} returned ${response.status}`);
      }
      setActionMessage(data.message ?? `Model ${action} request completed.`);
      await refreshRuntime();
    } catch (error: unknown) {
      setActionError(
        error instanceof Error ? error.message : `Could not ${action} model.`,
      );
    }
  }

  async function removeOllamaModel() {
    if (!activeModel) {
      return;
    }
    if (
      !window.confirm(
        `Remove ${activeModel} from Ollama? Its managed GGUF source file will remain.`,
      )
    ) {
      return;
    }
    setActionError(null);
    setActionMessage(null);
    try {
      const response = await fetch(`${apiBaseUrl}/runtime/ollama/models`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: activeModel, confirm: true }),
      });
      const data = (await response.json()) as {
        message?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(data.detail ?? `Ollama remove returned ${response.status}`);
      }
      setActionMessage(data.message ?? "Model removed from Ollama.");
      setSelectedModel("");
      setMessages([]);
      await refreshRuntime();
    } catch (error: unknown) {
      setActionError(
        error instanceof Error ? error.message : "Could not remove Ollama model.",
      );
    }
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = chatInput.trim();
    if (!content || !activeModel || sending) {
      return;
    }
    const nextMessages = [...messages, { role: "user" as const, content }];
    setMessages(nextMessages);
    setChatInput("");
    setSending(true);
    setActionError(null);
    setActionMessage(null);
    try {
      const response = await fetch(`${apiBaseUrl}/runtime/ollama/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: activeModel,
          messages: nextMessages.slice(-40),
        }),
      });
      const data = (await response.json()) as {
        message?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(data.detail ?? `Chat returned ${response.status}`);
      }
      setMessages([...nextMessages, { role: "assistant", content: data.message ?? "" }]);
      await refreshRuntime();
    } catch (error: unknown) {
      setMessages(messages);
      setChatInput(content);
      setActionError(
        error instanceof Error ? error.message : "Local chat request failed.",
      );
    } finally {
      setSending(false);
    }
  }

  return (
    <section className={styles.runtimeCard} aria-labelledby="ollama-title">
      <div className={styles.runtimeHeading}>
        <div>
          <p className={styles.sectionEyebrow}>Local inference</p>
          <h2 id="ollama-title">Ollama</h2>
        </div>
        <button
          className={styles.secondaryButton}
          type="button"
          onClick={() => void refreshRuntime()}
        >
          Check status
        </button>
      </div>

      <p className={styles.runtimeNotice}>
        {runtime?.message ?? runtimeError ?? "Checking local Ollama service..."}
        {runtime?.version ? ` Version ${runtime.version}.` : ""}
      </p>
      {runtime && !runtime.cli_available && (
        <p className={styles.runtimeNotice}>
          The Ollama CLI is not on PATH. Make the CLI available yourself; this app
          does not download or install runtimes.
        </p>
      )}
      {!runtime?.available && (
        <p className={styles.runtimeHint}>
          Importing, loading, and chatting require Ollama running locally at
          127.0.0.1:11434. Prompts are not sent to a cloud API.
        </p>
      )}

      <div className={styles.importList}>
        <h3>Import managed GGUF files</h3>
        {installedModels.length === 0 ? (
          <p>Download GGUF model files first.</p>
        ) : (
          installedModels.map((model) => {
            const ggufFiles = importableGgufFiles(model.files);
            return ggufFiles.map((file) => {
              const importKey = `${model.model_id}/${file.path}`;
              return (
                <div className={styles.importRow} key={importKey}>
                  <div>
                    <strong>{model.model_id}</strong>
                    <span>{file.path} · {formatBytes(file.size_bytes)}</span>
                  </div>
                  <button
                    className={styles.primaryButton}
                    type="button"
                    disabled={
                      !runtime?.available ||
                      !runtime.cli_available ||
                      importing !== null
                    }
                    onClick={() => void importModel(model.model_id, file.path)}
                  >
                    {importing === importKey ? "Importing..." : "Import into Ollama"}
                  </button>
                </div>
              );
            });
          })
        )}
      </div>

      {runtime?.available && runtime.models.length > 0 && (
        <div className={styles.localChat}>
          <div className={styles.modelControls}>
            <label htmlFor="ollama-model">AI Model Manager model</label>
            <select
              id="ollama-model"
              value={activeModel}
              disabled={sending}
              onChange={(event) => {
                setSelectedModel(event.target.value);
                setMessages([]);
              }}
            >
              {runtime.models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}{model.running ? " · running" : ""}
                </option>
              ))}
            </select>
            <button
              className={styles.secondaryButton}
              type="button"
              disabled={
                sending ||
                !activeModel ||
                runtime.models.find((m) => m.name === activeModel)?.running
              }
              onClick={() => void runModelAction("load")}
            >
              Load model
            </button>
            <button
              className={styles.secondaryButton}
              type="button"
              disabled={
                sending ||
                !runtime.models.find((m) => m.name === activeModel)?.running
              }
              onClick={() => void runModelAction("stop")}
            >
              Stop model
            </button>
            <button
              className={styles.removeButton}
              type="button"
              disabled={sending || !activeModel}
              onClick={() => void removeOllamaModel()}
            >
              Remove from Ollama...
            </button>
          </div>

          <div className={styles.chatMessages} aria-live="polite">
            {messages.length === 0 ? (
              <p>
                Chat history is kept only in this page. Sending a message loads
                the selected model if needed.
              </p>
            ) : (
              messages.map((message, index) => (
                <article
                  className={message.role === "user" ? styles.userMessage : styles.assistantMessage}
                  key={`${message.role}-${index}`}
                >
                  <strong>{message.role === "user" ? "You" : "Model"}</strong>
                  <p>{message.content}</p>
                </article>
              ))
            )}
            {sending && <p>Generating locally...</p>}
          </div>

          <form className={styles.chatForm} onSubmit={sendMessage}>
            <label htmlFor="chat-prompt">Message the local model</label>
            <textarea
              id="chat-prompt"
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              disabled={sending}
              maxLength={12000}
              rows={3}
              placeholder="Your prompt is sent to Ollama on this device."
            />
            <button
              className={styles.primaryButton}
              type="submit"
              disabled={!activeModel || !chatInput.trim() || sending}
            >
              {sending ? "Generating..." : "Send"}
            </button>
          </form>
        </div>
      )}

      {actionMessage && <p className={styles.runtimeSuccess}>{actionMessage}</p>}
      {actionError && <p className={styles.downloadError}>{actionError}</p>}
      {runtimeError && <p className={styles.downloadError}>{runtimeError}</p>}
    </section>
  );
}
