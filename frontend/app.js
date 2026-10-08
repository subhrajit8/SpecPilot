const userIdInput = document.querySelector("#user-id");
const newThreadButton = document.querySelector("#new-thread");
const refreshButton = document.querySelector("#refresh-threads");
const threadList = document.querySelector("#thread-list");
const workspace = document.querySelector("#workspace");
const breadcrumbTitle = document.querySelector("#breadcrumb-title");
const toast = document.querySelector("#status-toast");
const generateForm = document.querySelector("#generate-form");
const generateButton = document.querySelector("#generate-button");
const complianceInput = document.querySelector("#compliance-file");
const requirementsInput = document.querySelector("#requirements-file");
const complianceName = document.querySelector("#compliance-name");
const requirementsName = document.querySelector("#requirements-name");

const USER_KEY = "specpilot-user-id";
const state = {
  threads: [],
  activeThreadId: null,
  toastTimer: null,
};

userIdInput.value = localStorage.getItem(USER_KEY) || "";

function notify(message, kind = "success") {
  window.clearTimeout(state.toastTimer);
  toast.textContent = message;
  toast.className = `status-toast visible${kind === "error" ? " error" : kind === "info" ? " info" : ""}`;
  state.toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 4200);
}

async function api(path, options = {}) {
  const userId = userIdInput.value.trim();
  if (!userId) {
    throw new Error("Enter your user ID in the lower-left corner to connect.");
  }
  localStorage.setItem(USER_KEY, userId);

  const headers = new Headers(options.headers || {});
  headers.set("X-User-Id", userId);
  const response = await fetch(path, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object" && payload !== null
      ? payload.detail || JSON.stringify(payload)
      : payload;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

async function streamApi(path, options = {}, onEvent = () => {}) {
  const userId = userIdInput.value.trim();
  if (!userId) {
    throw new Error("Enter your user ID in the lower-left corner to connect.");
  }
  localStorage.setItem(USER_KEY, userId);

  const headers = new Headers(options.headers || {});
  headers.set("X-User-Id", userId);
  headers.set("Accept", "text/event-stream");
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    const detail = typeof payload === "object" && payload !== null
      ? payload.detail || JSON.stringify(payload)
      : payload;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  if (!response.body) throw new Error("This browser does not support streamed responses.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = null;

  function dispatch(block) {
    let event = "message";
    const dataLines = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    const data = JSON.parse(dataLines.join("\n"));
    if (event === "error") throw new Error(data.detail || data.message || "Stream failed.");
    if (event === "complete") completed = data;
    onEvent(event, data);
  }

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) dispatch(block);
    if (done) break;
  }
  if (buffer.trim()) dispatch(buffer);
  if (!completed) throw new Error("The server ended the stream without a completion event.");
  return completed;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (label) button.querySelector("span:first-child").textContent = label;
}

function canGenerate() {
  generateButton.disabled = !(
    state.activeThreadId &&
    complianceInput.files.length &&
    requirementsInput.files.length
  );
}

function updateFileLabel(input, label) {
  const file = input.files[0];
  label.textContent = file ? file.name : "Choose a file";
  label.classList.toggle("has-file", Boolean(file));
  if (file) {
    const arrow = document.createElement("span");
    arrow.textContent = "✓";
    label.append(" ", arrow);
  } else {
    const arrow = document.createElement("span");
    arrow.textContent = "→";
    label.append(" ", arrow);
  }
  canGenerate();
}

function makeThreadButton(thread) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `thread-item${thread.thread_id === state.activeThreadId ? " active" : ""}`;
  button.addEventListener("click", () => openThread(thread));

  const title = document.createElement("span");
  title.className = "thread-title";
  title.textContent = thread.title || "New chat";
  const preview = document.createElement("span");
  preview.className = "thread-preview";
  preview.textContent = thread.last_message || "Add your source files to begin";
  button.append(title, preview);
  return button;
}

function renderThreadList() {
  threadList.replaceChildren();
  if (!state.threads.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "Your workspaces will show up here.";
    threadList.append(empty);
    return;
  }
  for (const thread of state.threads) {
    threadList.append(makeThreadButton(thread));
  }
}

async function loadThreads() {
  try {
    const data = await api("/threads");
    state.threads = Array.isArray(data.threads) ? data.threads : [];
    renderThreadList();
  } catch (error) {
    notify(error.message, "error");
  }
}

function resetSetupView(title = "New project") {
  breadcrumbTitle.textContent = title;
  workspace.replaceChildren(generateForm);
  generateForm.reset();
  document.querySelector("#instructions").value = "Generate the PRD and technical documentation.";
  updateFileLabel(complianceInput, complianceName);
  updateFileLabel(requirementsInput, requirementsName);
  canGenerate();
}

function renderDocumentView(thread, data, progressMessage = "") {
  const panel = document.createElement("div");
  panel.className = "document-workspace";

  const heading = document.createElement("div");
  heading.className = "document-heading";
  const kicker = document.createElement("div");
  kicker.className = "eyebrow";
  kicker.textContent = "YOUR DOCUMENTATION";
  const title = document.createElement("h1");
  title.textContent = thread.title || "Your workspace";
  const subtitle = document.createElement("p");
  subtitle.className = "welcome-copy";
  subtitle.textContent = "Review your generated documents or ask for a focused change.";
  heading.append(kicker, title, subtitle);
  if (progressMessage) {
    const progress = document.createElement("p");
    progress.className = "generation-progress";
    progress.textContent = progressMessage;
    heading.append(progress);
  }

  const tabs = document.createElement("div");
  tabs.className = "document-tabs";
  const contents = document.createElement("section");
  contents.className = "document-content";

  const documents = [
    ["PRD", data.prd || (progressMessage ? "The PRD is being prepared…" : "No PRD is available in this thread yet.")],
    ["Technical documentation", data.tech_doc || (progressMessage ? "The technical documentation is being prepared…" : "No technical documentation is available yet.")],
    ["Conversation", null],
  ];

  function showTab(index) {
    tabs.replaceChildren();
    documents.forEach(([name], tabIndex) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = `document-tab${tabIndex === index ? " selected" : ""}`;
      tab.textContent = name;
      tab.addEventListener("click", () => showTab(tabIndex));
      tabs.append(tab);
    });
    contents.replaceChildren();
    if (index < 2) {
      const pre = document.createElement("pre");
      pre.className = "document-markdown";
      pre.textContent = documents[index][1];
      contents.append(pre);
      return;
    }
    renderConversation(contents, Array.isArray(data.messages) ? data.messages : []);
  }

  panel.append(heading, tabs, contents);
  workspace.replaceChildren(panel);
  breadcrumbTitle.textContent = thread.title || "Workspace";
  showTab(0);
}

function renderConversation(container, messages) {
  const conversation = document.createElement("div");
  conversation.className = "conversation";
  for (const message of messages) {
    const bubble = document.createElement("article");
    bubble.className = `message-bubble ${message.role === "user" ? "user-message" : "assistant-message"}`;
    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "YOU" : "SPEC PILOT";
    const content = document.createElement("p");
    content.textContent = message.content ?? "";
    bubble.append(role, content);
    conversation.append(bubble);
  }
  const chatForm = document.createElement("form");
  chatForm.className = "chat-form";
  const input = document.createElement("textarea");
  input.name = "message";
  input.rows = 2;
  input.placeholder = "Ask a question or request a change...";
  input.required = true;
  const submit = document.createElement("button");
  submit.className = "primary-button";
  submit.type = "submit";
  submit.textContent = "Send message ↗";
  chatForm.append(input, submit);
  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message || !state.activeThreadId) return;
    submit.disabled = true;
    submit.textContent = "Thinking…";
    try {
      await streamApi("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.activeThreadId, message }),
      }, (event, data) => {
        if (event === "reply") notify("Assistant replied. Loading the updated thread…", "info");
      });
      await openThreadById(state.activeThreadId);
      await loadThreads();
    } catch (error) {
      notify(error.message, "error");
      submit.disabled = false;
      submit.textContent = "Send message ↗";
    }
  });
  conversation.append(chatForm);
  container.append(conversation);
}

async function openThreadById(threadId) {
  const thread = state.threads.find((item) => item.thread_id === threadId) || {
    thread_id: threadId,
    title: "Workspace",
  };
  await openThread(thread);
}

async function openThread(thread) {
  state.activeThreadId = thread.thread_id;
  renderThreadList();
  if (!thread.last_message) {
    resetSetupView(thread.title || "New project");
    return;
  }
  try {
    const data = await api(`/threads/${encodeURIComponent(thread.thread_id)}`);
    renderDocumentView(thread, data);
  } catch (error) {
    notify(error.message, "error");
  }
}

async function createThread() {
  try {
    const data = await api("/threads", { method: "POST" });
    const threadId = data.thread_id || data.session_id;
    if (!threadId) throw new Error("The API did not return a thread ID.");
    const thread = {
      thread_id: threadId,
      session_id: threadId,
      title: "New chat",
      last_message: null,
    };
    state.threads = [thread, ...state.threads.filter((item) => item.thread_id !== threadId)];
    state.activeThreadId = threadId;
    renderThreadList();
    resetSetupView("New project");
    notify("Workspace created. Add your two source files to continue.");
  } catch (error) {
    notify(error.message, "error");
  }
}

newThreadButton.addEventListener("click", createThread);
refreshButton.addEventListener("click", loadThreads);
userIdInput.addEventListener("change", async () => {
  localStorage.setItem(USER_KEY, userIdInput.value.trim());
  state.activeThreadId = null;
  await loadThreads();
});
complianceInput.addEventListener("change", () => updateFileLabel(complianceInput, complianceName));
requirementsInput.addEventListener("change", () => updateFileLabel(requirementsInput, requirementsName));

generateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.activeThreadId) {
    notify("Create a workspace before uploading files.", "error");
    return;
  }
  const formData = new FormData();
  formData.append("session_id", state.activeThreadId);
  formData.append("instructions", document.querySelector("#instructions").value.trim() || "Generate the PRD and technical documentation.");
  formData.append("compliance_pdf", complianceInput.files[0]);
  formData.append("requirements_doc", requirementsInput.files[0]);

  setBusy(generateButton, true, "Generating…");
  try {
    notify("Your documents are being generated. This may take a few minutes.", "info");
    const streamedDocuments = { prd: null, tech_doc: null, messages: [] };
    const progressLabels = {
      compliance_extractor: "Extracting compliance obligations…",
      requirements_analyst: "Analyzing requirements…",
      prd_writer: "Writing the PRD…",
      tech_doc_writer: "Writing technical documentation…",
    };
    const result = await streamApi("/generate", { method: "POST", body: formData }, (event, data) => {
      if (event === "progress") {
        const progressMessage = progressLabels[data.node] || "Generating your documents…";
        notify(progressMessage, "info");
        renderDocumentView(
          { title: "Generating documents" },
          streamedDocuments,
          progressMessage,
        );
      }
      if (event === "document") {
        streamedDocuments[data.type] = data.content;
        const nextStage = data.type === "prd"
          ? progressLabels.tech_doc_writer
          : "Finishing your documents…";
        renderDocumentView(
          { title: "Generating documents" },
          streamedDocuments,
          nextStage,
        );
      }
    });
    await openThreadById(state.activeThreadId);
    await loadThreads();
    notify(`Documents generated for thread ${result.session_id || state.activeThreadId}.`);
  } catch (error) {
    notify(error.message, "error");
    setBusy(generateButton, false, "Generate documents");
    canGenerate();
  }
});

document.querySelector("#account-button").addEventListener("click", () => userIdInput.focus());

if (userIdInput.value.trim()) {
  loadThreads();
}
