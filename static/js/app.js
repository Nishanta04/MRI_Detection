(() => {
  "use strict";

  const CLASS_COLORS = {
    glioma: "#fb7185",
    meningioma: "#fbbf24",
    notumor: "#34d399",
    pituitary: "#a78bfa",
  };
  const FALLBACK_COLORS = ["#fb7185", "#fbbf24", "#34d399", "#a78bfa", "#38bdf8", "#f472b6"];
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"];

  const form = document.getElementById("upload-form");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzoneEmpty = document.getElementById("dropzone-empty");
  const preview = document.getElementById("preview");
  const previewImage = document.getElementById("preview-image");
  const previewName = document.getElementById("preview-name");
  const previewSize = document.getElementById("preview-size");
  const removeButton = document.getElementById("remove-button");
  const analyzeButton = document.getElementById("analyze-button");
  const formError = document.getElementById("form-error");
  const status = document.getElementById("status");
  const statusText = document.getElementById("status-text");
  const results = document.getElementById("results");
  const confidenceRing = document.getElementById("confidence-ring");
  const confidenceValue = document.getElementById("confidence-value");
  const verdictClass = document.getElementById("verdict-class");
  const verdictNote = document.getElementById("verdict-note");
  const probabilityList = document.getElementById("probability-list");
  const resetButton = document.getElementById("reset-button");

  const maxUploadMb = Number(form.dataset.maxUploadMb) || 8;
  let selectedFile = null;
  let previewUrl = null;
  let isBusy = false;

  const formatBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const colorFor = (className, index) =>
    CLASS_COLORS[className] || FALLBACK_COLORS[index % FALLBACK_COLORS.length];

  const setStatus = (message, state = "") => {
    statusText.textContent = message;
    status.className = `status${state ? ` is-${state}` : ""}`;
  };

  const showError = (message) => {
    formError.textContent = message;
    formError.hidden = false;
  };

  const clearError = () => {
    formError.textContent = "";
    formError.hidden = true;
  };

  const revokePreview = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
  };

  const clearSelection = ({ focus = false } = {}) => {
    selectedFile = null;
    fileInput.value = "";
    revokePreview();
    previewImage.removeAttribute("src");
    preview.hidden = true;
    dropzoneEmpty.hidden = false;
    dropzone.classList.remove("has-image");
    analyzeButton.disabled = true;
    clearError();
    setStatus("Ready for an image");
    if (focus) dropzone.focus();
  };

  const validateFile = (file) => {
    if (!file) return "Please choose an image file.";
    const typeOk =
      ALLOWED_TYPES.includes(file.type) ||
      /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name);
    if (!typeOk) {
      return "Unsupported file type. Please upload a JPG, PNG, WEBP, BMP, or TIFF image.";
    }
    if (file.size > maxUploadMb * 1024 * 1024) {
      return `That image is ${formatBytes(file.size)}. Maximum size is ${maxUploadMb} MB.`;
    }
    if (file.size === 0) {
      return "That file is empty.";
    }
    return null;
  };

  const selectFile = (file) => {
    clearError();
    const error = validateFile(file);
    if (error) {
      showError(error);
      setStatus("Image not accepted", "error");
      return;
    }

    selectedFile = file;
    revokePreview();
    previewUrl = URL.createObjectURL(file);

    previewImage.onload = () => {
      preview.hidden = false;
      dropzoneEmpty.hidden = true;
      dropzone.classList.add("has-image");
    };
    previewImage.onerror = () => {
      showError("That image could not be displayed. Try a different file.");
      clearSelection();
      setStatus("Image could not be read", "error");
    };
    previewImage.src = previewUrl;
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    analyzeButton.disabled = false;
    setStatus("Image ready — run the analysis", "done");
    results.hidden = true;
  };

  const openPicker = () => {
    if (!isBusy && !selectedFile) fileInput.click();
  };

  /* ---------- Selection events ---------- */

  dropzone.addEventListener("click", (event) => {
    if (event.target.closest("#remove-button")) return;
    openPicker();
  });

  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) selectFile(fileInput.files[0]);
  });

  removeButton.addEventListener("click", (event) => {
    event.stopPropagation();
    if (!isBusy) clearSelection({ focus: true });
  });

  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!isBusy) dropzone.classList.add("is-dragging");
    });
  });

  ["dragleave", "dragend"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragging");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove("is-dragging");
    if (isBusy) return;
    const file = event.dataTransfer?.files?.[0];
    if (file) selectFile(file);
  });

  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  /* ---------- Results rendering ---------- */

  const animateConfidence = (targetPercent, color) => {
    confidenceRing.style.setProperty("--ring-color", color);
    const duration = 750;
    const start = performance.now();

    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = targetPercent * eased;
      confidenceRing.style.setProperty("--value", value.toFixed(2));
      confidenceValue.textContent = `${Math.round(value)}%`;
      if (progress < 1) requestAnimationFrame(step);
    };

    requestAnimationFrame(step);
  };

  const renderResults = (data) => {
    const sorted = [...data.probabilities].sort((a, b) => b.probability - a.probability);
    const topClass = data.prediction;
    const topColor = colorFor(topClass, 0);

    verdictClass.textContent = data.display_name;
    verdictClass.style.color = topColor;
    verdictNote.textContent =
      topClass === "notumor"
        ? "No tumor pattern is the most likely class for this scan."
        : `Features most consistent with ${data.display_name.toLowerCase()}.`;

    probabilityList.replaceChildren();
    sorted.forEach((item, index) => {
      const color = colorFor(item.class_name, index);
      const percent = item.probability * 100;

      const li = document.createElement("li");
      li.className = "probability-item";
      if (item.class_name === topClass) li.classList.add("is-top");
      li.style.setProperty("--bar-color", color);

      const name = document.createElement("span");
      name.className = "probability-name";
      name.textContent = item.display_name;
      name.title = item.display_name;

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.setAttribute("role", "progressbar");
      fill.setAttribute("aria-valuemin", "0");
      fill.setAttribute("aria-valuemax", "100");
      fill.setAttribute("aria-valuenow", percent.toFixed(1));
      fill.setAttribute("aria-label", `${item.display_name} probability`);
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "probability-value";
      value.textContent = `${percent.toFixed(1)}%`;

      li.append(name, track, value);
      probabilityList.appendChild(li);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          fill.style.width = `${Math.max(percent, 0.8)}%`;
        });
      });
    });

    results.hidden = false;
    animateConfidence(data.confidence * 100, topColor);
    results.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  /* ---------- Submit ---------- */

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (isBusy) return;

    if (!selectedFile) {
      showError("Please choose an MRI image first.");
      setStatus("No image selected", "error");
      return;
    }

    clearError();
    isBusy = true;
    analyzeButton.disabled = true;
    analyzeButton.classList.add("is-loading");
    analyzeButton.querySelector(".button-label").textContent = "Analyzing…";
    setStatus("Running model inference…", "busy");
    results.hidden = true;

    const body = new FormData();
    body.append("image", selectedFile, selectedFile.name);

    try {
      const response = await fetch("/predict", { method: "POST", body });
      let payload = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        throw new Error(payload?.error || `The server returned an error (${response.status}).`);
      }
      if (!payload || !Array.isArray(payload.probabilities)) {
        throw new Error("The server returned an unexpected response.");
      }

      renderResults(payload);
      setStatus(
        `Prediction: ${payload.display_name} (${(payload.confidence * 100).toFixed(1)}% confidence)`,
        "done"
      );
    } catch (error) {
      const message =
        error instanceof TypeError
          ? "Could not reach the server. Make sure the Flask app is running."
          : error.message;
      showError(message);
      setStatus("Analysis failed", "error");
    } finally {
      isBusy = false;
      analyzeButton.disabled = !selectedFile;
      analyzeButton.classList.remove("is-loading");
      analyzeButton.querySelector(".button-label").textContent = "Analyze scan";
    }
  });

  resetButton.addEventListener("click", () => {
    if (isBusy) return;
    clearSelection({ focus: true });
    results.hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  clearSelection();
})();
