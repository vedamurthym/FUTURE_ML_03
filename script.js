/**
 * Resume Screening & Candidate Matching System — Frontend Script
 *
 * Connects to the FastAPI backend at /api/* endpoints.
 * Manages the full workflow: analyze → edit skills → rank → inspect candidate.
 */

(function () {
  "use strict";

  // ========================================================================
  // CONFIGURATION
  // ========================================================================
  const API_BASE = window.location.origin;

  // ========================================================================
  // DOM REFERENCES
  // ========================================================================
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    // Nav
    navStatus: $("#nav-status"),
    statusDot: $("#nav-status .status-dot"),
    statusText: $("#nav-status .status-text"),
    navLinks: $$(".nav-link"),
    // Hero
    statTotal: $("#stat-total .stat-number"),
    statEligible: $("#stat-eligible .stat-number"),
    statCategories: $("#stat-categories .stat-number"),
    jdInput: $("#jd-input"),
    jdFileInput: $("#jd-file-input"),
    jdFileBadge: $("#jd-file-badge"),
    charCount: $("#char-count"),
    btnAnalyze: $("#btn-analyze"),
    inputError: $("#input-error"),
    // Sections
    inputSection: $("#input-section"),
    skillsSection: $("#skills-section"),
    resultsSection: $("#results-section"),
    candidateSection: $("#candidate-section"),
    // Skills
    analysisMessages: $("#analysis-messages"),
    skillsByCategory: $("#skills-by-category"),
    essentialSkillsList: $("#essential-skills-list"),
    optionalSkillsList: $("#optional-skills-list"),
    addEssentialInput: $("#add-essential-input"),
    addOptionalInput: $("#add-optional-input"),
    btnAddEssential: $("#btn-add-essential"),
    btnAddOptional: $("#btn-add-optional"),
    categoryFilter: $("#category-filter"),
    minScoreFilter: $("#min-score-filter"),
    topNFilter: $("#top-n-filter"),
    btnRank: $("#btn-rank"),
    // Loading
    loadingOverlay: $("#loading-overlay"),
    loadingText: $("#loading-text"),
    // Results
    pillEvaluated: $("#pill-evaluated"),
    pillMatched: $("#pill-matched"),
    pillReturned: $("#pill-returned"),
    resultsSummary: $("#results-summary"),
    resultsFailedBanner: $("#results-failed-banner"),
    resultsFilterBar: $("#results-filter-bar"),
    resultsCategoryFilter: $("#results-category-filter"),
    resultsMinScoreFilter: $("#results-min-score-filter"),
    resultsTopNFilter: $("#results-top-n-filter"),
    btnResetFilters: $("#btn-reset-filters"),
    chartContainer: $("#chart-container"),
    scoreChart: $("#score-chart"),
    resultsTbody: $("#results-tbody"),
    emptyResults: $("#empty-results"),
    // Candidate
    candidateDetailTitle: $("#candidate-detail-title"),
    btnBackResults: $("#btn-back-results"),
    detailId: $("#detail-id"),
    detailFilename: $("#detail-filename"),
    detailCategory: $("#detail-category"),
    detailEssCov: $("#detail-ess-cov"),
    detailOptCov: $("#detail-opt-cov"),
    detailExplanation: $("#detail-explanation"),
    detailEssMatched: $("#detail-ess-matched"),
    detailEssMissing: $("#detail-ess-missing"),
    detailOptMatched: $("#detail-opt-matched"),
    detailOptMissing: $("#detail-opt-missing"),
    candidateResumeCard: $("#candidate-resume-card"),
    detailAllSkills: $("#detail-all-skills"),
    detailSkillCount: $("#detail-skill-count"),
    detailWordCount: $("#detail-word-count"),
    detailTextPreview: $("#detail-text-preview"),
    resumePreviewDetails: $("#resume-preview-details"),
    // Resume Uploads
    uploadArea: $("#upload-area"),
    resumeDropzone: $("#resume-dropzone"),
    resumeFileInput: $("#resume-file-input"),
    resumeFolderInput: $("#resume-folder-input"),
    uploadQueue: $("#upload-queue"),
    queueCountLabel: $("#queue-count-label"),
    queueFileList: $("#queue-file-list"),
    btnAddMoreFiles: $("#btn-add-more-files"),
    btnAddFolder: $("#btn-add-folder"),
    btnSelectFolder: $("#btn-select-folder"),
    btnClearQueue: $("#btn-clear-queue"),
    btnScreenResumes: $("#btn-screen-resumes"),
    btnProcessResumes: $("#btn-process-resumes"),
    uploadError: $("#upload-error"),
    uploadResults: $("#upload-results"),
    pillUploadTotal: $("#pill-upload-total"),
    pillUploadValid: $("#pill-upload-valid"),
    pillUploadErrors: $("#pill-upload-errors"),
    processedCardsList: $("#processed-cards-list"),
  };

  // ========================================================================
  // STATE
  // ========================================================================
  let state = {
    jobDescription: "",
    uploadedJdFile: null,
    analysisResult: null,
    essentialSkills: [],
    optionalSkills: [],
    rankingResult: null,
    selectedCandidate: null,
    selectedResumeFiles: [],
    processedResumes: [],
    skillsConfirmedForJd: null,
  };

  // ========================================================================
  // UTILITY FUNCTIONS
  // ========================================================================

  async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function showLoading(text) {
    dom.loadingText.textContent = text;
    dom.loadingOverlay.style.display = "flex";
  }

  function hideLoading() {
    dom.loadingOverlay.style.display = "none";
  }

  function showError(el, msg) {
    el.textContent = msg;
    el.style.display = "block";
  }

  function hideError(el) {
    el.style.display = "none";
  }

  // ========================================================================
  // ROUTING & NAVIGATION
  // ========================================================================
  const ROUTE_MAP = {
    home: "input-section",
    "input-section": "input-section",
    skills: "skills-section",
    "skills-section": "skills-section",
    rankings: "results-section",
    ranking: "results-section",
    results: "results-section",
    "results-section": "results-section",
    candidate: "candidate-section",
    candidates: "candidate-section",
    "candidate-section": "candidate-section",
  };

  const SECTION_TO_ROUTE = {
    "input-section": "home",
    "skills-section": "skills",
    "results-section": "rankings",
    "candidate-section": "candidate",
  };

  function prepareSkillsView() {
    if (state.analysisResult) {
      renderSkillsSection(state.analysisResult);
    } else {
      // Graceful view when no job description has been analyzed yet
      dom.analysisMessages.innerHTML = `
        <div class="validation-msg">
          ℹ️ No job description has been analyzed yet. You can paste a job description on the 
          <a href="#home" class="inline-link">Home page</a> to extract skills automatically, 
          or manually add required skills below.
        </div>
      `;
      if (!dom.skillsByCategory.innerHTML.trim() || dom.skillsByCategory.innerHTML.includes("No skills")) {
        dom.skillsByCategory.innerHTML =
          '<p style="color:var(--text-muted);font-size:0.85rem;font-style:italic;">No extracted skills yet. Paste a job description on the Home page to extract skills.</p>';
      }
      renderSkillTags("essential");
      renderSkillTags("optional");
    }
  }

  function prepareRankingsView() {
    if (state.rankingResult && state.rankingResult.ranked_candidates && state.rankingResult.ranked_candidates.length > 0) {
      renderResults(state.rankingResult);
      requestAnimationFrame(() => {
        drawScoreChart(state.rankingResult.ranked_candidates);
      });
    } else {
      // Graceful empty state when no candidates have been ranked yet
      dom.pillEvaluated.innerHTML = `<strong>0</strong> Evaluated`;
      dom.pillMatched.innerHTML = `<strong>0</strong> Matched`;
      dom.pillReturned.innerHTML = `<strong>0</strong> Returned`;
      dom.resultsSummary.textContent = "No candidate rankings available yet. Enter a job description and run ranking to view results.";
      dom.chartContainer.style.display = "none";
      const tableWrapper = document.querySelector(".table-wrapper");
      if (tableWrapper) tableWrapper.style.display = "none";
      dom.emptyResults.style.display = "block";
      const emptyTitle = dom.emptyResults.querySelector(".empty-title");
      const emptyDesc = dom.emptyResults.querySelector(".empty-desc");
      if (emptyTitle) emptyTitle.textContent = "No Candidates Ranked Yet";
      if (emptyDesc) {
        emptyDesc.innerHTML = 'Paste a job description on the <a href="#home" class="inline-link">Home page</a>, confirm skills, and click <strong>Analyze &amp; Rank Candidates</strong> to view candidate match scores here.';
      }
    }
  }

  function navigateTo(target, updateHash = true) {
    const rawTarget = (target || "home").toString().toLowerCase().replace(/^#\/?/, "");
    const sectionId = ROUTE_MAP[rawTarget] || "input-section";
    const sectionEl = document.getElementById(sectionId) || dom.inputSection;
    const route = SECTION_TO_ROUTE[sectionEl.id] || "home";

    // Toggle display of all sections FIRST so active section has layout geometry
    [dom.inputSection, dom.skillsSection, dom.resultsSection, dom.candidateSection].forEach((s) => {
      if (s) {
        s.style.display = (s.id === sectionEl.id) ? "block" : "none";
      }
    });

    // Prepare content for specific pages
    if (sectionEl.id === "skills-section") {
      prepareSkillsView();
    } else if (sectionEl.id === "results-section") {
      prepareRankingsView();
    } else if (sectionEl.id === "candidate-section") {
      if (!state.selectedCandidate) {
        if (state.rankingResult && state.rankingResult.ranked_candidates && state.rankingResult.ranked_candidates.length > 0) {
          viewCandidate(state.rankingResult.ranked_candidates[0]);
          return;
        } else {
          navigateTo("rankings", updateHash);
          return;
        }
      }
    }

    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Update active nav links
    dom.navLinks.forEach((link) => {
      const isActive = (link.dataset.section === sectionEl.id);
      link.classList.toggle("active", isActive);
    });

    // Update URL hash
    if (updateHash && window.location.hash !== "#" + route) {
      window.location.hash = route;
    }
  }

  function showSection(sectionEl) {
    const route = SECTION_TO_ROUTE[sectionEl.id] || "home";
    navigateTo(route, true);
  }

  function formatScore(val) {
    if (val === null || val === undefined) return "N/A";
    return Number(val).toFixed(1) + "%";
  }

  // ========================================================================
  // INITIALIZATION — Health Check & Categories
  // ========================================================================

  async function init() {
    // Health check
    try {
      const health = await apiFetch("/api/health");
      dom.statusDot.classList.add("online");
      dom.statusText.textContent = "System Online";
    } catch (e) {
      dom.statusDot.classList.add("offline");
      dom.statusText.textContent = "Offline";
    }

    // Categories
    try {
      const cats = await apiFetch("/api/categories");
      const catNames = (cats.categories || []).map((c) => c.name);
      updateCategoryFilters(catNames);
    } catch (e) {
      // ignore
    }

    // Initialize handlers
    setupUploadHandlers();
    setupJdUploadHandler();
    setupLiveFilterHandlers();
  }

  // ========================================================================
  // JOB DESCRIPTION INPUT
  // ========================================================================

  dom.jdInput.addEventListener("input", () => {
    const text = dom.jdInput.value.trim();
    const len = text.length;
    dom.charCount.textContent = `${len} characters`;
    dom.btnAnalyze.disabled = len < 10;
    hideError(dom.inputError);

    // If the JD content changed, invalidate stale state from previous jobs
    if (text !== state.jobDescription) {
      state.jobDescription = text;
      state.uploadedJdFile = null;
      if (dom.jdFileBadge) dom.jdFileBadge.style.display = "none";
      state.essentialSkills = [];
      state.optionalSkills = [];
      state.analysisResult = null;
      state.skillsConfirmedForJd = null;
    }
  });

  dom.btnAnalyze.addEventListener("click", analyzeJob);

  async function analyzeJob() {
    const text = dom.jdInput.value.trim();
    if (text.length < 10) {
      showError(dom.inputError, "Please enter at least 10 characters.");
      return;
    }

    state.jobDescription = text;
    state.uploadedJdFile = null;
    showLoading("Analyzing job description...");
    hideError(dom.inputError);

    try {
      const result = await apiFetch("/api/analyze-job", {
        method: "POST",
        body: JSON.stringify({ job_description: text }),
      });

      state.analysisResult = result;
      state.essentialSkills = [...(result.essential_skills || [])];
      state.optionalSkills = [...(result.optional_skills || [])];
      state.skillsConfirmedForJd = text;

      renderSkillsSection(result);
      hideLoading();
      showSection(dom.skillsSection);
    } catch (e) {
      hideLoading();
      showError(dom.inputError, e.message);
    }
  }

  // ========================================================================
  // RESUME UPLOADS (MULTIPLE PDF & DOCX)
  // ========================================================================
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB per file
  const ALLOWED_EXTS = [".pdf", ".docx"];

  /**
   * Recursively extract File objects from FileSystemEntry items (supports dropped folders).
   */
  async function extractFilesFromEntries(entries) {
    const files = [];

    async function readEntry(entry) {
      if (entry.isFile) {
        return new Promise((resolve) => {
          entry.file((f) => {
            files.push(f);
            resolve();
          }, () => resolve());
        });
      } else if (entry.isDirectory) {
        const dirReader = entry.createReader();
        let batch;
        do {
          batch = await new Promise((resolve) => {
            dirReader.readEntries((results) => resolve(results), () => resolve([]));
          });
          for (const child of batch) {
            await readEntry(child);
          }
        } while (batch.length > 0);
      }
    }

    for (const entry of entries) {
      await readEntry(entry);
    }
    return files;
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function handleFilesAdded(fileList) {
    hideError(dom.uploadError);
    if (!fileList || fileList.length === 0) return;

    const errors = [];
    let addedCount = 0;

    Array.from(fileList).forEach((file) => {
      const ext = "." + file.name.split(".").pop().toLowerCase();
      // 1. Check extension
      if (!ALLOWED_EXTS.includes(ext)) {
        errors.push(`"${file.name}": Unsupported format. Only PDF and DOCX files are accepted.`);
        return;
      }
      // 2. Check size
      if (file.size > MAX_FILE_SIZE) {
        errors.push(`"${file.name}": Exceeds the 10 MB limit (${formatBytes(file.size)}).`);
        return;
      }
      // 3. Check empty
      if (file.size === 0) {
        errors.push(`"${file.name}": File is empty (0 bytes).`);
        return;
      }
      // 4. Duplicate prevention (same name & size)
      const isDuplicate = state.selectedResumeFiles.some(
        (f) => f.name === file.name && f.size === file.size
      );
      if (isDuplicate) {
        return;
      }

      state.selectedResumeFiles.push(file);
      addedCount++;
    });

    if (errors.length > 0) {
      showError(dom.uploadError, errors.join(" • "));
    }

    renderQueue();
  }

  /**
   * Filter a FileList from a folder input, keeping only PDF/DOCX files.
   */
  function handleFolderFilesAdded(fileList) {
    if (!fileList || fileList.length === 0) return;
    const filtered = Array.from(fileList).filter((file) => {
      const ext = "." + file.name.split(".").pop().toLowerCase();
      return ALLOWED_EXTS.includes(ext);
    });
    if (filtered.length === 0) {
      showError(dom.uploadError, "No PDF or DOCX files found in the selected folder.");
      return;
    }
    // Reuse the standard handler which validates size, duplicates, etc.
    handleFilesAdded(filtered);
  }

  function renderQueue() {
    const list = state.selectedResumeFiles;
    if (list.length === 0) {
      dom.uploadQueue.style.display = "none";
      dom.queueFileList.innerHTML = "";
      return;
    }

    dom.uploadQueue.style.display = "block";
    dom.queueCountLabel.textContent = `Selected Resumes (${list.length})`;
    dom.queueFileList.innerHTML = "";

    list.forEach((file, index) => {
      const ext = file.name.split(".").pop().toUpperCase();
      const li = document.createElement("li");
      li.className = "queue-item";
      li.innerHTML = `
        <div class="queue-item-left">
          <span class="queue-item-badge queue-item-badge--${ext.toLowerCase()}">${ext}</span>
          <span class="queue-item-name" title="${file.name}">${file.name}</span>
          <span class="queue-item-size">(${formatBytes(file.size)})</span>
        </div>
        <button type="button" class="queue-item-remove" data-index="${index}" title="Remove file">&times;</button>
      `;
      dom.queueFileList.appendChild(li);
    });

    // Remove buttons
    dom.queueFileList.querySelectorAll(".queue-item-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const idx = parseInt(e.currentTarget.dataset.index, 10);
        state.selectedResumeFiles.splice(idx, 1);
        renderQueue();
      });
    });
  }

  function setupUploadHandlers() {
    if (!dom.resumeDropzone || !dom.resumeFileInput) return;

    // Dropzone click -> file input
    dom.resumeDropzone.addEventListener("click", () => {
      dom.resumeFileInput.click();
    });

    // File input change
    dom.resumeFileInput.addEventListener("change", (e) => {
      handleFilesAdded(e.target.files);
      dom.resumeFileInput.value = "";
    });

    // Add more files button
    dom.btnAddMoreFiles.addEventListener("click", () => {
      dom.resumeFileInput.click();
    });

    // Select folder button (in dropzone)
    dom.btnSelectFolder.addEventListener("click", (e) => {
      e.stopPropagation(); // Don't trigger the dropzone file picker
      dom.resumeFolderInput.click();
    });

    // Folder input change
    dom.resumeFolderInput.addEventListener("change", (e) => {
      handleFolderFilesAdded(e.target.files);
      dom.resumeFolderInput.value = "";
    });

    // Add folder button (in queue header)
    if (dom.btnAddFolder) {
      dom.btnAddFolder.addEventListener("click", () => {
        dom.resumeFolderInput.click();
      });
    }

    // Clear queue button
    dom.btnClearQueue.addEventListener("click", () => {
      state.selectedResumeFiles = [];
      hideError(dom.uploadError);
      renderQueue();
    });

    // Drag & drop events
    dom.resumeDropzone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dom.resumeDropzone.classList.add("drag-over");
    });
    dom.resumeDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dom.resumeDropzone.classList.add("drag-over");
    });
    dom.resumeDropzone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dom.resumeDropzone.classList.remove("drag-over");
    });
    dom.resumeDropzone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dom.resumeDropzone.classList.remove("drag-over");

      // Check if a folder was dropped (using DataTransfer items API)
      if (e.dataTransfer && e.dataTransfer.items) {
        const entries = [];
        for (const item of e.dataTransfer.items) {
          const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
          if (entry) entries.push(entry);
        }

        // If any entry is a directory, recursively extract files
        const hasDirectory = entries.some((entry) => entry.isDirectory);
        if (hasDirectory) {
          const allFiles = await extractFilesFromEntries(entries);
          handleFolderFilesAdded(allFiles);
          return;
        }
      }

      // Standard file drop
      if (e.dataTransfer && e.dataTransfer.files) {
        handleFilesAdded(e.dataTransfer.files);
      }
    });

    // Screen Resumes button
    if (dom.btnScreenResumes) {
      dom.btnScreenResumes.addEventListener("click", screenUploadedResumes);
    }
    if (dom.btnProcessResumes) {
      dom.btnProcessResumes.addEventListener("click", screenUploadedResumes);
    }
  }

  function setupJdUploadHandler() {
    if (!dom.jdFileInput) return;

    dom.jdFileInput.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;

      const ext = "." + file.name.split(".").pop().toLowerCase();
      if (ext !== ".txt" && ext !== ".pdf") {
        showError(dom.inputError, `"${file.name}": Unsupported format. Only .txt and .pdf files are supported for job descriptions.`);
        dom.jdFileInput.value = "";
        return;
      }

      state.uploadedJdFile = file;
      state.jobDescription = "";
      state.essentialSkills = [];
      state.optionalSkills = [];
      state.analysisResult = null;
      state.skillsConfirmedForJd = null;
      hideError(dom.inputError);

      if (dom.jdFileBadge) {
        dom.jdFileBadge.style.display = "inline-block";
        dom.jdFileBadge.textContent = `📄 ${file.name}`;
      }

      if (ext === ".txt") {
        const reader = new FileReader();
        reader.onload = (evt) => {
          dom.jdInput.value = evt.target.result;
          state.jobDescription = evt.target.result;
          dom.charCount.textContent = `${evt.target.result.length} characters`;
        };
        reader.readAsText(file);
      } else {
        if (!dom.jdInput.value.trim() || dom.jdInput.value.startsWith("[Uploaded Job Description")) {
          dom.jdInput.value = `[Uploaded Job Description PDF: ${file.name}]`;
          dom.charCount.textContent = `${dom.jdInput.value.length} characters`;
        }
      }
    });
  }

  async function screenUploadedResumes() {
    const jdText = dom.jdInput.value.trim();
    const jdFile = state.uploadedJdFile;

    hideError(dom.inputError);
    hideError(dom.uploadError);

    // 1. Validate Job Description
    if (!jdText && !jdFile) {
      showError(dom.inputError, "Please enter or upload a job description (TXT or PDF).");
      dom.jdInput.focus();
      return;
    }

    // 2. Validate Resumes
    if (state.selectedResumeFiles.length === 0) {
      showError(dom.uploadError, "Please upload at least one candidate resume (PDF or DOCX).");
      return;
    }

    showLoading(`Screening ${state.selectedResumeFiles.length} candidate resume(s) against job description...`);

    const formData = new FormData();
    if (jdFile) {
      formData.append("jd_file", jdFile, jdFile.name);
    }
    if (jdText) {
      formData.append("job_description", jdText);
    }

    // Only attach recruiter-confirmed skills if they belong to THIS current JD
    const currentJdKey = jdFile ? jdFile.name : jdText;
    const isSkillsConfirmedForCurrentJd = Boolean(
      state.skillsConfirmedForJd && state.skillsConfirmedForJd === currentJdKey
    );

    if (isSkillsConfirmedForCurrentJd) {
      if (state.essentialSkills && state.essentialSkills.length > 0) {
        formData.append("essential_skills", JSON.stringify(state.essentialSkills));
      }
      if (state.optionalSkills && state.optionalSkills.length > 0) {
        formData.append("optional_skills", JSON.stringify(state.optionalSkills));
      }
    }

    state.selectedResumeFiles.forEach((file) => {
      formData.append("files", file, file.name);
    });

    try {
      const res = await fetch(`${API_BASE}/api/screen`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Screening failed with status ${res.status}`);
      }

      const result = await res.json();
      state.rankingResult = result;
      state.jobDescription = jdText;
      state.skillsConfirmedForJd = currentJdKey;

      if (result.job_analysis) {
        state.essentialSkills = [...(result.job_analysis.essential_skills || [])];
        state.optionalSkills = [...(result.job_analysis.optional_skills || [])];
        state.analysisResult = result.job_analysis;
      }

      // Show results view first so canvas and containers are in the active layout
      showSection(dom.resultsSection);
      renderResults(result);
      hideLoading();
    } catch (e) {
      hideLoading();
      showError(dom.uploadError, e.message);
    }
  }

  async function processUploadedResumes() {
    if (state.selectedResumeFiles.length === 0) {
      showError(dom.uploadError, "Please select at least one PDF or DOCX resume to process.");
      return;
    }

    hideError(dom.uploadError);
    showLoading(`Processing ${state.selectedResumeFiles.length} resume(s) in-memory...`);

    const formData = new FormData();
    state.selectedResumeFiles.forEach((file) => {
      formData.append("files", file, file.name);
    });

    try {
      const res = await fetch(`${API_BASE}/api/upload-resumes`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      state.processedResumes = data.files || [];
      renderProcessedUploads(data);
      hideLoading();

      // Clear the selection queue after successful upload
      state.selectedResumeFiles = [];
      renderQueue();

      dom.uploadResults.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      hideLoading();
      showError(dom.uploadError, `Processing failed: ${e.message}`);
    }
  }

  function renderProcessedUploads(data) {
    dom.uploadResults.style.display = "block";
    dom.pillUploadTotal.innerHTML = `<strong>${data.total_files}</strong> Total`;
    dom.pillUploadValid.innerHTML = `<strong>${data.valid_count}</strong> Valid`;
    dom.pillUploadErrors.innerHTML = `<strong>${data.error_count}</strong> Failed`;

    dom.processedCardsList.innerHTML = "";

    (data.files || []).forEach((f, idx) => {
      const card = document.createElement("div");
      card.className = `processed-card ${f.is_valid ? "processed-card--valid" : "processed-card--error"}`;

      if (f.is_valid) {
        const skills = f.extracted_skills || [];
        const skillPillsHtml = skills.length > 0
          ? skills.map((s) => `<span class="skill-pill skill-pill--matched">${s}</span>`).join(" ")
          : '<span class="skill-pill skill-pill--none">No skills detected</span>';

        card.innerHTML = `
          <div class="processed-card-header">
            <span class="processed-card-title">
              <span class="queue-item-badge queue-item-badge--${f.file_type.toLowerCase()}">${f.file_type}</span>
              ${f.filename}
            </span>
            <span class="processed-status-badge processed-status-badge--success">✓ Extracted</span>
          </div>
          <div class="processed-card-meta">
            <span>Size: <strong>${f.file_size_formatted}</strong></span>
            <span>Words: <strong>${f.word_count}</strong></span>
            <span>Characters: <strong>${f.char_count}</strong></span>
            <span>Detected Skills: <strong>${f.skill_count}</strong></span>
          </div>
          <div class="processed-skills-wrap">
            ${skillPillsHtml}
          </div>
          <button type="button" class="processed-preview-toggle" data-idx="${idx}">Preview Raw Text &darr;</button>
          <div class="processed-preview-box" id="preview-box-${idx}" style="display:none;">${f.text_preview || "No text available"}</div>
        `;
      } else {
        card.innerHTML = `
          <div class="processed-card-header">
            <span class="processed-card-title">
              <span class="queue-item-badge queue-item-badge--pdf">${f.file_type || "FILE"}</span>
              ${f.filename}
            </span>
            <span class="processed-status-badge processed-status-badge--fail">✗ Failed</span>
          </div>
          <div class="processed-card-meta">
            <span>Size: <strong>${f.file_size_formatted}</strong></span>
          </div>
          <p class="processed-error-msg">⚠️ ${f.error || "File validation failed."}</p>
        `;
      }

      dom.processedCardsList.appendChild(card);
    });

    // Toggle preview boxes
    dom.processedCardsList.querySelectorAll(".processed-preview-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = btn.dataset.idx;
        const box = document.getElementById(`preview-box-${idx}`);
        if (box) {
          const isHidden = box.style.display === "none";
          box.style.display = isHidden ? "block" : "none";
          btn.innerHTML = isHidden ? "Hide Text Preview &uarr;" : "Preview Raw Text &darr;";
        }
      });
    });
  }

  // ========================================================================
  // SKILLS EDITING
  // ========================================================================

  function renderSkillsSection(analysis) {
    // Validation messages
    dom.analysisMessages.innerHTML = "";
    (analysis.validation_messages || []).forEach((msg) => {
      const div = document.createElement("div");
      div.className = "validation-msg";
      div.textContent = msg;
      dom.analysisMessages.appendChild(div);
    });

    // All extracted skills by category
    dom.skillsByCategory.innerHTML = "";
    const byCat = analysis.by_category || {};
    const catNames = Object.keys(byCat).sort();
    if (catNames.length === 0) {
      dom.skillsByCategory.innerHTML =
        '<p style="color:var(--text-muted);font-size:0.85rem;font-style:italic;">No skills detected.</p>';
    }
    catNames.forEach((cat) => {
      const h4 = document.createElement("h4");
      h4.textContent = cat;
      dom.skillsByCategory.appendChild(h4);
      const wrap = document.createElement("div");
      wrap.className = "skills-tag-list";
      byCat[cat].forEach((skill) => {
        const tag = document.createElement("span");
        tag.className = "skill-tag";
        tag.textContent = skill;
        wrap.appendChild(tag);
      });
      dom.skillsByCategory.appendChild(wrap);
    });

    // Essential & optional
    renderSkillTags("essential");
    renderSkillTags("optional");
  }

  function renderSkillTags(type) {
    const list = type === "essential" ? state.essentialSkills : state.optionalSkills;
    const container =
      type === "essential" ? dom.essentialSkillsList : dom.optionalSkillsList;
    container.innerHTML = "";

    if (list.length === 0) {
      container.innerHTML =
        '<span class="skill-tag" style="opacity:0.5;font-style:italic;">None</span>';
      return;
    }

    list.forEach((skill) => {
      const tag = document.createElement("span");
      tag.className = "skill-tag";
      tag.innerHTML = `${skill} <span class="tag-remove" title="Remove">×</span>`;
      tag.querySelector(".tag-remove").addEventListener("click", () => {
        removeSkill(type, skill);
      });
      container.appendChild(tag);
    });
  }

  function removeSkill(type, skill) {
    if (type === "essential") {
      state.essentialSkills = state.essentialSkills.filter((s) => s !== skill);
    } else {
      state.optionalSkills = state.optionalSkills.filter((s) => s !== skill);
    }
    const currentJd = (dom.jdInput && dom.jdInput.value.trim()) || state.jobDescription;
    state.skillsConfirmedForJd = state.uploadedJdFile ? state.uploadedJdFile.name : currentJd;
    renderSkillTags(type);
  }

  function addSkill(type) {
    const input =
      type === "essential" ? dom.addEssentialInput : dom.addOptionalInput;
    const val = input.value.trim();
    if (!val) return;

    const list = type === "essential" ? state.essentialSkills : state.optionalSkills;
    if (!list.includes(val)) {
      list.push(val);
      const currentJd = (dom.jdInput && dom.jdInput.value.trim()) || state.jobDescription;
      state.skillsConfirmedForJd = state.uploadedJdFile ? state.uploadedJdFile.name : currentJd;
      renderSkillTags(type);
    }
    input.value = "";
  }

  dom.btnAddEssential.addEventListener("click", () => addSkill("essential"));
  dom.btnAddOptional.addEventListener("click", () => addSkill("optional"));

  dom.addEssentialInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") addSkill("essential");
  });
  dom.addOptionalInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") addSkill("optional");
  });

  // ========================================================================
  // RANKING
  // ========================================================================

  dom.btnRank.addEventListener("click", rankCandidates);

  async function rankCandidates() {
    if (state.selectedResumeFiles.length > 0) {
      return screenUploadedResumes();
    }

    const currentJd = (dom.jdInput && dom.jdInput.value.trim()) || state.jobDescription;
    const currentJdKey = state.uploadedJdFile ? state.uploadedJdFile.name : currentJd;
    const isSkillsConfirmedForCurrentJd = Boolean(
      state.skillsConfirmedForJd && state.skillsConfirmedForJd === currentJdKey
    );

    showLoading("Ranking uploaded candidates with TF-IDF similarity & skill matching...");

    const payload = {
      job_description: currentJd,
      essential_skills:
        isSkillsConfirmedForCurrentJd && state.essentialSkills.length > 0 ? state.essentialSkills : null,
      optional_skills:
        isSkillsConfirmedForCurrentJd && state.optionalSkills.length > 0 ? state.optionalSkills : null,
      category_filter: dom.categoryFilter.value,
      min_score: parseFloat(dom.minScoreFilter.value) || 0,
      top_n: parseInt(dom.topNFilter.value, 10) || null,
    };

    try {
      const result = await apiFetch("/api/rank", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      state.rankingResult = result;
      showSection(dom.resultsSection);
      renderResults(result);
      hideLoading();
    } catch (e) {
      hideLoading();
      showError(dom.inputError, e.message);
      showSection(dom.skillsSection);
    }
  }

  // ========================================================================
  // RESULTS RENDERING
  // ========================================================================

  // ========================================================================
  // CATEGORIES & DYNAMIC FILTER OPTIONS
  // ========================================================================

  function generateMaxResultsOptions(count) {
    if (!count || count <= 0) {
      return '<option value="">All</option>';
    }
    if (count === 1) {
      return '<option value="" selected>All (1)</option>';
    }

    let steps = [];
    if (count <= 3) {
      steps = [1, 2];
    } else if (count <= 5) {
      steps = [2, 3];
    } else if (count <= 8) {
      steps = [3, 5];
    } else if (count <= 12) {
      steps = [3, 5, 10];
    } else if (count <= 25) {
      steps = [5, 10, 15];
    } else {
      steps = [5, 10, 20, 50];
    }

    steps = steps.filter((s) => s < count);

    let html = `<option value="" selected>All (${count})</option>`;
    steps.forEach((s) => {
      html += `<option value="${s}">Top ${s}</option>`;
    });
    return html;
  }

  function updateCategoryFilters(categories) {
    const catList = Array.isArray(categories) ? categories : [];
    let optionsHtml = '<option value="">All Categories</option>';
    catList.forEach((cat) => {
      const name = typeof cat === "string" ? cat : (cat && (cat.name || cat.category)) || "";
      if (name) {
        optionsHtml += `<option value="${name}">${name}</option>`;
      }
    });

    if (dom.categoryFilter) {
      const curVal = dom.categoryFilter.value;
      dom.categoryFilter.innerHTML = optionsHtml;
      if (curVal && catList.includes(curVal)) {
        dom.categoryFilter.value = curVal;
      }
    }

    if (dom.resultsCategoryFilter) {
      const curVal = dom.resultsCategoryFilter.value;
      dom.resultsCategoryFilter.innerHTML = optionsHtml;
      if (curVal && catList.includes(curVal)) {
        dom.resultsCategoryFilter.value = curVal;
      }
    }
  }

  function updateMaxResultsOptions(count) {
    const optionsHtml = generateMaxResultsOptions(count);
    if (dom.topNFilter) {
      dom.topNFilter.innerHTML = optionsHtml;
    }
    if (dom.resultsTopNFilter) {
      dom.resultsTopNFilter.innerHTML = optionsHtml;
    }
  }

  function setupLiveFilterHandlers() {
    if (dom.resultsCategoryFilter) {
      dom.resultsCategoryFilter.addEventListener("change", () => {
        if (dom.categoryFilter) dom.categoryFilter.value = dom.resultsCategoryFilter.value;
        applyLiveFilters();
      });
    }
    if (dom.categoryFilter) {
      dom.categoryFilter.addEventListener("change", () => {
        if (dom.resultsCategoryFilter) dom.resultsCategoryFilter.value = dom.categoryFilter.value;
        applyLiveFilters();
      });
    }

    if (dom.resultsMinScoreFilter) {
      dom.resultsMinScoreFilter.addEventListener("input", () => {
        if (dom.minScoreFilter) dom.minScoreFilter.value = dom.resultsMinScoreFilter.value;
        applyLiveFilters();
      });
    }
    if (dom.minScoreFilter) {
      dom.minScoreFilter.addEventListener("input", () => {
        if (dom.resultsMinScoreFilter) dom.resultsMinScoreFilter.value = dom.minScoreFilter.value;
        applyLiveFilters();
      });
    }

    if (dom.resultsTopNFilter) {
      dom.resultsTopNFilter.addEventListener("change", () => {
        if (dom.topNFilter) dom.topNFilter.value = dom.resultsTopNFilter.value;
        applyLiveFilters();
      });
    }
    if (dom.topNFilter) {
      dom.topNFilter.addEventListener("change", () => {
        if (dom.resultsTopNFilter) dom.resultsTopNFilter.value = dom.topNFilter.value;
        applyLiveFilters();
      });
    }

    if (dom.btnResetFilters) {
      dom.btnResetFilters.addEventListener("click", () => {
        if (dom.resultsCategoryFilter) dom.resultsCategoryFilter.value = "";
        if (dom.categoryFilter) dom.categoryFilter.value = "";
        if (dom.resultsMinScoreFilter) dom.resultsMinScoreFilter.value = "";
        if (dom.minScoreFilter) dom.minScoreFilter.value = "";
        if (dom.resultsTopNFilter) dom.resultsTopNFilter.value = "";
        if (dom.topNFilter) dom.topNFilter.value = "";
        applyLiveFilters();
      });
    }
  }

  function applyLiveFilters() {
    if (!state.rankingResult || !state.rankingResult.ranked_candidates) return;

    const allCandidates = state.rankingResult.ranked_candidates;
    const selectedCat = (dom.resultsCategoryFilter && dom.resultsCategoryFilter.value) || 
                        (dom.categoryFilter && dom.categoryFilter.value) || "";
    const rawMin = (dom.resultsMinScoreFilter && dom.resultsMinScoreFilter.value) ||
                   (dom.minScoreFilter && dom.minScoreFilter.value) || "";
    const minScore = parseFloat(rawMin);
    const rawTopN = (dom.resultsTopNFilter && dom.resultsTopNFilter.value) ||
                    (dom.topNFilter && dom.topNFilter.value) || "";
    const topN = rawTopN ? parseInt(rawTopN, 10) : null;

    let filtered = allCandidates.filter((c) => {
      if (selectedCat && selectedCat !== "") {
        if (c.category !== selectedCat) return false;
      }
      if (!isNaN(minScore) && minScore > 0) {
        const score = typeof c.combined_score === "number" ? c.combined_score : parseFloat(c.combined_score) || 0;
        if (score < minScore) return false;
      }
      return true;
    });

    const matchedCount = filtered.length;

    if (topN && !isNaN(topN) && topN > 0) {
      filtered = filtered.slice(0, topN);
    }

    const totalEval = state.rankingResult.total_evaluated || allCandidates.length;
    dom.pillEvaluated.innerHTML = `<strong>${totalEval.toLocaleString()}</strong> Evaluated`;
    dom.pillMatched.innerHTML = `<strong>${matchedCount.toLocaleString()}</strong> Matched`;
    dom.pillReturned.innerHTML = `<strong>${filtered.length.toLocaleString()}</strong> Returned`;

    if (filtered.length === 0) {
      dom.emptyResults.style.display = "block";
      const wrapper = document.querySelector(".table-wrapper");
      if (wrapper) wrapper.style.display = "none";
    } else {
      dom.emptyResults.style.display = "none";
      const wrapper = document.querySelector(".table-wrapper");
      if (wrapper) wrapper.style.display = "block";
    }

    renderTableRows(filtered);
    drawScoreChart(filtered.length > 0 ? filtered : allCandidates);
  }

  function renderTableRows(candidates) {
    dom.resultsTbody.innerHTML = "";
    const jdHasEssential = state.essentialSkills && state.essentialSkills.length > 0;
    const jdHasOptional = state.optionalSkills && state.optionalSkills.length > 0;

    candidates.forEach((c, i) => {
      const tr = document.createElement("tr");
      const filename = c.filename || c.candidate_id || `Candidate #${i + 1}`;
      const ext = filename.includes(".") ? filename.split(".").pop().toUpperCase() : "RESUME";

      // Essential Matched
      let essMatched;
      const matchedEssList = c.essential_skills_matched || c.matched_essential || [];
      if (!jdHasEssential) {
        essMatched = '<span class="skill-pill skill-pill--missing" title="No essential skills identified in JD">Check job description</span>';
      } else if (matchedEssList.length > 0) {
        essMatched = matchedEssList
          .map((s) => `<span class="skill-pill skill-pill--matched">${s}</span>`)
          .join(" ");
      } else {
        essMatched = '<span class="skill-pill skill-pill--none">None</span>';
      }

      // Essential Missing
      let essMissing;
      const missingEssList = c.essential_skills_missing || c.missing_essential || [];
      if (!jdHasEssential) {
        essMissing = '<span class="skill-pill skill-pill--none">—</span>';
      } else if (missingEssList.length > 0) {
        essMissing = missingEssList
          .map((s) => `<span class="skill-pill skill-pill--missing">${s}</span>`)
          .join(" ");
      } else {
        essMissing = '<span class="skill-pill skill-pill--matched">None (All Matched)</span>';
      }

      // Optional Matched
      let optMatched;
      const matchedOptList = c.optional_skills_matched || c.matched_optional || [];
      if (!jdHasOptional) {
        optMatched = '<span class="skill-pill skill-pill--none" style="font-style:italic;">No optional skills specified.</span>';
      } else if (matchedOptList.length > 0) {
        optMatched = matchedOptList
          .map((s) => `<span class="skill-pill skill-pill--matched">${s}</span>`)
          .join(" ");
      } else {
        optMatched = '<span class="skill-pill skill-pill--none" style="font-style:italic;">No optional skills matched.</span>';
      }

      const explanation = c.match_explanation || "No explanation available.";
      const shortExplanation = explanation.length > 130 ? explanation.slice(0, 130) + "..." : explanation;

      tr.innerHTML = `
        <td class="rank-cell">#${c.candidate_rank || i + 1}</td>
        <td>
          <div style="display:flex; flex-direction:column; gap:2px;">
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="queue-item-badge queue-item-badge--${ext.toLowerCase()}">${ext}</span>
              <span style="font-weight:600; color:var(--crimson-red);">${filename}</span>
            </div>
            <span style="font-size:0.75rem; color:var(--text-muted); padding-left:2px;">${c.category || "Other / Unclassified"}</span>
          </div>
        </td>
        <td>
          <div class="score-bar">
            <span class="score-value">${formatScore(c.combined_score)}</span>
            <div class="score-bar-track"><div class="score-bar-fill" style="width:${c.combined_score}%"></div></div>
          </div>
        </td>
        <td>${formatScore(c.skill_match_score)}</td>
        <td>${formatScore(c.text_similarity_score)}</td>
        <td><div style="display:flex; flex-wrap:wrap; gap:3px;">${essMatched}</div></td>
        <td><div style="display:flex; flex-wrap:wrap; gap:3px;">${essMissing}</div></td>
        <td><div style="display:flex; flex-wrap:wrap; gap:3px;">${optMatched}</div></td>
        <td style="max-width:240px; font-size:0.8rem; color:var(--text-secondary); line-height:1.4;" title="${explanation}">${shortExplanation}</td>
        <td><button class="btn-view" data-idx="${i}">View</button></td>
      `;
      dom.resultsTbody.appendChild(tr);
    });

    dom.resultsTbody.querySelectorAll(".btn-view").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.dataset.idx, 10);
        viewCandidate(candidates[idx]);
      });
    });
  }

  function renderResults(result) {
    const candidates = result.ranked_candidates || [];

    // Failed files warning banner
    if (dom.resultsFailedBanner) {
      if (result.failed_files && result.failed_files.length > 0) {
        dom.resultsFailedBanner.style.display = "block";
        const fileItems = result.failed_files
          .map((f) => `<li><strong>${f.filename}</strong>: ${f.error}</li>`)
          .join("");
        dom.resultsFailedBanner.innerHTML = `
          <strong>⚠️ ${result.failed_files.length} resume(s) could not be parsed:</strong>
          <ul style="margin: 6px 0 0 18px; line-height: 1.4;">${fileItems}</ul>
        `;
      } else {
        dom.resultsFailedBanner.style.display = "none";
        dom.resultsFailedBanner.innerHTML = "";
      }
    }

    // Categories & Top-N dropdown dynamic updates
    const availableCategories = result.available_categories || 
      Array.from(new Set(candidates.map((c) => c.category).filter(Boolean)));
    updateCategoryFilters(availableCategories);
    updateMaxResultsOptions(result.total_evaluated || candidates.length);

    // Summary
    const essSkills = (result.essential_skills || []).join(", ");
    const optSkills = (result.optional_skills || []).join(", ");
    let summaryText = "Ranked by combined match score.";
    if (essSkills) {
      summaryText += ` Essential skills: ${essSkills}.`;
    } else {
      summaryText += ` ⚠️ No essential skills identified in job description.`;
    }
    if (optSkills) {
      summaryText += ` Optional skills: ${optSkills}.`;
    } else {
      summaryText += ` (No optional skills specified).`;
    }
    dom.resultsSummary.textContent = summaryText;

    if (candidates.length === 0) {
      dom.emptyResults.style.display = "block";
      dom.chartContainer.style.display = "none";
      const wrapper = document.querySelector(".table-wrapper");
      if (wrapper) wrapper.style.display = "none";
      dom.pillEvaluated.innerHTML = `<strong>0</strong> Evaluated`;
      dom.pillMatched.innerHTML = `<strong>0</strong> Matched`;
      dom.pillReturned.innerHTML = `<strong>0</strong> Returned`;
      return;
    }

    dom.chartContainer.style.display = "block";

    // Apply filters which renders rows, updates pills, and draws chart
    applyLiveFilters();

    requestAnimationFrame(() => {
      drawScoreChart(candidates);
    });
  }

  // ========================================================================
  // SCORE DISTRIBUTION CHART (Canvas)
  // ========================================================================

  function drawScoreChart(candidates) {
    const canvas = dom.scoreChart;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;

    // Resolve non-zero width even if container was just unhidden
    const containerW = dom.chartContainer ? dom.chartContainer.clientWidth : 0;
    const parentW = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
    let clientW = canvas.clientWidth || (containerW > 40 ? containerW - 48 : 0) || (parentW > 40 ? parentW - 48 : 0);

    if (!clientW || clientW < 200) {
      clientW = Math.max(300, window.innerWidth ? Math.min(window.innerWidth - 80, 1100) : 800);
    }

    const h = 200;
    canvas.width = Math.round(clientW * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = "100%";
    canvas.style.height = `${h}px`;

    ctx.save();
    ctx.scale(dpr, dpr);

    const w = clientW;
    ctx.clearRect(0, 0, w, h);

    // Bucket scores into 10-point ranges (0-10, 10-20, ..., 90-100)
    const buckets = new Array(10).fill(0);
    (candidates || []).forEach((c) => {
      const raw = typeof c.combined_score === "number" ? c.combined_score : parseFloat(c.combined_score);
      const score = Number.isFinite(raw) ? raw : 0;
      const idx = Math.max(0, Math.min(Math.floor(score / 10), 9));
      buckets[idx]++;
    });

    const maxCount = Math.max(...buckets, 1);
    const offsetX = 50;
    const offsetY = 16;
    const chartH = h - 56;
    const chartW = Math.max(100, w - offsetX - 20);
    const barWidth = chartW / 10;

    // Grid lines & Y-axis labels
    ctx.strokeStyle = "rgba(203, 14, 49, 0.12)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = offsetY + chartH - (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(offsetX, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();

      ctx.fillStyle = "#943a47";
      ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(Math.round((maxCount / 4) * i), offsetX - 8, y + 4);
    }

    // Bars
    buckets.forEach((count, i) => {
      const barH = (count / maxCount) * chartH;
      const x = offsetX + i * barWidth + 4;
      const barW = Math.max(4, barWidth - 8);
      const y = offsetY + chartH - barH;

      if (count > 0 && barH > 0) {
        // Gradient fill: crimson red to lime green
        const grad = ctx.createLinearGradient(x, y + barH, x, y);
        if (i >= 6) {
          grad.addColorStop(0, "#CB0E31");
          grad.addColorStop(1, "#BAD62C");
        } else {
          grad.addColorStop(0, "#8f0720");
          grad.addColorStop(1, "#CB0E31");
        }
        ctx.fillStyle = grad;

        const radius = Math.min(4, barH / 2, barW / 2);
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(x, y, barW, barH, [radius, radius, 0, 0]);
        } else {
          ctx.moveTo(x + radius, y);
          ctx.lineTo(x + barW - radius, y);
          ctx.quadraticCurveTo(x + barW, y, x + barW, y + radius);
          ctx.lineTo(x + barW, y + barH);
          ctx.lineTo(x, y + barH);
          ctx.lineTo(x, y + radius);
          ctx.quadraticCurveTo(x, y, x + radius, y);
          ctx.closePath();
        }
        ctx.fill();

        // Count above bar
        ctx.fillStyle = "#45030f";
        ctx.font = "bold 11px 'Plus Jakarta Sans', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(count, x + barW / 2, Math.max(12, y - 4));
      }

      // X-axis bucket label
      ctx.fillStyle = "#6b1523";
      ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${i * 10}-${i * 10 + 10}`, x + barW / 2, offsetY + chartH + 16);
    });

    // Baseline axis
    ctx.strokeStyle = "rgba(203, 14, 49, 0.3)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY + chartH);
    ctx.lineTo(w - 20, offsetY + chartH);
    ctx.stroke();

    // X-axis label
    ctx.fillStyle = "#943a47";
    ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Combined Score Range (0 - 100)", offsetX + chartW / 2, h - 2);

    ctx.restore();
  }

  // ========================================================================
  // CANDIDATE DETAIL VIEW
  // ========================================================================

  async function viewCandidate(candidate) {
    state.selectedCandidate = candidate;
    const displayName = candidate.filename || candidate.candidate_id || "Uploaded Candidate";

    dom.candidateDetailTitle.textContent = displayName;
    if (dom.detailFilename) dom.detailFilename.textContent = candidate.filename || displayName;
    dom.detailId.textContent = candidate.candidate_id || candidate.filename || "—";
    dom.detailCategory.textContent = candidate.category || "Other / Unclassified";
    dom.detailEssCov.textContent = formatScore(candidate.essential_coverage_percent);
    dom.detailOptCov.textContent = formatScore(candidate.optional_coverage_percent);
    dom.detailExplanation.textContent =
      candidate.detailed_explanation || candidate.match_explanation || "No explanation available.";

    // Score rings - ensuring mapped to Combined, Skill Match, Text Sim
    drawScoreRing("#score-combined canvas", candidate.combined_score);
    drawScoreRing("#score-skill canvas", candidate.skill_match_score);
    drawScoreRing("#score-text canvas", candidate.text_similarity_score);

    // Skill pills
    renderSkillPills(dom.detailEssMatched, candidate.essential_skills_matched || candidate.matched_essential, "matched");
    renderSkillPills(dom.detailEssMissing, candidate.essential_skills_missing || candidate.missing_essential, "missing");
    renderSkillPills(dom.detailOptMatched, candidate.optional_skills_matched || candidate.matched_optional, "matched");
    renderSkillPills(dom.detailOptMissing, candidate.optional_skills_missing || candidate.missing_optional, "missing");

    // Collapsed original text preview by default
    if (dom.resumePreviewDetails) {
      dom.resumePreviewDetails.open = false;
    }

    // Fetch candidate profile or use attached metadata
    try {
      const profile = candidate.all_skills
        ? candidate
        : await apiFetch(`/api/candidate/${encodeURIComponent(candidate.candidate_id)}`);

      const skills = profile.all_skills || profile.skills || [];
      dom.detailAllSkills.textContent = skills.join(", ") || "None detected";
      if (dom.detailSkillCount) {
        dom.detailSkillCount.textContent = profile.skill_count !== undefined ? profile.skill_count : skills.length;
      }
      if (dom.detailWordCount) {
        dom.detailWordCount.textContent = profile.word_count || "—";
      }
      dom.detailTextPreview.textContent = profile.text_preview || "Not available.";
      dom.candidateResumeCard.style.display = "block";
    } catch (e) {
      const skills = candidate.all_skills || candidate.skills || [];
      dom.detailAllSkills.textContent = skills.join(", ") || "None detected";
      if (dom.detailSkillCount) {
        dom.detailSkillCount.textContent = candidate.skill_count !== undefined ? candidate.skill_count : skills.length;
      }
      if (dom.detailWordCount) {
        dom.detailWordCount.textContent = candidate.word_count || "—";
      }
      dom.detailTextPreview.textContent = candidate.text_preview || "Not available.";
      dom.candidateResumeCard.style.display = "block";
    }

    showSection(dom.candidateSection);
  }

  function renderSkillPills(container, skills, type) {
    container.innerHTML = "";
    if (!skills || skills.length === 0) {
      const pill = document.createElement("span");
      pill.className = "skill-pill skill-pill--none";
      pill.textContent = "None";
      container.appendChild(pill);
      return;
    }
    skills.forEach((s) => {
      const pill = document.createElement("span");
      pill.className = `skill-pill skill-pill--${type}`;
      pill.textContent = s;
      container.appendChild(pill);
    });
  }

  function drawScoreRing(selector, score) {
    const canvas = document.querySelector(selector);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const size = 120;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    const cx = size / 2;
    const cy = size / 2;
    const radius = 46;
    const lineWidth = 8;
    const pct = Math.min(Math.max(score || 0, 0), 100) / 100;

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(203, 14, 49, 0.12)";
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();

    // Score ring
    if (pct > 0) {
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + Math.PI * 2 * pct;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, endAngle);

      // Color based on score (#BAD62C lime green for high, #CB0E31 crimson red for low)
      let color = "#CB0E31"; // red for low
      if (score >= 60) color = "#BAD62C"; // vibrant lime green for high
      else if (score >= 35) color = "#d4a017"; // golden lime amber for mid
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.stroke();
    }

    // Score text
    ctx.fillStyle = "#CB0E31";
    ctx.font = "bold 22px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(formatScore(score), cx, cy);
  }

  // ========================================================================
  // NAVIGATION EVENT HANDLERS
  // ========================================================================

  dom.navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const sectionId = link.dataset.section;
      const route = SECTION_TO_ROUTE[sectionId] || "home";
      if (window.location.hash === "#" + route) {
        navigateTo(route, false);
      } else {
        window.location.hash = route;
      }
    });
  });

  const brandEl = document.querySelector(".nav-brand");
  if (brandEl) {
    brandEl.style.cursor = "pointer";
    brandEl.addEventListener("click", () => {
      if (window.location.hash === "#home" || window.location.hash === "") {
        navigateTo("home", false);
      } else {
        window.location.hash = "home";
      }
    });
  }

  dom.btnBackResults.addEventListener("click", () => {
    window.location.hash = "rankings";
  });

  window.addEventListener("hashchange", () => {
    const raw = window.location.hash.replace(/^#\/?/, "").toLowerCase() || "home";
    navigateTo(raw, false);
  });

  // Responsive chart redraw on window resize or container visibility change
  window.addEventListener("resize", () => {
    if (dom.resultsSection && dom.resultsSection.style.display !== "none" && state.rankingResult && state.rankingResult.ranked_candidates) {
      drawScoreChart(state.rankingResult.ranked_candidates);
    }
  });

  if (window.ResizeObserver && dom.chartContainer) {
    const chartObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 50 && state.rankingResult && state.rankingResult.ranked_candidates) {
          drawScoreChart(state.rankingResult.ranked_candidates);
        }
      }
    });
    chartObserver.observe(dom.chartContainer);
  }

  // ========================================================================
  // LAUNCH
  // ========================================================================
  init();

  // Set initial route from URL hash (defaults to home)
  const initialRoute = window.location.hash.replace(/^#\/?/, "").toLowerCase() || "home";
  navigateTo(initialRoute, false);
})();
