(() => {
  "use strict";

  const tabs = Array.from(document.querySelectorAll("[data-suite-tab]"));
  const tabsContainer = document.querySelector(".suite-tabs");
  const panels = Array.from(document.querySelectorAll("[data-suite-panel]"));
  const videos = Array.from(document.querySelectorAll("video"));
  const taskPlayer = document.querySelector("#task-player");
  let selectTaskVideo = () => {};

  if (taskPlayer && typeof taskPlayer.querySelector === "function") {
    const play = taskPlayer.querySelector("#task-play");
    const playIcon = play.querySelector("[data-play-icon]");
    const pauseIcon = play.querySelector("[data-pause-icon]");
    const seek = taskPlayer.querySelector("#task-seek");
    const elapsed = taskPlayer.querySelector("#task-elapsed");
    const duration = taskPlayer.querySelector("#task-duration");
    const fullscreen = taskPlayer.querySelector("#task-fullscreen");
    const status = taskPlayer.querySelector("#task-player-status");
    let currentVideo;

    function timestamp(seconds) {
      const value = Math.max(0, Math.floor(seconds || 0));
      return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
    }

    function updatePlayer() {
      if (!currentVideo) return;
      const paused = currentVideo.paused;
      const ready = Number.isFinite(currentVideo.duration) && currentVideo.duration > 0;
      playIcon.hidden = !paused;
      pauseIcon.hidden = paused;
      play.setAttribute("aria-label", paused ? "Play demonstration" : "Pause demonstration");
      play.title = paused ? "Play" : "Pause";
      elapsed.textContent = timestamp(currentVideo.currentTime);
      duration.textContent = ready ? timestamp(currentVideo.duration) : "--:--";
      seek.disabled = !ready;
      seek.max = ready ? currentVideo.duration : 100;
      seek.value = currentVideo.currentTime || 0;
      seek.setAttribute("aria-valuetext", `${timestamp(currentVideo.currentTime)} of ${ready ? timestamp(currentVideo.duration) : "unknown duration"}`);
    }

    selectTaskVideo = (video) => {
      currentVideo = video;
      status.textContent = "";
      updatePlayer();
    };
    taskPlayer.hidden = false;
    videos.forEach((video) => {
      video.controls = false;
      ["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "ended", "emptied"].forEach((name) => {
        video.addEventListener(name, () => {
          if (video === currentVideo) updatePlayer();
        });
      });
      video.addEventListener("webkitendfullscreen", () => { video.controls = false; });
    });
    play.addEventListener("click", async () => {
      const video = currentVideo;
      if (!video) return;
      if (!video.paused) {
        video.pause();
        return;
      }
      try {
        status.textContent = "";
        await video.play();
      } catch (error) {
        if (video === currentVideo) status.textContent = "Unable to play this demonstration. Try again.";
      }
    });
    seek.addEventListener("input", () => {
      if (currentVideo && Number.isFinite(currentVideo.duration) && currentVideo.duration > 0) {
        currentVideo.currentTime = Number(seek.value);
        updatePlayer();
      }
    });
    fullscreen.hidden = !videos.some((video) => video.requestFullscreen || video.webkitEnterFullscreen);
    fullscreen.addEventListener("click", async () => {
      const video = currentVideo;
      if (!video) return;
      video.controls = true;
      try {
        if (video.requestFullscreen) await video.requestFullscreen();
        else video.webkitEnterFullscreen();
      } catch (error) {
        video.controls = false;
        status.textContent = "Fullscreen is unavailable.";
      }
    });
    document.addEventListener("fullscreenchange", () => {
      videos.forEach((video) => { video.controls = document.fullscreenElement === video; });
    });
  }

  function selectVisibleVideo(panel) {
    const task = Array.from(panel.querySelectorAll(".task-example")).find((item) => !item.hidden);
    selectTaskVideo(task?.querySelectorAll("video")[0]);
  }

  function activateTab(selectedTab, moveFocus = false) {
    const suite = selectedTab.dataset.suiteTab;
    tabs.forEach((tab) => {
      const selected = tab === selectedTab;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      const selected = panel.dataset.suitePanel === suite;
      if (!selected) {
        panel.querySelectorAll("video").forEach((video) => video.pause());
      }
      panel.hidden = !selected;
      if (selected) selectVisibleVideo(panel);
    });
    if (moveFocus) selectedTab.focus();
  }

  function bindTabList(group, activate) {
    group.forEach((tab, index) => {
      tab.addEventListener("click", () => activate(tab));
      tab.addEventListener("keydown", (event) => {
        let nextIndex;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % group.length;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + group.length) % group.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = group.length - 1;
        if (nextIndex !== undefined) {
          event.preventDefault();
          activate(group[nextIndex], true);
        }
      });
    });
  }

  panels.forEach((panel) => {
    const taskTabs = Array.from(panel.querySelectorAll("[data-task-tab]"));
    const tasks = Array.from(panel.querySelectorAll(".task-example"));
    if (!taskTabs.length) return;

    function activateTask(selectedTab, moveFocus = false) {
      taskTabs.forEach((tab) => {
        const selected = tab === selectedTab;
        tab.setAttribute("aria-selected", String(selected));
        tab.tabIndex = selected ? 0 : -1;
      });
      tasks.forEach((task) => {
        const selected = task.dataset.task === selectedTab.dataset.taskTab;
        if (!selected) task.querySelectorAll("video").forEach((video) => video.pause());
        task.hidden = !selected;
      });
      selectVisibleVideo(panel);
      if (moveFocus) selectedTab.focus();
    }

    panel.querySelector(".task-tabs").hidden = false;
    bindTabList(taskTabs, activateTask);
    activateTask(taskTabs.find((tab) => tab.getAttribute("aria-selected") === "true") || taskTabs[0]);
  });

  bindTabList(tabs, activateTab);

  if (tabs.length) {
    if (tabsContainer) tabsContainer.hidden = false;
    activateTab(tabs.find((tab) => tab.getAttribute("aria-selected") === "true") || tabs[0]);
  }

  videos.forEach((video) => {
    video.addEventListener("play", () => {
      videos.forEach((other) => {
        if (other !== video) other.pause();
      });
    });
  });

  const navigationLinks = Array.from(document.querySelectorAll("nav a"));
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.find((entry) => entry.isIntersecting);
      if (!visible) return;
      navigationLinks.forEach((link) => {
        if (link.getAttribute("href") === `#${visible.target.id}`) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    }, { rootMargin: "-15% 0px -65% 0px" });
    navigationLinks.forEach((link) => {
      const section = document.querySelector(link.getAttribute("href"));
      if (section) observer.observe(section);
    });
  }

  const viewer = document.querySelector("#figure-viewer");
  if (viewer && typeof viewer.showModal === "function") {
    const canvas = viewer.querySelector(".viewer-canvas");
    const viewerTitle = viewer.querySelector("#figure-viewer-title");
    const original = viewer.querySelector("#figure-original");
    const zoomButton = viewer.querySelector("#figure-zoom");
    let lastFigure;

    document.querySelectorAll(".figure-link").forEach((link) => {
      link.addEventListener("click", (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        lastFigure = link;
        const source = link.querySelector("img");
        const enlarged = document.createElement("img");
        enlarged.src = source.src;
        enlarged.alt = source.alt;
        enlarged.width = source.width;
        enlarged.height = source.height;
        canvas.replaceChildren(enlarged);
        viewerTitle.textContent = link.dataset.figureTitle;
        original.href = link.href;
        viewer.style.setProperty("--figure-width", `${source.getAttribute("width")}px`);
        viewer.classList.remove("is-zoomed");
        zoomButton.setAttribute("aria-pressed", "false");
        zoomButton.setAttribute("aria-label", "View figure at original size");
        zoomButton.title = "Original size";
        viewer.showModal();
        canvas.scrollTo(0, 0);
      });
    });

    zoomButton.addEventListener("click", () => {
      const zoomed = viewer.classList.toggle("is-zoomed");
      zoomButton.setAttribute("aria-pressed", String(zoomed));
      zoomButton.setAttribute("aria-label", zoomed ? "Fit figure to window" : "View figure at original size");
      zoomButton.title = zoomed ? "Fit to window" : "Original size";
      canvas.scrollTo(0, 0);
    });
    viewer.querySelector("#figure-close").addEventListener("click", () => viewer.close());
    viewer.addEventListener("click", (event) => {
      if (event.target === viewer) viewer.close();
    });
    viewer.addEventListener("close", () => {
      canvas.replaceChildren();
      if (lastFigure) lastFigure.focus({ preventScroll: true });
    });
  }

  const copyButton = document.querySelector("#copy-citation");
  const copyStatus = document.querySelector("#copy-status");

  function fallbackCopy(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  }

  if (copyButton) {
    const copyLabel = copyButton.querySelector?.("[data-copy-label]") || copyButton;
    copyButton.hidden = false;
    copyButton.addEventListener("click", async () => {
      const target = document.getElementById(copyButton.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          try {
            await navigator.clipboard.writeText(text);
          } catch (error) {
            if (!fallbackCopy(text)) throw new Error("Copy command was unavailable");
          }
        } else if (!fallbackCopy(text)) {
          throw new Error("Copy command was unavailable");
        }
        copyStatus.textContent = "Citation copied.";
        copyLabel.textContent = "Copied";
      } catch (error) {
        copyStatus.textContent = "Copy unavailable. Select the BibTeX text to copy it manually.";
        copyLabel.textContent = "Copy";
      }
    });
  }
})();
