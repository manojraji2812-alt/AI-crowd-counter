const socket = io();

const localVideo = document.getElementById("localVideo");
const ipCameraImg = document.getElementById("ipCameraImg");
const videoCanvas = document.getElementById("videoCanvas");
const ctx = videoCanvas.getContext("2d");
const videoOverlay = document.getElementById("videoOverlay");

const currentCountEl = document.getElementById("currentCount");
const totalPeopleEl = document.getElementById("totalPeople");
const crowdCard = document.getElementById("crowdCard");
const crowdLevelEl = document.getElementById("crowdLevel");
const crowdIconEl = document.getElementById("crowdIcon");
const brandDotEl = document.getElementById("brandDot");
const MAX_SEND_FPS = 6;
const MAX_PROCESS_W = 640;
const MAX_PROCESS_H = 360;
const cameraStatusEl = document.getElementById("cameraStatusText");
const cameraStatusBadge = document.getElementById("cameraStatus");
const fpsCounterEl = document.getElementById("fpsCounter");
const dateEl = document.getElementById("currentDate");
const personSelect = document.getElementById("personSelect");
const trackingList = document.getElementById("trackingList");
const personDetail = document.getElementById("personDetail");

let cameraMode = null;
let stream = null;
let sendingFrame = false;
let frameCount = 0;
let fps = 0;
let lastFpsTime = Date.now();
let lastFrameSent = 0;
let lastListDomUpdate = 0;
let allPersons = [];
let activePersons = [];
let livePersons = [];
let lastLiveUpdate = 0;

function updateDateTime() {
    const now = new Date();
    dateEl.textContent = now.toLocaleDateString("en-US", {
        month: "short", day: "numeric", year: "numeric",
    }) + " " + now.toLocaleTimeString("en-US", {
        hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
}
setInterval(updateDateTime, 1000);
updateDateTime();

// ===== IP Camera =====
function startIPCamera() {
    let baseUrl = document.getElementById("ipCameraUrl").value.trim();
    if (!baseUrl) return;
    baseUrl = baseUrl.replace(/\/+$/, "");
    cameraMode = "ip";

    videoOverlay.classList.add("hidden");
    videoCanvas.style.display = "block";
    videoCanvas.width = 640;
    videoCanvas.height = 480;

    const snapshotUrl = "/api/snapshot?url=" + encodeURIComponent(baseUrl + "/shot.jpg");
    cameraStatusEl.textContent = "Connected (IP)";
    cameraStatusEl.style.color = "#22C55E";
    cameraStatusBadge.innerHTML = '<i class="fas fa-wifi"></i> IP Camera';
    cameraStatusBadge.classList.add("connected");

    sendIPFrame(snapshotUrl);
}

function sendIPFrame(snapshotUrl) {
    if (cameraMode !== "ip") return;
    const img = new Image();
    img.onload = function () {
        if (img.naturalWidth > 0) {
            videoCanvas.width = img.naturalWidth;
            videoCanvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0);
        }
        sendFrameToServer();
        setTimeout(() => sendIPFrame(snapshotUrl), 50);
    };
    img.onerror = function () {
        setTimeout(() => sendIPFrame(snapshotUrl), 500);
    };
    img.src = snapshotUrl + "&t=" + Date.now();
}

// ===== Browser Camera =====
async function startBrowserCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        cameraMode = "browser";
        localVideo.srcObject = stream;
        localVideo.style.display = "block";
        videoCanvas.style.display = "block";
        videoOverlay.classList.add("hidden");

        localVideo.onloadedmetadata = () => {
            videoCanvas.width = localVideo.videoWidth;
            videoCanvas.height = localVideo.videoHeight;
            sendBrowserFrame();
        };
        cameraStatusEl.textContent = "Connected";
        cameraStatusEl.style.color = "#22C55E";
        cameraStatusBadge.innerHTML = '<i class="fas fa-video"></i> Connected';
        cameraStatusBadge.classList.add("connected");
    } catch (err) {
        console.error("Camera error:", err);
        cameraStatusEl.textContent = "Error";
        cameraStatusEl.style.color = "#EF4444";
    }
}

function sendBrowserFrame() {
    if (cameraMode !== "browser" || !stream) return;
    if (!sendFrameToServer()) {
        setTimeout(sendBrowserFrame, 50);
        return;
    }
    requestAnimationFrame(sendBrowserFrame);
}

function sendFrameToServer() {
    if (sendingFrame) return false;
    const now = Date.now();
    const minInterval = Math.max(120, Math.round(1000 / MAX_SEND_FPS));
    if (now - lastFrameSent < minInterval) return false;
    if (cameraMode === "browser" && localVideo.readyState >= 2) {
        ctx.drawImage(localVideo, 0, 0, videoCanvas.width, videoCanvas.height);
    }
    lastFrameSent = now;
    sendingFrame = true;
    socket.emit("video_frame", { frame: videoCanvas.toDataURL("image/jpeg", 0.5) });
    frameCount++;
    const fpsNow = Date.now();
    if (fpsNow - lastFpsTime >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFpsTime = fpsNow;
        fpsCounterEl.textContent = fps;
    }
    return true;
}

// ===== Socket events =====
socket.on("processed_frame", (data) => {
    sendingFrame = false;
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
    img.src = data.frame;

    currentCountEl.textContent = data.current_count;
    totalPeopleEl.textContent = data.all_persons?.length || 0;
    updateCrowdLevel(data.current_count || 0);

    activePersons = data.active_persons || [];
    allPersons = data.all_persons || [];
    livePersons = data.live_persons || [];
    lastLiveUpdate = Date.now();

    const now = Date.now();
    if (now - lastListDomUpdate >= 300) {
        lastListDomUpdate = now;
        updateTrackingList();
        updateDropdown();
    }
});

socket.on("error", (data) => {
    console.error("Server error:", data.message);
    sendingFrame = false;
});

socket.on("connect", () => {
    document.getElementById("systemStatus").textContent = "Active";
    document.getElementById("systemStatus").style.color = "#22C55E";
});

socket.on("disconnect", () => {
    document.getElementById("systemStatus").textContent = "Offline";
    document.getElementById("systemStatus").style.color = "#EF4444";
});

function selectPerson(id) {
    personSelect.value = id;
    showPersonDetail();
}

// ===== Tracked List (>1 min persons) =====
const TRACKED_LIST_LIMIT = 10;
let trackingListExpanded = false;

function buildTrackedItem(p, forceLive) {
    const firstSeen = new Date(p.first_seen * 1000).toLocaleTimeString();
    const lastSeen = new Date(p.last_seen * 1000).toLocaleTimeString();
    const duration = formatDuration(p.duration);
    const statusDot = forceLive || p.currently_visible
        ? '<span class="list-dot-green"></span>'
        : '<span class="list-dot-gray"></span>';
    return `
        <div class="tracked-list-item" onclick="selectPerson(${p.id})">
            <div class="list-item-header">
                <span class="list-item-id">${statusDot} ID ${p.id}</span>
                <span class="list-item-status">${forceLive || p.currently_visible ? "Active" : "Left"}</span>
            </div>
            <div class="list-item-details">
                <span><i class="fas fa-clock me-1"></i>${firstSeen}</span>
                <span class="list-item-duration"><i class="fas fa-hourglass-half me-1"></i>${duration}</span>
            </div>
            <div class="list-item-details">
                <span><i class="fas fa-sign-out-alt me-1"></i>${lastSeen}</span>
            </div>
        </div>`;
}

function updateTrackingList() {
    const isLive = cameraMode !== null && (Date.now() - lastLiveUpdate) < 5000;

    const liveIds = new Set((livePersons || []).map((p) => p.id));
    const merged = [
        ...(livePersons || []),
        ...(activePersons || []).filter((p) => !liveIds.has(p.id)),
    ];
    activePersons = merged;

    if (merged.length === 0) {
        trackingListExpanded = false;
        trackingList.innerHTML = `
            <div class="tracked-list-empty">
                <i class="fas fa-clock me-1"></i> ${isLive ? "Waiting for persons to be detected..." : "No persons stayed in view for 1+ minute yet"}
            </div>`;
        return;
    }

    if (merged.length <= TRACKED_LIST_LIMIT) {
        trackingListExpanded = false;
    }

    const visible = trackingListExpanded
        ? merged
        : merged.slice(0, TRACKED_LIST_LIMIT);

    let html = visible.map((p) => buildTrackedItem(p, liveIds.has(p.id))).join("");

    if (merged.length > TRACKED_LIST_LIMIT) {
        const label = trackingListExpanded
            ? `<span>Showing all ${merged.length}</span><i class="fas fa-chevron-up"></i>`
            : `<span>${TRACKED_LIST_LIMIT} of ${merged.length}</span><i class="fas fa-chevron-down"></i>`;
        html += `
            <div class="tracked-list-more" onclick="toggleTrackingList()">
                ${label}
            </div>`;
    }

    trackingList.innerHTML = html;
}

function toggleTrackingList() {
    trackingListExpanded = !trackingListExpanded;
    updateTrackingList();
}

// ===== Crowd Level =====
const CROWD_LOW_MAX = 5;
const CROWD_HIGH_MIN = 15;

function updateCrowdLevel(count) {
    let level = "low";
    let label = "Low";
    if (count >= CROWD_HIGH_MIN) {
        level = "high";
        label = "High";
    } else if (count >= CROWD_LOW_MAX) {
        level = "medium";
        label = "Medium";
    }
    crowdCard.className = "header-card level-" + level;
    crowdLevelEl.textContent = label;
    crowdIconEl.querySelector("i").className = "fas fa-signal";
    brandDotEl.className = "brand-dot level-" + level;
}

// ===== Dropdown (all persons) =====
function updateDropdown() {
    const currentVal = personSelect.value;
    let html = '<option value="">-- View All ID --</option>';
    [...allPersons]
        .sort((a, b) => a.id - b.id)
        .forEach((p) => {
            const dur = formatDuration(p.duration);
            const firstSeen = new Date(p.first_seen * 1000).toLocaleTimeString();
            const isActive = p.currently_visible && p.duration >= 60;
            const tag = isActive ? " [active]" : (p.currently_visible ? " [seen]" : " [away]");
            html += `<option value="${p.id}">ID ${p.id} · ${firstSeen} · ${dur}${tag}</option>`;
        });
    personSelect.innerHTML = html;
    if (currentVal) personSelect.value = currentVal;
}

// ===== Person Detail =====
function showPersonDetail() {
    const id = parseInt(personSelect.value);
    if (!id) { personDetail.style.display = "none"; return; }

    const person = allPersons.find((p) => p.id === id);
    if (!person) { personDetail.style.display = "none"; return; }

    document.getElementById("detailId").textContent = person.id;
    document.getElementById("detailFirstSeen").textContent = new Date(person.first_seen * 1000).toLocaleTimeString();
    document.getElementById("detailLastSeen").textContent = new Date(person.last_seen * 1000).toLocaleTimeString();
    document.getElementById("detailDuration").textContent = formatDuration(person.duration);
    document.getElementById("detailStatus").innerHTML = person.currently_visible
        ? (person.duration >= 60
            ? '<span class="status-tracked"><span class="status-dot-green"></span> Active (>= 1 min)</span>'
            : '<span style="color:#F59E0B;">Recently Seen</span>')
        : '<span style="color:rgba(255,255,255,0.4);">Left View</span>';

    personDetail.style.display = "block";
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}m ${s}s`;
}
