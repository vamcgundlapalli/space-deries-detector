// Space Debris Frontend JS
const API_BASE = 'http://localhost:8000';

// Colors (RGB from BGR in detect.py)
const CLASS_COLORS = {
    'debris': [255, 50, 0],
    'defunct_satellite': [0, 200, 255],
    'rocket_body': [255, 0, 200],
    'unknown': [0, 255, 128]
};

let currentParams = {
    conf: 0.25,
    iou: 0.45,
    imgsz: 640,
    denoise: false,
    enhance: false,
    weights: 'models/space_debris_yolov8/weights/best.pt'
};

let webcamStream = null;
let predictInterval = null;
let video = null;
let webcamCtx = null;

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initTabs();
    initImageUpload();
    initWebcam();
});

// Sidebar controls
function initSidebar() {
    const sliders = {
        conf: document.getElementById('conf'),
        iou: document.getElementById('iou'),
        imgsz: document.getElementById('imgsz')
    };

    // Sliders
    sliders.conf.addEventListener('input', (e) => {
        currentParams.conf = parseFloat(e.target.value);
        document.getElementById('conf-val').textContent = currentParams.conf.toFixed(2);
    });
    sliders.iou.addEventListener('input', (e) => {
        currentParams.iou = parseFloat(e.target.value);
        document.getElementById('iou-val').textContent = currentParams.iou.toFixed(2);
    });
    sliders.imgsz.addEventListener('change', (e) => {
        currentParams.imgsz = parseInt(e.target.value);
        document.getElementById('imgsz-val').textContent = currentParams.imgsz;
    });

    // Checkboxes
    document.getElementById('denoise').addEventListener('change', (e) => {
        currentParams.denoise = e.target.checked;
    });
    document.getElementById('enhance').addEventListener('change', (e) => {
        currentParams.enhance = e.target.checked;
    });

    // Init displays
    document.getElementById('conf-val').textContent = currentParams.conf;
    document.getElementById('iou-val').textContent = currentParams.iou;
    document.getElementById('imgsz-val').textContent = currentParams.imgsz;
}

// Tabs
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
        });
    });
}

// Image upload & predict
function initImageUpload() {
    const upload = document.getElementById('image-upload');
    const uploadArea = document.getElementById('upload-area');
    const predictBtn = document.getElementById('predict-btn');

    uploadArea.addEventListener('click', () => upload.click());
    upload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const origCanvas = document.getElementById('orig-canvas');
            const ctx = origCanvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                origCanvas.width = img.width;
                origCanvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                predictBtn.disabled = false;
                predictBtn.textContent = '🔍 Detect Debris';
            };
            img.src = URL.createObjectURL(file);
            window.currentImageFile = file;  // Store for predict
        }
    });

    predictBtn.addEventListener('click', async () => {
        if (!window.currentImageFile) return;
        predictBtn.disabled = true;
        predictBtn.textContent = '⏳ Detecting...';

        try {
            const formData = new FormData();
            formData.append('image', window.currentImageFile);
            Object.entries(currentParams).forEach(([k,v]) => {
                formData.append(k, v);
            });

            const response = await fetch(`${API_BASE}/detect`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`API error: ${response.status}`);

            const data = await response.json();
            displayImageResults(data);

        } catch (error) {
            alert(`Detection failed: ${error.message}\nEnsure API running: uvicorn api.app:app --reload`);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = '🔍 Detect Debris';
        }
    });
}

function displayImageResults(data) {
    // Annotated canvas
    const detectCanvas = document.getElementById('detect-canvas');
    const ctx = detectCanvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
        detectCanvas.width = img.naturalWidth;
        detectCanvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
    };
    img.src = data.annotated_image_b64;

    document.getElementById('infer-time').textContent = `Inference: ${data.inference_ms} ms`;

    // Metrics
    document.getElementById('total-det').textContent = data.total_detections;
    document.getElementById('debris-count').textContent = data.counts.debris || 0;
    document.getElementById('sat-count').textContent = data.counts.defunct_satellite || 0;
    document.getElementById('rocket-count').textContent = data.counts.rocket_body || 0;

    // Table
    const tbody = document.getElementById('det-table-body');
    tbody.innerHTML = '';
    data.detections.forEach((det, i) => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${i+1}</td>
            <td><strong>${det.class_name}</strong></td>
            <td>${det.confidence.toFixed(3)}</td>
            <td>(${det.bbox.join(', ')})</td>
        `;
    });

    // Download
    document.getElementById('download-link').href = data.annotated_image_b64;
    document.getElementById('download-link').download = 'debris_detected.png';
    document.getElementById('download-link').style.display = 'inline-block';

    // Show sections
    document.getElementById('image-results').style.display = 'grid';
    document.getElementById('metrics').style.display = 'block';
    document.getElementById('details-table').style.display = 'block';
}

// Webcam
function initWebcam() {
    const startBtn = document.getElementById('start-cam');
    const stopBtn = document.getElementById('stop-cam');
    const canvas = document.getElementById('webcam-canvas');
    video = document.createElement('video');
    webcamCtx = canvas.getContext('2d');

    // Devices
    navigator.mediaDevices.enumerateDevices().then(devices => {
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        const select = document.getElementById('cam-select');
        videoDevices.forEach((dev, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = dev.label || `Camera ${i}`;
            select.appendChild(opt);
        });
    });

    startBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 }
            });
            webcamStream = stream;
            video.srcObject = stream;
            video.play();
            canvas.style.display = 'block';

            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                startLiveDetection();
                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-block';
            };
        } catch (err) {
            alert('Webcam access denied or unavailable');
        }
    });

    stopBtn.addEventListener('click', stopWebcam);
}

function startLiveDetection() {
    let lastTime = 0;
    predictInterval = setInterval(() => {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            webcamCtx.save();
            webcamCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Predict & overlay
            predictWebcamFrame().then(data => {
                if (data && data.detections) {
                    overlayDetections(webcamCtx, data.detections, canvas.width, canvas.height);
                    updateWebcamMetrics(data);
                }
            }).catch(err => console.error('Webcam predict error:', err));

            webcamCtx.restore();

            // FPS
            const now = performance.now();
            const fps = 1000 / (now - lastTime);
            document.getElementById('fps-display').textContent = `FPS: ${fps.toFixed(1)}`;
            lastTime = now;
        }
    }, 100);  // ~10 FPS
}

async function predictWebcamFrame() {
    const canvas = document.getElementById('webcam-canvas');
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'webcam-frame.png');
        Object.entries(currentParams).forEach(([k,v]) => formData.append(k, v));

        try {
            const res = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });
            return await res.json();
        } catch {
            return null;
        }
    });
}

function overlayDetections(ctx, detections, w, h) {
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const color = CLASS_COLORS[det.class_name] || CLASS_COLORS.unknown;
        ctx.strokeStyle = `rgb(${color.join(',')})`;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);

        // Label
        ctx.fillStyle = `rgba(${color.join(',')}, 0.9)`;
        ctx.font = 'bold 20px Arial';
        ctx.fillRect(x1, y1-30, 200, 30);
        ctx.fillStyle = 'white';
        ctx.fillText(`${det.class_name} ${det.confidence.toFixed(2)}`, x1+5, y1-8);
    });
}

function updateWebcamMetrics(data) {
    const metrics = document.getElementById('webcam-metrics');
    metrics.innerHTML = `
        Total: ${data.total_detections} | 
        Debris: ${data.counts.debris || 0} | 
        Satellites: ${data.counts.defunct_satellite || 0} | 
        Rockets: ${data.counts.rocket_body || 0}
    `;
    metrics.style.display = 'block';
}

function stopWebcam() {
    if (predictInterval) clearInterval(predictInterval);
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
    document.getElementById('webcam-canvas').style.display = 'none';
    document.getElementById('start-cam').disabled = false;
    document.getElementById('start-cam').style.display = 'inline-block';
    document.getElementById('stop-cam').style.display = 'none';
    document.getElementById('webcam-metrics').style.display = 'none';
}

