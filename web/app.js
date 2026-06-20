(function() {
    'use strict';

    let selectedReference = null;
    let isGenerating = false;
    let statusPollInterval = null;

    async function getJSON(url) {
        const resp = await fetch(url);
        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }
        return resp.json();
    }

    async function postJSON(url, body) {
        const resp = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        return data;
    }

    function setStatus(state, text) {
        const el = document.getElementById("status");
        if (!el) {
            return;
        }
        el.textContent = text || "";
        el.setAttribute("state", state || "");
    }

    async function loadHistory() {
        try {
            const items = await getJSON("/api/outputs");
            const list = document.getElementById("history-list");
            if (!list) {
                return;
            }
            list.innerHTML = "";
            items.forEach(function(item) {
                const li = document.createElement("li");
                const audio = document.createElement("audio");
                audio.src = "/outputs/" + item.name;
                audio.controls = true;
                li.appendChild(audio);
                const meta = document.createElement("div");
                meta.textContent = item.name + " (" + (item.size_bytes / 1024).toFixed(1) + " KB)";
                li.appendChild(meta);
                list.appendChild(li);
            });
        } catch (e) {
            console.error("loadHistory failed:", e);
        }
    }

    async function loadUploads() {
        try {
            const items = await getJSON("/api/uploads");
            const list = document.getElementById("uploads-list");
            if (!list) {
                return;
            }
            list.innerHTML = "";
            items.forEach(function(item) {
                const li = document.createElement("li");
                li.textContent = item.name + " (" + (item.size_bytes / 1024).toFixed(1) + " KB)";
                list.appendChild(li);
            });
        } catch (e) {
            console.error("loadUploads failed:", e);
        }
    }

    function showSelectedReference(name) {
        const div = document.getElementById("selected-reference");
        if (!div) {
            return;
        }
        div.style.display = "block";
        div.innerHTML = "";
        const text = document.createElement("span");
        text.textContent = "Using reference: " + name + " ";
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "Remove";
        btn.addEventListener("click", function() {
            selectedReference = null;
            div.style.display = "none";
            div.innerHTML = "";
            document.getElementById("reference_upload").value = "";
        });
        div.appendChild(text);
        div.appendChild(btn);
    }

    async function onGenerateFormSubmit(event) {
        event.preventDefault();
        if (isGenerating) {
            return;
        }
        isGenerating = true;

        const btn = document.getElementById("generate-btn");
        btn.disabled = true;
        btn.textContent = "Generating...";
        setStatus("loading", "Generating...");

        const text = document.getElementById("text").value;
        const voiceDescription = document.getElementById("voice_description").value || null;
        const cfgValue = parseFloat(document.getElementById("cfg_value").value);
        const inferenceTimesteps = parseInt(document.getElementById("inference_timesteps").value, 10);
        const normalize = document.getElementById("normalize").checked;
        const attempts = parseInt(document.getElementById("attempts").value, 10);
        const seedRaw = document.getElementById("seed").value;
        const seed = seedRaw ? parseInt(seedRaw, 10) : null;

        const params = {
            text: text,
            voice_description: voiceDescription,
            reference_wav_path: selectedReference ? selectedReference.path : null,
            cfg_value: cfgValue,
            inference_timesteps: inferenceTimesteps,
            normalize: normalize,
            attempts: attempts,
            seed: seed,
        };

        try {
            const result = await postJSON("/api/generate", params);
            const player = document.getElementById("player");
            player.src = result.url;
            player.style.display = "block";
            await loadHistory();
            setStatus("ready", "Ready");
        } catch (e) {
            console.error(e);
            setStatus("error", "Error: " + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = "Generate";
            isGenerating = false;
        }
    }

    function onStreamBtnClick() {
        if (isGenerating) {
            return;
        }
        isGenerating = true;

        const streamBtn = document.getElementById("stream-btn");
        const progressDiv = document.getElementById("stream-progress");
        streamBtn.disabled = true;
        streamBtn.textContent = "Streaming...";
        progressDiv.style.display = "block";
        progressDiv.textContent = "Streaming...";
        setStatus("loading", "Streaming...");

        const text = document.getElementById("text").value;
        const voiceDescription = document.getElementById("voice_description").value || null;
        const cfgValue = parseFloat(document.getElementById("cfg_value").value);
        const inferenceTimesteps = parseInt(document.getElementById("inference_timesteps").value, 10) || 12;
        const normalize = document.getElementById("normalize").checked;
        const seedRaw = document.getElementById("seed").value;
        const seed = seedRaw ? parseInt(seedRaw, 10) : null;

        const player = new StreamPlayer();
        let finalUrl = null;

        const ws = new WebSocket("ws://" + location.host + "/api/generate/stream");

        ws.onopen = function() {
            ws.send(JSON.stringify({
                type: "start",
                params: {
                    text: text,
                    voice_description: voiceDescription,
                    reference_wav_path: selectedReference ? selectedReference.path : null,
                    cfg_value: cfgValue,
                    inference_timesteps: inferenceTimesteps,
                    normalize: normalize,
                    seed: seed,
                },
            }));
        };

        ws.onmessage = function(event) {
            if (event.data instanceof Blob) {
                event.data.arrayBuffer().then(function(ab) {
                    const float32 = new Float32Array(ab, 0, ab.byteLength / 4);
                    player.ensureContext(48000);
                    player.enqueueChunk(float32);
                    progressDiv.textContent = "Streaming... chunk #" + player.receivedChunks;
                });
                return;
            }

            const msg = JSON.parse(event.data);
            if (msg.type === "meta") {
                player.ensureContext(msg.sample_rate);
            } else if (msg.type === "progress") {
                progressDiv.textContent = "Chunk " + player.receivedChunks + " | " + (msg.chunk_samples / 48000).toFixed(2) + "s";
            } else if (msg.type === "saved") {
                finalUrl = msg.url;
            } else if (msg.type === "done") {
                progressDiv.textContent = "Done!";
                if (finalUrl) {
                    const audioEl = document.getElementById("player");
                    audioEl.src = finalUrl;
                    audioEl.style.display = "block";
                }
                loadHistory();
                streamBtn.disabled = false;
                streamBtn.textContent = "Generate & Stream Live";
                setStatus("ready", "Ready");
                isGenerating = false;
            } else if (msg.type === "error") {
                progressDiv.textContent = "Error: " + msg.message;
                streamBtn.disabled = false;
                streamBtn.textContent = "Generate & Stream Live";
                setStatus("error", "Error: " + msg.message);
                isGenerating = false;
            }
        };

        ws.onerror = function(_err) {
            progressDiv.textContent = "WebSocket error";
            streamBtn.disabled = false;
            streamBtn.textContent = "Generate & Stream Live";
            setStatus("error", "WebSocket error");
            isGenerating = false;
        };

        ws.onclose = function() {
            player.close();
        };
    }

    function startStatusPolling() {
        statusPollInterval = window.setInterval(async function() {
            try {
                const data = await getJSON("/api/status");
                if (data.state === "ready") {
                    const info = await getJSON("/api/info");
                    setStatus("ready", "Ready — " + info.device + (info.cuda_available ? " (CUDA)" : "") + " | VoxCPM v" + (info.voxcpm_version || "?"));
                    window.clearInterval(statusPollInterval);
                } else if (data.state === "loading") {
                    setStatus("loading", "Loading model...");
                } else if (data.state === "uninitialized") {
                    setStatus("", "Connecting...");
                }
            } catch (_e) {
            }
        }, 1000);
    }

    class StreamPlayer {
        constructor() {
            this.audioCtx = null;
            this.nextStartTime = 0;
            this.receivedChunks = 0;
        }

        ensureContext(sampleRate) {
            if (!this.audioCtx) {
                try {
                    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: sampleRate});
                } catch (_e) {
                    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                }
            }
            if (this.audioCtx.state === "suspended") {
                this.audioCtx.resume();
            }
        }

        enqueueChunk(float32Array) {
            if (!this.audioCtx) {
                return;
            }
            const buffer = this.audioCtx.createBuffer(1, float32Array.length, this.audioCtx.sampleRate);
            buffer.getChannelData(0).set(float32Array);
            const source = this.audioCtx.createBufferSource();
            source.buffer = buffer;
            source.connect(this.audioCtx.destination);
            const now = Math.max(this.nextStartTime, this.audioCtx.currentTime);
            source.start(now);
            this.nextStartTime = now + buffer.duration;
            this.receivedChunks += 1;
        }

        close() {
            if (this.audioCtx) {
                console.log("StreamPlayer closed after " + this.receivedChunks + " chunks");
            }
        }
    }

    document.addEventListener("DOMContentLoaded", function() {
        document.getElementById("generate-form").addEventListener("submit", onGenerateFormSubmit);
        document.getElementById("cfg_value").addEventListener("input", function() {
            document.getElementById("cfg_value_display").textContent = this.value;
        });
        const streamBtn = document.getElementById("stream-btn");
        streamBtn.disabled = false;
        streamBtn.addEventListener("click", onStreamBtnClick);
        document.getElementById("reference_upload").addEventListener("change", async function(evt) {
            const file = evt.target.files[0];
            if (!file) {
                selectedReference = null;
                document.getElementById("selected-reference").style.display = "none";
                return;
            }
            const formData = new FormData();
            formData.append("reference", file);
            try {
                const resp = await fetch("/api/uploads", {method: "POST", body: formData});
                const data = await resp.json();
                if (!resp.ok) {
                    throw new Error(data.error || `HTTP ${resp.status}`);
                }
                selectedReference = {name: data.name, path: data.path};
                showSelectedReference(data.name);
                await loadUploads();
            } catch (e) {
                console.error(e);
                setStatus("error", "Upload error: " + e.message);
            }
        });

        startStatusPolling();
        loadHistory();
        loadUploads();
    });
})();
