(function() {
    'use strict';

    let selectedReference = null;
    let isGenerating = false;
    let statusPollInterval = null;

    const API = {
        get(url) {
            return fetch(url).then(function(resp) {
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                return resp.json();
            });
        },
        postJSON(url, body) {
            return fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            }).then(function(resp) {
                return resp.json().then(function(data) {
                    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
                    return data;
                });
            });
        },
    };

    function setStatus(state, text) {
        const el = document.getElementById('status');
        if (!el) return;
        el.textContent = text || '';
        el.setAttribute('state', state || '');
    }

    function formatSize(bytes) {
        return (bytes / 1024).toFixed(1) + ' KB';
    }

    function loadHistory() {
        return API.get('/api/outputs').then(function(items) {
            const list = document.getElementById('history-list');
            if (!list) return;
            list.innerHTML = '';
            items.forEach(function(item) {
                const li = document.createElement('li');
                const audio = document.createElement('audio');
                audio.src = '/outputs/' + item.name;
                audio.controls = true;
                li.appendChild(audio);
                const meta = document.createElement('div');
                meta.className = 'meta';
                meta.textContent = item.name + ' — ' + formatSize(item.size_bytes);
                li.appendChild(meta);
                list.appendChild(li);
            });
        }).catch(function(e) {
            console.error('loadHistory failed:', e);
        });
    }

    function loadUploads() {
        return API.get('/api/uploads').then(function(items) {
            const list = document.getElementById('uploads-list');
            if (!list) return;
            list.innerHTML = '';
            items.forEach(function(item) {
                const li = document.createElement('li');
                li.textContent = item.name + ' — ' + formatSize(item.size_bytes);
                list.appendChild(li);
            });
        }).catch(function(e) {
            console.error('loadUploads failed:', e);
        });
    }

    function showSelectedReference(name) {
        const div = document.getElementById('selected-reference');
        if (!div) return;
        div.style.display = '';
        div.innerHTML = '';
        var span = document.createElement('span');
        span.textContent = 'Using reference: ' + name + ' ';
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = 'Remove';
        btn.addEventListener('click', function() {
            selectedReference = null;
            div.style.display = 'none';
            div.innerHTML = '';
            document.getElementById('reference_upload').value = '';
        });
        div.appendChild(span);
        div.appendChild(btn);
    }

    function onGenerateFormSubmit(event) {
        event.preventDefault();
        if (isGenerating) return;
        isGenerating = true;

        var btn = document.getElementById('generate-btn');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        setStatus('loading', 'Generating...');

        var params = {
            text: document.getElementById('text').value,
            voice_description: document.getElementById('voice_description').value || null,
            reference_wav_path: selectedReference ? selectedReference.path : null,
            cfg_value: parseFloat(document.getElementById('cfg_value').value),
            inference_timesteps: parseInt(document.getElementById('inference_timesteps').value, 10),
            normalize: document.getElementById('normalize').checked,
            attempts: parseInt(document.getElementById('attempts').value, 10),
            seed: document.getElementById('seed').value
                ? parseInt(document.getElementById('seed').value, 10)
                : null,
        };

        API.postJSON('/api/generate', params).then(function(result) {
            var player = document.getElementById('player');
            player.src = result.url;
            player.style.display = '';
            return loadHistory();
        }).then(function() {
            setStatus('ready', 'Ready');
        }).catch(function(e) {
            console.error(e);
            setStatus('error', 'Error: ' + e.message);
        }).finally(function() {
            btn.disabled = false;
            btn.textContent = 'Generate';
            isGenerating = false;
        });
    }

    function onStreamBtnClick() {
        if (isGenerating) return;
        isGenerating = true;

        var streamBtn = document.getElementById('stream-btn');
        var progressDiv = document.getElementById('stream-progress');
        streamBtn.disabled = true;
        streamBtn.textContent = 'Streaming...';
        progressDiv.style.display = '';
        progressDiv.textContent = 'Streaming...';
        setStatus('loading', 'Streaming...');

        var text = document.getElementById('text').value;
        var voiceDescription = document.getElementById('voice_description').value || null;
        var cfgValue = parseFloat(document.getElementById('cfg_value').value);
        var inferenceTimesteps = parseInt(
            document.getElementById('inference_timesteps').value, 10
        ) || 12;
        var normalize = document.getElementById('normalize').checked;
        var seedRaw = document.getElementById('seed').value;
        var seed = seedRaw ? parseInt(seedRaw, 10) : null;

        var player = new StreamPlayer();
        var finalUrl = null;

        var ws = new WebSocket('ws://' + location.host + '/api/generate/stream');

        ws.onopen = function() {
            ws.send(JSON.stringify({
                type: 'start',
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
                    var float32 = new Float32Array(ab, 0, ab.byteLength / 4);
                    player.ensureContext(48000);
                    player.enqueueChunk(float32);
                    progressDiv.textContent = 'Streaming... chunk #' + player.receivedChunks;
                });
                return;
            }

            var msg = JSON.parse(event.data);
            if (msg.type === 'meta') {
                player.ensureContext(msg.sample_rate);
            } else if (msg.type === 'progress') {
                progressDiv.textContent = 'Chunk ' + player.receivedChunks + ' — ' +
                    (msg.chunk_samples / 48000).toFixed(2) + 's';
            } else if (msg.type === 'saved') {
                finalUrl = msg.url;
            } else if (msg.type === 'done') {
                progressDiv.textContent = 'Done!';
                if (finalUrl) {
                    var audioEl = document.getElementById('player');
                    audioEl.src = finalUrl;
                    audioEl.style.display = '';
                }
                loadHistory();
                streamBtn.disabled = false;
                streamBtn.textContent = 'Generate & Stream Live';
                setStatus('ready', 'Ready');
                isGenerating = false;
            } else if (msg.type === 'error') {
                progressDiv.textContent = 'Error: ' + msg.message;
                streamBtn.disabled = false;
                streamBtn.textContent = 'Generate & Stream Live';
                setStatus('error', 'Error: ' + msg.message);
                isGenerating = false;
            }
        };

        ws.onerror = function() {
            progressDiv.textContent = 'WebSocket error';
            streamBtn.disabled = false;
            streamBtn.textContent = 'Generate & Stream Live';
            setStatus('error', 'WebSocket error');
            isGenerating = false;
        };

        ws.onclose = function() {
            player.close();
        };
    }

    function startStatusPolling() {
        statusPollInterval = window.setInterval(function() {
            API.get('/api/status').then(function(data) {
                if (data.state === 'ready') {
                    return API.get('/api/info').then(function(info) {
                        setStatus('ready', 'Ready — ' + info.device +
                            (info.cuda_available ? ' (CUDA)' : '') +
                            ' | VoxCPM v' + (info.voxcpm_version || '?'));
                        window.clearInterval(statusPollInterval);
                    }).catch(function() {
                        setStatus('ready', 'Ready');
                        window.clearInterval(statusPollInterval);
                    });
                } else if (data.state === 'loading') {
                    setStatus('loading', 'Loading model...');
                } else if (data.state === 'uninitialized') {
                    setStatus('', 'Connecting...');
                }
            }).catch(function() {
                // server not ready yet, keep polling
            });
        }, 1000);
    }

    /* ── StreamPlayer: chunked audio playback ── */

    function StreamPlayer() {
        this.audioCtx = null;
        this.nextStartTime = 0;
        this.receivedChunks = 0;
    }

    StreamPlayer.prototype.ensureContext = function(sampleRate) {
        if (!this.audioCtx) {
            try {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: sampleRate,
                });
            } catch (_e) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
    };

    StreamPlayer.prototype.enqueueChunk = function(float32Array) {
        if (!this.audioCtx) return;
        var buffer = this.audioCtx.createBuffer(
            1, float32Array.length, this.audioCtx.sampleRate
        );
        buffer.getChannelData(0).set(float32Array);
        var source = this.audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioCtx.destination);
        var now = Math.max(this.nextStartTime, this.audioCtx.currentTime);
        source.start(now);
        this.nextStartTime = now + buffer.duration;
        this.receivedChunks += 1;
    };

    StreamPlayer.prototype.close = function() {
        if (this.audioCtx) {
            console.log('StreamPlayer closed after ' + this.receivedChunks + ' chunks');
        }
    };

    /* ── Wire up ── */

    document.addEventListener('DOMContentLoaded', function() {
        document.getElementById('generate-form').addEventListener('submit', onGenerateFormSubmit);

        var cfgSlider = document.getElementById('cfg_value');
        cfgSlider.addEventListener('input', function() {
            document.getElementById('cfg_value_display').textContent = this.value;
            this.setAttribute('aria-valuenow', this.value);
            this.setAttribute('aria-valuetext', this.value);
        });

        document.getElementById('stream-btn').disabled = false;
        document.getElementById('stream-btn').addEventListener('click', onStreamBtnClick);

        document.getElementById('reference_upload').addEventListener('change', async function(evt) {
            var file = evt.target.files[0];
            if (!file) {
                selectedReference = null;
                document.getElementById('selected-reference').style.display = 'none';
                return;
            }
            var formData = new FormData();
            formData.append('reference', file);
            try {
                var resp = await fetch('/api/uploads', { method: 'POST', body: formData });
                var data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Upload failed');
                selectedReference = { name: data.name, path: data.path };
                showSelectedReference(data.name);
                await loadUploads();
            } catch (e) {
                setStatus('error', 'Upload error: ' + e.message);
                console.error(e);
            }
        });

        startStatusPolling();
        loadHistory();
        loadUploads();
    });

})();
