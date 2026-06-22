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

        var player = new RingBufferPlayer();
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
                    player.writeChunk(float32);
                    progressDiv.textContent = 'Streaming... chunk #' + player.receivedChunks;
                });
                return;
            }

            var msg = JSON.parse(event.data);
            if (msg.type === 'meta') {
                player.ensureContext(msg.sample_rate);
                player.startDrain();
            } else if (msg.type === 'progress') {
                progressDiv.textContent = 'Chunk ' + player.receivedChunks + ' — ' +
                    (msg.chunk_samples / 48000).toFixed(2) + 's';
            } else if (msg.type === 'saved') {
                finalUrl = msg.url;
            } else if (msg.type === 'done') {
                player.flush();
                progressDiv.textContent = 'Done!';
                if (finalUrl) {
                    var audioEl = document.getElementById('player');
                    audioEl.src = finalUrl;
                    audioEl.style.display = '';
                }
                loadHistory();
                streamBtn.disabled = false;
                streamBtn.textContent = 'Stream Live';
                setStatus('ready', 'Ready');
                isGenerating = false;
            } else if (msg.type === 'error') {
                player.close();
                progressDiv.textContent = 'Error: ' + msg.message;
                streamBtn.disabled = false;
                streamBtn.textContent = 'Stream Live';
                setStatus('error', 'Error: ' + msg.message);
                isGenerating = false;
            }
        };

        ws.onerror = function() {
            player.close();
            progressDiv.textContent = 'WebSocket error';
            streamBtn.disabled = false;
            streamBtn.textContent = 'Stream Live';
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

    /* ── RingBufferPlayer: ring-buffered chunked audio playback ── */
    /* Decouples generation speed from playback speed via a circular   */
    /* ring buffer + 50ms drain ticker. Eliminates stutters/clicks.     */

    function RingBufferPlayer() {
        this.audioCtx = null;
        this.sampleRate = 48000;
        this.RING_CAPACITY = 4096 * 48;   // 196608 samples = ~2 seconds at 48kHz
        this.ringBuffer = null;            // Float32Array, created in ensureContext
        this.writePos = 0;                 // circular write pointer
        this.readPos = 0;                  // circular read pointer
        this.available = 0;                // samples currently in buffer
        this.alignment = 4096;             // chunk alignment modulus

        this.prevSource = null;            // last scheduled source for cross-fade
        this.prevBuffer = null;            // last scheduled AudioBuffer for cross-fade
        this.nextStartTime = 0;            // AudioContext time for next segment start
        this.drainTicker = null;           // setInterval ref
        this.drainActive = false;
        this.flushing = false;             // true when flush() called
        this.paused = false;               // low-watermark hysteresis state

        this.lowWatermark = 9600;          // 200ms at 48k — below this, busy-wait
        this.highWatermark = 38400;        // 800ms at 48k — above this, proceed
        this.segmentSize = 8192;           // samples per AudioBufferSourceNode (171ms)
        this.crossfadeSamples = 480;       // 10ms at 48k
        this.scheduleHorizon = 1.0;        // seconds ahead to schedule

        this.pendingChunks = [];           // chunks received before audioCtx ready
        this.receivedChunks = 0;
        this.onDone = null;                // optional callback when drain completes
    }

    RingBufferPlayer.prototype.ensureContext = function(sampleRate) {
        if (!this.audioCtx) {
            try {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: sampleRate,
                });
            } catch (_e) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
        }
        this.sampleRate = this.audioCtx.sampleRate;
        if (!this.ringBuffer) {
            this.ringBuffer = new Float32Array(this.RING_CAPACITY);
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
        // If drain was requested before audioCtx was ready, start it now
        if (this.drainActive && this.drainTicker === null) {
            var self = this;
            this.drainTicker = window.setInterval(function() {
                self._drainTick();
            }, 50);
        }
        // Replay any chunks received before audioCtx was ready
        if (this.pendingChunks.length > 0) {
            var pending = this.pendingChunks;
            this.pendingChunks = [];
            for (var i = 0; i < pending.length; i++) {
                this.writeChunk(pending[i]);
            }
        }
    };

    RingBufferPlayer.prototype.writeChunk = function(float32Array) {
        // If audioCtx isn't ready yet, buffer the chunk for replay in ensureContext
        if (!this.audioCtx || !this.ringBuffer) {
            this.pendingChunks.push(float32Array);
            return;
        }
        var n = float32Array.length;
        for (var i = 0; i < n; i++) {
            // If ring buffer is full, discard oldest sample (advance readPos)
            if (this.available >= this.RING_CAPACITY) {
                this.readPos = (this.readPos + 1) % this.RING_CAPACITY;
                this.available -= 1;
            }
            this.ringBuffer[this.writePos] = float32Array[i];
            this.writePos = (this.writePos + 1) % this.RING_CAPACITY;
            this.available += 1;
        }
        this.receivedChunks += 1;
    };

    RingBufferPlayer.prototype.startDrain = function() {
        // Idempotent — no-op if already running
        if (this.drainTicker !== null) return;
        this.drainActive = true;
        // If no audioCtx yet, drain will start when ensureContext is called
        if (!this.audioCtx) return;
        var self = this;
        this.drainTicker = window.setInterval(function() {
            self._drainTick();
        }, 50);
    };

    RingBufferPlayer.prototype._drainTick = function() {
        if (!this.audioCtx || !this.ringBuffer) return;

        var avail = this.available;

        // Low-watermark guard with hysteresis:
        // Below lowWatermark (9600): pause scheduling (busy-wait).
        // Once paused, stay paused until fill >= highWatermark (38400) to avoid flapping.
        if (!this.flushing) {
            if (this.paused) {
                if (avail < this.highWatermark) return;  // still paused
                this.paused = false;                      // resume above high watermark
            } else if (avail < this.lowWatermark) {
                this.paused = true;                        // pause below low watermark
                return;
            }
        }

        // Nothing to play
        if (avail === 0) {
            if (this.flushing) {
                this._finish();
            }
            return;
        }

        // Determine how many samples to schedule this tick
        var toSchedule = Math.min(avail, this.segmentSize);

        // Read samples from ring buffer into a new AudioBuffer (handle wrap-around)
        var buffer = this.audioCtx.createBuffer(1, toSchedule, this.sampleRate);
        var out = buffer.getChannelData(0);
        var firstRead = Math.min(toSchedule, this.RING_CAPACITY - this.readPos);
        for (var i = 0; i < firstRead; i++) {
            out[i] = this.ringBuffer[this.readPos + i];
        }
        if (toSchedule > firstRead) {
            var secondRead = toSchedule - firstRead;
            for (var j = 0; j < secondRead; j++) {
                out[firstRead + j] = this.ringBuffer[j];
            }
        }

        // Determine start time and whether segments are contiguous
        var now = this.audioCtx.currentTime;
        var contiguous = this.nextStartTime > now;
        var startTime = contiguous ? this.nextStartTime : (now + 0.05);

        // Cross-fade: if previous segment is still playing (contiguous), apply
        // 10ms linear cross-fade between tail of prev buffer and head of new buffer.
        // This modifies already-scheduled buffer data, which is safe because the
        // source hasn't reached those samples yet (they're in the lookahead horizon).
        if (contiguous && this.prevBuffer) {
            var prevData = this.prevBuffer.getChannelData(0);
            var prevLen = prevData.length;
            var cf = Math.min(this.crossfadeSamples, prevLen, toSchedule);
            for (var k = 0; k < cf; k++) {
                prevData[prevLen - cf + k] *= (1 - k / cf);
                out[k] *= (k / cf);
            }
        }

        // If this is the final segment during flush, apply 10ms fade-out on tail
        var isFinal = this.flushing && (avail === toSchedule);
        if (isFinal) {
            var fadeOut = Math.min(this.crossfadeSamples, toSchedule);
            for (var m = 0; m < fadeOut; m++) {
                out[toSchedule - fadeOut + m] *= (1 - m / fadeOut);
            }
        }

        // Create source and schedule
        var source = this.audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioCtx.destination);
        source.start(startTime);
        this.nextStartTime = startTime + (toSchedule / this.sampleRate);

        // Advance read pointer
        this.readPos = (this.readPos + toSchedule) % this.RING_CAPACITY;
        this.available -= toSchedule;

        // Store for cross-fade on next segment
        this.prevSource = source;
        this.prevBuffer = buffer;

        // If flushing and we've drained everything, stop ticker and call onDone
        if (this.flushing && this.available === 0) {
            this._finish();
        }
    };

    RingBufferPlayer.prototype._finish = function() {
        this._stopTicker();
        if (this.onDone) {
            var cb = this.onDone;
            this.onDone = null;
            cb();
        }
    };

    RingBufferPlayer.prototype._stopTicker = function() {
        if (this.drainTicker !== null) {
            window.clearInterval(this.drainTicker);
            this.drainTicker = null;
        }
        this.drainActive = false;
    };

    RingBufferPlayer.prototype.flush = function() {
        this.flushing = true;
        // The next _drainTick will drain everything regardless of lowWatermark.
        // After all samples are drained, the ticker stops itself via _finish().
    };

    RingBufferPlayer.prototype.close = function() {
        this._stopTicker();
        // Don't close audioCtx — browser may reuse it; just stop scheduling
        if (this.audioCtx) {
            console.log('RingBufferPlayer closed after ' + this.receivedChunks + ' chunks');
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
