(function() {
    'use strict';

    let selectedReference = null;
    let isGenerating = false;
    let statusPollInterval = null;
    let enginesData = [];
    let activeJobPollInterval = null;
    let activeJobName = null;

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
                    if (!resp.ok) throw new Error(data.error || data.detail || `HTTP ${resp.status}`);
                    return data;
                });
            });
        },
    };

    function generateJobName() {
        const now = new Date();
        const pad = (n) => n.toString().padStart(2, '0');
        return `job-${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    }

    function setStatus(state, text) {
        const el = document.getElementById('status');
        if (!el) return;
        el.textContent = text || '';
        el.setAttribute('state', state || '');
    }

    function formatSize(bytes) {
        return (bytes / 1024).toFixed(1) + ' KB';
    }

    function loadEngines() {
        return API.get('/api/engines').then(function(data) {
            if (Array.isArray(data)) {
                enginesData = data;
            } else if (typeof data === 'object') {
                enginesData = Object.values(data);
            } else {
                enginesData = [];
            }
            renderEngines();
        }).catch(function(e) {
            console.error('loadEngines failed:', e);
        });
    }

    function renderEngines() {
        const bar = document.getElementById('engines-bar');
        if (!bar) return;
        const selectedBefore = new Set(getSelectedEngines());
        bar.innerHTML = '';
        
        let hasSelected = false;
        
        enginesData.forEach(function(eng) {
            const label = document.createElement('label');
            label.className = 'engine-toggle';
            
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.value = eng.engine_id;
            cb.dataset.capabilities = JSON.stringify(eng.capabilities || {});
            
            if (!eng.installed) {
                label.classList.add('disabled');
                cb.disabled = true;
            } else {
                if (selectedBefore.has(eng.engine_id) || (selectedBefore.size === 0 && !hasSelected && eng.engine_id === 'voxcpm')) {
                    cb.checked = true;
                    hasSelected = true;
                }
            }
            
            cb.addEventListener('change', function() {
                if (cb.checked) label.classList.add('selected');
                else label.classList.remove('selected');
                updateStreamButtonState();
            });
            
            const nameDiv = document.createElement('div');
            nameDiv.className = 'engine-name';
            nameDiv.textContent = eng.display_name || eng.engine_id;
            
            const statDiv = document.createElement('div');
            statDiv.className = 'engine-status';
            if (eng.installed) {
                statDiv.textContent = (eng.status && eng.status.state) ? eng.status.state : 'unloaded';
                if (eng.status && eng.status.load_error) statDiv.textContent += ' — ' + eng.status.load_error;
            } else {
                statDiv.textContent = (eng.status && eng.status.install_hint) || eng.install_hint || 'Not installed';
            }
            
            label.appendChild(cb);
            label.appendChild(nameDiv);
            label.appendChild(statDiv);
            bar.appendChild(label);
            
            if (cb.checked) label.classList.add('selected');
        });
        
        if (!hasSelected) {
            const firstAvail = bar.querySelector('input:not([disabled])');
            if (firstAvail) {
                firstAvail.checked = true;
                firstAvail.parentElement.classList.add('selected');
            }
        }
        updateStreamButtonState();
    }

    function getSelectedEngines() {
        const checked = Array.from(document.querySelectorAll('#engines-bar input:checked'));
        return checked.map(function(cb) { return cb.value; });
    }

    function updateStreamButtonState() {
        const btn = document.getElementById('stream-btn');
        if (!btn) return;
        
        const checked = Array.from(document.querySelectorAll('#engines-bar input:checked'));
        if (checked.length !== 1) {
            btn.disabled = true;
            btn.title = 'Select exactly one engine to stream';
            return;
        }
        
        try {
            const caps = JSON.parse(checked[0].dataset.capabilities || '{}');
            if (caps.supports_streaming) {
                btn.disabled = false;
                btn.title = '';
            } else {
                btn.disabled = true;
                btn.title = 'Selected engine does not support streaming';
            }
        } catch (e) {
            btn.disabled = true;
        }
    }

    function renderJob(job, container) {
        container.innerHTML = '';
        
        const card = document.createElement('div');
        card.className = 'job-card';
        
        const header = document.createElement('div');
        header.className = 'job-header';
        
        const title = document.createElement('div');
        title.className = 'job-title';
        title.textContent = job.job_name;
        
        const status = document.createElement('div');
        status.className = 'job-status';
        status.dataset.status = job.status;
        status.textContent = job.status;
        
        header.appendChild(title);
        header.appendChild(status);
        card.appendChild(header);
        
        if (job.results) {
            const resultsArray = Array.isArray(job.results) ? job.results : Object.values(job.results);
            resultsArray.forEach(function(res) {
                const resDiv = document.createElement('div');
                resDiv.className = 'engine-result';
                
                const rHead = document.createElement('div');
                rHead.className = 'engine-result-header';
                
                const rName = document.createElement('div');
                rName.className = 'engine-result-name';
                rName.textContent = res.display_name || res.engine_id || 'Unknown Engine';
                
                const rStat = document.createElement('div');
                rStat.className = 'engine-result-status';
                let statText = res.status;
                if (res.elapsed_s) statText += ' (' + res.elapsed_s.toFixed(1) + 's)';
                rStat.textContent = statText;
                
                rHead.appendChild(rName);
                rHead.appendChild(rStat);
                resDiv.appendChild(rHead);
                
                if (res.error) {
                    const errDiv = document.createElement('div');
                    errDiv.className = 'engine-result-error';
                    errDiv.textContent = 'Error: ' + res.error;
                    resDiv.appendChild(errDiv);
                }
                if (res.capability_notes && res.capability_notes.length) {
                    const fbDiv = document.createElement('div');
                    fbDiv.className = 'engine-result-notes';
                    fbDiv.textContent = 'Note: ' + res.capability_notes.join(' ');
                    resDiv.appendChild(fbDiv);
                }

                const meta = document.createElement('div');
                meta.className = 'engine-result-meta';
                const metaParts = [];
                if (res.file) metaParts.push(res.file);
                if (res.duration_s) metaParts.push(res.duration_s.toFixed(2) + 's audio');
                if (res.sample_rate) metaParts.push(res.sample_rate + 'Hz');
                meta.textContent = metaParts.join(' — ');
                if (metaParts.length) resDiv.appendChild(meta);
                
                if (res.url) {
                    const audio = document.createElement('audio');
                    audio.src = res.url;
                    audio.controls = true;
                    resDiv.appendChild(audio);
                }
                
                card.appendChild(resDiv);
            });
        }
        
        container.appendChild(card);
    }

    function loadJobsHistory() {
        return API.get('/api/jobs').then(function(data) {
            const list = document.getElementById('job-history-list');
            if (!list) return;
            list.innerHTML = '';
            
            const jobs = Array.isArray(data) ? data : Object.values(data);
            jobs.sort(function(a, b) { return (b.created_at || '').localeCompare(a.created_at || ''); });
            jobs.slice(0, 25).forEach(function(job) {
                API.get('/api/jobs/' + job.job_name).then(function(detail) {
                    const container = document.createElement('div');
                    renderJob(detail, container);
                    list.appendChild(container.firstElementChild);
                }).catch(function() {
                    const container = document.createElement('div');
                    renderJob(job, container);
                    list.appendChild(container.firstElementChild);
                });
            });
        }).catch(function(e) {
            console.error('loadJobsHistory failed:', e);
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

    function startJobPolling(jobName) {
        stopJobPolling();
        activeJobName = jobName;
        document.getElementById('active-job-container').style.display = '';
        let btn = document.getElementById('generate-btn');
        let sawJobMetadata = false;
        
        function poll() {
            API.get('/api/jobs/' + jobName).then(function(job) {
                sawJobMetadata = true;
                renderJob(job, document.getElementById('active-job-container'));
                
                if (job.status === 'completed' || job.status === 'completed_with_errors' || job.status === 'failed') {
                    window.clearInterval(activeJobPollInterval);
                    btn.disabled = false;
                    btn.textContent = 'Generate';
                    isGenerating = false;
                    setStatus('ready', 'Job ' + job.status);
                    loadJobsHistory();
                }
            }).catch(function(e) {
                if (sawJobMetadata) console.error('Poll error', e);
            });
        }
        
        window.setTimeout(poll, 250);
        activeJobPollInterval = window.setInterval(function() {
            loadEngines();
            poll();
        }, 2000);
    }

    function stopJobPolling() {
        if (activeJobPollInterval) window.clearInterval(activeJobPollInterval);
        activeJobPollInterval = null;
        activeJobName = null;
    }

    function onGenerateFormSubmit(event) {
        event.preventDefault();
        if (isGenerating) return;
        isGenerating = true;

        var btn = document.getElementById('generate-btn');
        btn.disabled = true;
        btn.textContent = 'Generating...';
        setStatus('loading', 'Submitting job...');

        const engine_ids = getSelectedEngines();
        if (engine_ids.length === 0) {
            alert('Please select at least one engine.');
            isGenerating = false;
            btn.disabled = false;
            btn.textContent = 'Generate';
            setStatus('error', 'No engine selected');
            return;
        }

        var params = {
            job_name: document.getElementById('job_name').value || generateJobName(),
            engine_ids: engine_ids,
            text: document.getElementById('text').value,
            voice_description: document.getElementById('voice_description').value || null,
            reference_wav_path: selectedReference ? selectedReference.path : null,
            reference_text: document.getElementById('reference_text').value || null,
            cfg_value: parseFloat(document.getElementById('cfg_value').value),
            inference_timesteps: parseInt(document.getElementById('inference_timesteps').value, 10),
            normalize: document.getElementById('normalize').checked,
            attempts: parseInt(document.getElementById('attempts').value, 10),
            seed: document.getElementById('seed').value
                ? parseInt(document.getElementById('seed').value, 10)
                : null,
        };

        const jobName = params.job_name;
        renderJob({ job_name: jobName, status: 'in_progress', results: [] }, document.getElementById('active-job-container'));
        startJobPolling(jobName);
        API.postJSON('/api/jobs', params).then(function(jobData) {
            document.getElementById('job_name').value = generateJobName();
            if (jobData && jobData.job_name && jobData.job_name !== jobName) startJobPolling(jobData.job_name);
        }).catch(function(e) {
            console.error(e);
            setStatus('error', 'Job error: ' + e.message);
            stopJobPolling();
            document.getElementById('active-job-container').style.display = 'none';
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
        var streamSampleRate = 48000;

        var ws = new WebSocket('ws://' + location.host + '/api/generate/stream');

        var selected = getSelectedEngines();

        ws.onopen = function() {
            ws.send(JSON.stringify({
                type: 'start',
                params: {
                    engine_id: selected[0] || 'voxcpm',
                    text: text,
                    voice_description: voiceDescription,
                    reference_wav_path: selectedReference ? selectedReference.path : null,
                    reference_text: document.getElementById('reference_text').value || null,
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
                    player.ensureContext(streamSampleRate);
                    player.enqueueChunk(float32);
                    progressDiv.textContent = 'Streaming... chunk #' + player.receivedChunks;
                });
                return;
            }

            var msg = JSON.parse(event.data);
            if (msg.type === 'meta') {
                streamSampleRate = msg.sample_rate || streamSampleRate;
                player.ensureContext(streamSampleRate);
            } else if (msg.type === 'progress') {
                progressDiv.textContent = 'Chunk ' + player.receivedChunks + ' — ' +
                    (msg.chunk_samples / (player.audioCtx ? player.audioCtx.sampleRate : streamSampleRate)).toFixed(2) + 's';
            } else if (msg.type === 'saved') {
                finalUrl = msg.url;
            } else if (msg.type === 'done') {
                progressDiv.textContent = 'Done!';
                let audioEl = document.getElementById('player');
                if (finalUrl) {
                    audioEl.src = finalUrl;
                    audioEl.style.display = '';
                }
                loadJobsHistory();
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
        document.getElementById('job_name').value = generateJobName();
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
            let resp, data;
            var file = evt.target.files[0];
            if (!file) {
                selectedReference = null;
                document.getElementById('selected-reference').style.display = 'none';
                return;
            }
            var formData = new FormData();
            formData.append('reference', file);
            try {
                resp = await fetch('/api/uploads', { method: 'POST', body: formData });
                data = await resp.json();
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
        loadEngines().then(function() {
            loadJobsHistory();
        });
        window.setInterval(loadEngines, 5000);
        loadUploads();
    });

})();
