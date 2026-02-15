// =============================================================================
//  Phone Mirror — OpenClaw Empire Dashboard v2
//  Real-time phone screenshot stream with click-to-interact overlay.
// =============================================================================

'use strict';

// ---------------------------------------------------------------------------
//  PhoneMirror
// ---------------------------------------------------------------------------

function PhoneMirror() {
    // Canvas state
    this._container = null;
    this._canvas = null;
    this._ctx = null;
    this._overlayCanvas = null;
    this._overlayCtx = null;

    // Device state
    this._activeDeviceId = null;
    this._devices = [];
    this._streaming = false;
    this._fps = 1;
    this._fpsActual = 0;
    this._frameCount = 0;
    this._fpsTimer = null;
    this._lastScreenshot = null;
    this._lastFrameTime = 0;

    // Phone dimensions (default portrait)
    this._phoneWidth = 1080;
    this._phoneHeight = 1920;

    // Device info overlay
    this._deviceInfo = {
        name: '--',
        battery: null,
        wifi: null,
        status: 'disconnected'
    };

    // Interaction state
    this._swipeStart = null;
    this._isSwipe = false;
    this._controlsVisible = false;
    this._animationFrame = null;

    // Control button layout
    this._controls = {
        back: { x: 0, y: 0, w: 0, h: 0, label: 'Back', icon: '\u25C0' },
        home: { x: 0, y: 0, w: 0, h: 0, label: 'Home', icon: '\u25CB' },
        recent: { x: 0, y: 0, w: 0, h: 0, label: 'Recent', icon: '\u25A1' },
        screenshot: { x: 0, y: 0, w: 0, h: 0, label: 'Capture', icon: '\uD83D\uDCF7' }
    };

    // Dropdown state
    this._dropdownOpen = false;
}

// ---- Public: Initialization -----------------------------------------------

/**
 * Initialize the phone mirror into the given container element.
 * Creates the canvas, overlay, and control elements.
 */
PhoneMirror.prototype.init = function init(containerId) {
    this._container = document.getElementById(containerId);
    if (!this._container) {
        console.error('[PhoneMirror] Container #' + containerId + ' not found');
        return;
    }

    // Clear existing content
    this._container.innerHTML = '';
    this._container.style.position = 'relative';
    this._container.style.overflow = 'hidden';
    this._container.style.background = '#000';
    this._container.style.borderRadius = '12px';
    this._container.style.userSelect = 'none';

    // Main canvas for screenshots
    this._canvas = document.createElement('canvas');
    this._canvas.style.display = 'block';
    this._canvas.style.width = '100%';
    this._canvas.style.height = '100%';
    this._canvas.style.objectFit = 'contain';
    this._container.appendChild(this._canvas);
    this._ctx = this._canvas.getContext('2d');

    // Overlay canvas for controls and interaction feedback
    this._overlayCanvas = document.createElement('canvas');
    this._overlayCanvas.style.position = 'absolute';
    this._overlayCanvas.style.top = '0';
    this._overlayCanvas.style.left = '0';
    this._overlayCanvas.style.width = '100%';
    this._overlayCanvas.style.height = '100%';
    this._overlayCanvas.style.cursor = 'crosshair';
    this._container.appendChild(this._overlayCanvas);
    this._overlayCtx = this._overlayCanvas.getContext('2d');

    // Build the info bar
    this._buildInfoBar();

    // Build the device selector dropdown
    this._buildDeviceSelector();

    // Bind events
    this._bindEvents();

    // Size canvases
    this._resize();

    // Render initial state
    this._renderPlaceholder();

    // Start FPS counter
    this._startFPSCounter();

    // Listen for WebSocket phone mirror events
    if (typeof wsManager !== 'undefined') {
        var self = this;
        wsManager.on('phone_mirror', function (data) {
            if (data.device_id === self._activeDeviceId) {
                self.updateFrame(data.screenshot_base64);
                if (data.battery !== null && data.battery !== undefined) {
                    self._deviceInfo.battery = data.battery;
                }
                if (data.wifi !== null && data.wifi !== undefined) {
                    self._deviceInfo.wifi = data.wifi;
                }
                self._renderOverlay();
            }
        });

        wsManager.on('device_update', function (data) {
            if (data.devices) {
                self._devices = data.devices;
                self._updateDeviceSelector();
            }
        });
    }
};

// ---- Public: Stream control -----------------------------------------------

/**
 * Start streaming screenshots from the given device.
 */
PhoneMirror.prototype.startStream = function startStream(deviceId) {
    if (!deviceId) {
        console.warn('[PhoneMirror] No device ID provided');
        return;
    }

    this._activeDeviceId = deviceId;
    this._streaming = true;
    this._deviceInfo.status = 'connecting';
    this._renderOverlay();

    // Subscribe to phone mirror channel via WebSocket
    if (typeof wsManager !== 'undefined') {
        wsManager.subscribe('/ws/phone-mirror/' + deviceId);
        wsManager.send('phone_mirror_start', {
            device_id: deviceId,
            fps: this._fps
        });
    }

    this._deviceInfo.status = 'streaming';
    this._renderOverlay();

    console.log('[PhoneMirror] Started stream for device: ' + deviceId);
};

/**
 * Stop the current screenshot stream.
 */
PhoneMirror.prototype.stopStream = function stopStream() {
    if (!this._streaming) {
        return;
    }

    this._streaming = false;
    this._deviceInfo.status = 'disconnected';

    if (typeof wsManager !== 'undefined' && this._activeDeviceId) {
        wsManager.unsubscribe('/ws/phone-mirror/' + this._activeDeviceId);
        wsManager.send('phone_mirror_stop', {
            device_id: this._activeDeviceId
        });
    }

    this._renderOverlay();
    console.log('[PhoneMirror] Stopped stream');
};

/**
 * Update the canvas with a new screenshot frame (base64-encoded image).
 */
PhoneMirror.prototype.updateFrame = function updateFrame(base64Image) {
    if (!base64Image || !this._ctx) {
        return;
    }

    var self = this;
    var img = new Image();

    img.onload = function () {
        self._phoneWidth = img.naturalWidth || 1080;
        self._phoneHeight = img.naturalHeight || 1920;

        // Resize canvas to match phone aspect ratio
        self._resize();

        // Draw the screenshot
        self._ctx.clearRect(0, 0, self._canvas.width, self._canvas.height);
        self._ctx.drawImage(img, 0, 0, self._canvas.width, self._canvas.height);

        self._lastScreenshot = base64Image;
        self._lastFrameTime = Date.now();
        self._frameCount++;
    };

    img.onerror = function () {
        console.error('[PhoneMirror] Failed to load frame image');
    };

    // Handle both raw base64 and data URI prefix
    if (base64Image.indexOf('data:') === 0) {
        img.src = base64Image;
    } else {
        img.src = 'data:image/png;base64,' + base64Image;
    }
};

// ---- Public: Interaction --------------------------------------------------

/**
 * Send a tap command at the given screen coordinates.
 * Coordinates are in canvas space and get mapped to phone resolution.
 */
PhoneMirror.prototype.onClick = function onClick(canvasX, canvasY) {
    if (!this._activeDeviceId || !this._streaming) {
        return;
    }

    var coords = this._canvasToPhone(canvasX, canvasY);

    // Visual tap feedback
    this._showTapFeedback(canvasX, canvasY);

    // Send tap command
    this._sendCommand('tap', {
        x: coords.x,
        y: coords.y
    });
};

/**
 * Send a swipe command between two canvas coordinates.
 */
PhoneMirror.prototype.onSwipe = function onSwipe(startX, startY, endX, endY) {
    if (!this._activeDeviceId || !this._streaming) {
        return;
    }

    var start = this._canvasToPhone(startX, startY);
    var end = this._canvasToPhone(endX, endY);

    // Visual swipe feedback
    this._showSwipeFeedback(startX, startY, endX, endY);

    this._sendCommand('swipe', {
        start_x: start.x,
        start_y: start.y,
        end_x: end.x,
        end_y: end.y,
        duration: 300
    });
};

// ---- Public: Device management --------------------------------------------

PhoneMirror.prototype.setDevice = function setDevice(deviceId) {
    if (this._streaming) {
        this.stopStream();
    }
    this._activeDeviceId = deviceId;

    var device = this._findDevice(deviceId);
    if (device) {
        this._deviceInfo.name = device.name || device.id || deviceId;
    } else {
        this._deviceInfo.name = deviceId;
    }

    this._renderOverlay();
};

PhoneMirror.prototype.getDeviceList = function getDeviceList() {
    return this._devices.slice();
};

PhoneMirror.prototype.showControls = function showControls() {
    this._controlsVisible = true;
    this._renderOverlay();
};

PhoneMirror.prototype.hideControls = function hideControls() {
    this._controlsVisible = false;
    this._renderOverlay();
};

PhoneMirror.prototype.setFPS = function setFPS(fps) {
    this._fps = Math.max(0.5, Math.min(fps, 30));
    if (this._streaming && this._activeDeviceId) {
        if (typeof wsManager !== 'undefined') {
            wsManager.send('phone_mirror_fps', {
                device_id: this._activeDeviceId,
                fps: this._fps
            });
        }
    }
};

PhoneMirror.prototype.isStreaming = function isStreaming() {
    return this._streaming;
};

PhoneMirror.prototype.getLastScreenshot = function getLastScreenshot() {
    return this._lastScreenshot;
};

// ---- Internal: UI building ------------------------------------------------

PhoneMirror.prototype._buildInfoBar = function _buildInfoBar() {
    var bar = document.createElement('div');
    bar.id = 'phone-mirror-info';
    bar.style.cssText = 'position:absolute;top:0;left:0;right:0;display:flex;justify-content:space-between;' +
        'align-items:center;padding:6px 10px;background:rgba(0,0,0,0.7);color:#e2e8f0;font-size:11px;' +
        'font-family:"JetBrains Mono",monospace;z-index:10;backdrop-filter:blur(4px);';

    bar.innerHTML =
        '<span id="pm-device-name">--</span>' +
        '<span id="pm-status-bar">' +
            '<span id="pm-fps-counter" style="margin-right:8px;">0 FPS</span>' +
            '<span id="pm-battery" style="margin-right:8px;"></span>' +
            '<span id="pm-wifi"></span>' +
        '</span>';

    this._container.appendChild(bar);
};

PhoneMirror.prototype._buildDeviceSelector = function _buildDeviceSelector() {
    var wrapper = document.createElement('div');
    wrapper.id = 'pm-device-selector';
    wrapper.style.cssText = 'position:absolute;top:30px;left:8px;z-index:15;';

    var btn = document.createElement('button');
    btn.id = 'pm-device-btn';
    btn.style.cssText = 'background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#e2e8f0;' +
        'padding:4px 10px;border-radius:6px;font-size:11px;cursor:pointer;font-family:inherit;';
    btn.textContent = 'Select Device';

    var dropdown = document.createElement('div');
    dropdown.id = 'pm-device-dropdown';
    dropdown.style.cssText = 'display:none;position:absolute;top:100%;left:0;min-width:180px;margin-top:4px;' +
        'background:#12121a;border:1px solid #1e1e2e;border-radius:8px;overflow:hidden;' +
        'box-shadow:0 8px 24px rgba(0,0,0,0.4);';

    wrapper.appendChild(btn);
    wrapper.appendChild(dropdown);
    this._container.appendChild(wrapper);

    var self = this;
    btn.addEventListener('click', function (e) {
        e.stopPropagation();
        self._dropdownOpen = !self._dropdownOpen;
        dropdown.style.display = self._dropdownOpen ? 'block' : 'none';
    });

    // Close dropdown when clicking elsewhere
    document.addEventListener('click', function () {
        self._dropdownOpen = false;
        dropdown.style.display = 'none';
    });
};

PhoneMirror.prototype._updateDeviceSelector = function _updateDeviceSelector() {
    var dropdown = document.getElementById('pm-device-dropdown');
    if (!dropdown) {
        return;
    }

    var self = this;
    if (this._devices.length === 0) {
        dropdown.innerHTML = '<div style="padding:10px;color:#94a3b8;font-size:11px;">No devices available</div>';
        return;
    }

    dropdown.innerHTML = this._devices.map(function (device) {
        var id = device.id || device.device_id;
        var name = device.name || id;
        var active = id === self._activeDeviceId;
        var statusColor = device.online ? '#22c55e' : '#ef4444';
        return '<div class="pm-device-option" data-device-id="' + id + '" ' +
            'style="padding:8px 12px;cursor:pointer;display:flex;align-items:center;gap:8px;' +
            'font-size:12px;border-bottom:1px solid rgba(30,30,46,0.5);' +
            (active ? 'background:rgba(99,102,241,0.15);' : '') +
            '" onmouseover="this.style.background=\'rgba(99,102,241,0.1)\'" ' +
            'onmouseout="this.style.background=\'' + (active ? 'rgba(99,102,241,0.15)' : 'transparent') + '\'">' +
            '<span style="width:6px;height:6px;border-radius:50%;background:' + statusColor + ';flex-shrink:0;"></span>' +
            '<span style="color:#e2e8f0;">' + name + '</span>' +
            (active ? '<span style="margin-left:auto;color:#6366f1;font-size:10px;">ACTIVE</span>' : '') +
            '</div>';
    }).join('');

    // Bind click handlers
    var options = dropdown.querySelectorAll('.pm-device-option');
    for (var i = 0; i < options.length; i++) {
        options[i].addEventListener('click', function (e) {
            e.stopPropagation();
            var deviceId = this.getAttribute('data-device-id');
            self.setDevice(deviceId);
            self.startStream(deviceId);
            self._dropdownOpen = false;
            dropdown.style.display = 'none';
        });
    }
};

// ---- Internal: Event binding ----------------------------------------------

PhoneMirror.prototype._bindEvents = function _bindEvents() {
    var self = this;
    var overlay = this._overlayCanvas;

    // Mouse / touch events for tap and swipe
    overlay.addEventListener('mousedown', function (e) {
        if (self._isControlHit(e)) {
            return;
        }
        var pos = self._getCanvasPos(e);
        self._swipeStart = { x: pos.x, y: pos.y, time: Date.now() };
        self._isSwipe = false;
    });

    overlay.addEventListener('mousemove', function (e) {
        if (!self._swipeStart) {
            return;
        }
        var pos = self._getCanvasPos(e);
        var dx = pos.x - self._swipeStart.x;
        var dy = pos.y - self._swipeStart.y;
        if (Math.sqrt(dx * dx + dy * dy) > 10) {
            self._isSwipe = true;
        }
    });

    overlay.addEventListener('mouseup', function (e) {
        if (!self._swipeStart) {
            return;
        }
        var pos = self._getCanvasPos(e);
        var elapsed = Date.now() - self._swipeStart.time;

        if (self._isSwipe) {
            self.onSwipe(self._swipeStart.x, self._swipeStart.y, pos.x, pos.y);
        } else if (elapsed < 500) {
            self.onClick(pos.x, pos.y);
        }

        self._swipeStart = null;
        self._isSwipe = false;
    });

    // Touch events
    overlay.addEventListener('touchstart', function (e) {
        e.preventDefault();
        if (e.touches.length !== 1) return;
        var pos = self._getTouchPos(e.touches[0]);
        self._swipeStart = { x: pos.x, y: pos.y, time: Date.now() };
        self._isSwipe = false;
    }, { passive: false });

    overlay.addEventListener('touchmove', function (e) {
        e.preventDefault();
        if (!self._swipeStart || e.touches.length !== 1) return;
        var pos = self._getTouchPos(e.touches[0]);
        var dx = pos.x - self._swipeStart.x;
        var dy = pos.y - self._swipeStart.y;
        if (Math.sqrt(dx * dx + dy * dy) > 10) {
            self._isSwipe = true;
        }
    }, { passive: false });

    overlay.addEventListener('touchend', function (e) {
        e.preventDefault();
        if (!self._swipeStart) return;
        var pos;
        if (e.changedTouches && e.changedTouches.length > 0) {
            pos = self._getTouchPos(e.changedTouches[0]);
        } else {
            pos = self._swipeStart;
        }
        if (self._isSwipe) {
            self.onSwipe(self._swipeStart.x, self._swipeStart.y, pos.x, pos.y);
        } else {
            self.onClick(pos.x, pos.y);
        }
        self._swipeStart = null;
        self._isSwipe = false;
    }, { passive: false });

    // Resize
    window.addEventListener('resize', function () {
        self._resize();
        self._renderOverlay();
    });
};

// ---- Internal: Rendering --------------------------------------------------

PhoneMirror.prototype._resize = function _resize() {
    if (!this._container || !this._canvas) {
        return;
    }

    var containerW = this._container.clientWidth;
    var containerH = this._container.clientHeight;
    if (containerW === 0 || containerH === 0) {
        return;
    }

    // Maintain 9:16 phone aspect ratio
    var phoneAspect = this._phoneWidth / this._phoneHeight;
    var containerAspect = containerW / containerH;
    var drawW, drawH;

    if (containerAspect > phoneAspect) {
        drawH = containerH;
        drawW = drawH * phoneAspect;
    } else {
        drawW = containerW;
        drawH = drawW / phoneAspect;
    }

    // Set canvas pixel dimensions
    var dpr = window.devicePixelRatio || 1;
    this._canvas.width = Math.round(drawW * dpr);
    this._canvas.height = Math.round(drawH * dpr);
    this._canvas.style.width = Math.round(drawW) + 'px';
    this._canvas.style.height = Math.round(drawH) + 'px';

    this._overlayCanvas.width = Math.round(drawW * dpr);
    this._overlayCanvas.height = Math.round(drawH * dpr);
    this._overlayCanvas.style.width = Math.round(drawW) + 'px';
    this._overlayCanvas.style.height = Math.round(drawH) + 'px';

    // Center in container
    var offsetX = Math.round((containerW - drawW) / 2);
    var offsetY = Math.round((containerH - drawH) / 2);
    this._canvas.style.position = 'absolute';
    this._canvas.style.left = offsetX + 'px';
    this._canvas.style.top = offsetY + 'px';
    this._overlayCanvas.style.left = offsetX + 'px';
    this._overlayCanvas.style.top = offsetY + 'px';

    // Update control button positions
    this._updateControlPositions(drawW, drawH, dpr);
};

PhoneMirror.prototype._updateControlPositions = function _updateControlPositions(w, h, dpr) {
    var btnW = 48 * dpr;
    var btnH = 36 * dpr;
    var margin = 8 * dpr;
    var bottomY = h * dpr - btnH - margin;

    // Navigation buttons at bottom
    this._controls.back.x = w * dpr * 0.15 - btnW / 2;
    this._controls.back.y = bottomY;
    this._controls.back.w = btnW;
    this._controls.back.h = btnH;

    this._controls.home.x = w * dpr * 0.5 - btnW / 2;
    this._controls.home.y = bottomY;
    this._controls.home.w = btnW;
    this._controls.home.h = btnH;

    this._controls.recent.x = w * dpr * 0.85 - btnW / 2;
    this._controls.recent.y = bottomY;
    this._controls.recent.w = btnW;
    this._controls.recent.h = btnH;

    // Screenshot button at top right
    this._controls.screenshot.x = w * dpr - btnW - margin;
    this._controls.screenshot.y = margin + 28 * dpr;
    this._controls.screenshot.w = btnW;
    this._controls.screenshot.h = btnH;
};

PhoneMirror.prototype._renderPlaceholder = function _renderPlaceholder() {
    if (!this._ctx) return;

    var w = this._canvas.width;
    var h = this._canvas.height;
    var dpr = window.devicePixelRatio || 1;

    this._ctx.fillStyle = '#0a0a0f';
    this._ctx.fillRect(0, 0, w, h);

    // Phone outline
    this._ctx.strokeStyle = '#1e1e2e';
    this._ctx.lineWidth = 2 * dpr;
    this._ctx.roundRect(4 * dpr, 4 * dpr, w - 8 * dpr, h - 8 * dpr, 16 * dpr);
    this._ctx.stroke();

    // Center text
    this._ctx.fillStyle = '#475569';
    this._ctx.font = (14 * dpr) + 'px "Inter", sans-serif';
    this._ctx.textAlign = 'center';
    this._ctx.fillText('No Device Connected', w / 2, h / 2 - 12 * dpr);

    this._ctx.fillStyle = '#334155';
    this._ctx.font = (11 * dpr) + 'px "Inter", sans-serif';
    this._ctx.fillText('Select a device to start mirroring', w / 2, h / 2 + 12 * dpr);
};

PhoneMirror.prototype._renderOverlay = function _renderOverlay() {
    if (!this._overlayCtx) return;

    var ctx = this._overlayCtx;
    var w = this._overlayCanvas.width;
    var h = this._overlayCanvas.height;
    var dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    // Update info bar text
    var nameEl = document.getElementById('pm-device-name');
    if (nameEl) {
        nameEl.textContent = this._deviceInfo.name || '--';
    }
    var fpsEl = document.getElementById('pm-fps-counter');
    if (fpsEl) {
        fpsEl.textContent = this._fpsActual + ' FPS';
    }
    var battEl = document.getElementById('pm-battery');
    if (battEl) {
        if (this._deviceInfo.battery !== null && this._deviceInfo.battery !== undefined) {
            battEl.textContent = this._deviceInfo.battery + '%';
            battEl.style.color = this._deviceInfo.battery > 20 ? '#22c55e' : '#ef4444';
        } else {
            battEl.textContent = '';
        }
    }
    var wifiEl = document.getElementById('pm-wifi');
    if (wifiEl) {
        if (this._deviceInfo.wifi !== null && this._deviceInfo.wifi !== undefined) {
            wifiEl.textContent = this._deviceInfo.wifi ? 'WiFi' : 'Data';
        } else {
            wifiEl.textContent = '';
        }
    }

    // Draw control buttons if visible
    if (this._controlsVisible) {
        this._renderControlButtons(ctx, dpr);
    }

    // Status indicator dot
    var dotRadius = 4 * dpr;
    var dotX = w - 12 * dpr;
    var dotY = 16 * dpr;

    ctx.beginPath();
    ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
    switch (this._deviceInfo.status) {
        case 'streaming':
            ctx.fillStyle = '#22c55e';
            break;
        case 'connecting':
            ctx.fillStyle = '#f59e0b';
            break;
        default:
            ctx.fillStyle = '#ef4444';
            break;
    }
    ctx.fill();
};

PhoneMirror.prototype._renderControlButtons = function _renderControlButtons(ctx, dpr) {
    var keys = ['back', 'home', 'recent', 'screenshot'];

    for (var i = 0; i < keys.length; i++) {
        var ctrl = this._controls[keys[i]];

        // Button background
        ctx.fillStyle = 'rgba(18, 18, 26, 0.8)';
        ctx.beginPath();
        var r = 6 * dpr;
        var x = ctrl.x;
        var y = ctrl.y;
        var bw = ctrl.w;
        var bh = ctrl.h;

        ctx.moveTo(x + r, y);
        ctx.lineTo(x + bw - r, y);
        ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
        ctx.lineTo(x + bw, y + bh - r);
        ctx.quadraticCurveTo(x + bw, y + bh, x + bw - r, y + bh);
        ctx.lineTo(x + r, y + bh);
        ctx.quadraticCurveTo(x, y + bh, x, y + bh - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        ctx.fill();

        // Border
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
        ctx.lineWidth = 1 * dpr;
        ctx.stroke();

        // Icon text
        ctx.fillStyle = '#e2e8f0';
        ctx.font = (12 * dpr) + 'px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(ctrl.icon, x + bw / 2, y + bh / 2);
    }
};

// ---- Internal: Visual feedback --------------------------------------------

PhoneMirror.prototype._showTapFeedback = function _showTapFeedback(x, y) {
    var ctx = this._overlayCtx;
    if (!ctx) return;

    var dpr = window.devicePixelRatio || 1;
    var cx = x * dpr;
    var cy = y * dpr;
    var maxRadius = 24 * dpr;
    var self = this;
    var start = performance.now();
    var duration = 300;

    function animate(now) {
        var elapsed = now - start;
        var progress = Math.min(elapsed / duration, 1);

        // Redraw overlay
        self._renderOverlay();

        var radius = maxRadius * progress;
        var alpha = 1 - progress;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(99, 102, 241, ' + (alpha * 0.4) + ')';
        ctx.fill();

        ctx.beginPath();
        ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(99, 102, 241, ' + (alpha * 0.7) + ')';
        ctx.fill();

        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
};

PhoneMirror.prototype._showSwipeFeedback = function _showSwipeFeedback(sx, sy, ex, ey) {
    var ctx = this._overlayCtx;
    if (!ctx) return;

    var dpr = window.devicePixelRatio || 1;
    var self = this;
    var start = performance.now();
    var duration = 400;

    function animate(now) {
        var elapsed = now - start;
        var progress = Math.min(elapsed / duration, 1);

        self._renderOverlay();

        var alpha = 1 - progress;
        ctx.beginPath();
        ctx.moveTo(sx * dpr, sy * dpr);

        // Animate along the path
        var curX = sx + (ex - sx) * progress;
        var curY = sy + (ey - sy) * progress;
        ctx.lineTo(curX * dpr, curY * dpr);

        ctx.strokeStyle = 'rgba(99, 102, 241, ' + alpha + ')';
        ctx.lineWidth = 3 * dpr;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Dot at current position
        ctx.beginPath();
        ctx.arc(curX * dpr, curY * dpr, 6 * dpr, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(99, 102, 241, ' + (alpha * 0.8) + ')';
        ctx.fill();

        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }

    requestAnimationFrame(animate);
};

// ---- Internal: Coordinate mapping -----------------------------------------

PhoneMirror.prototype._canvasToPhone = function _canvasToPhone(canvasX, canvasY) {
    var scaleX = this._phoneWidth / (this._canvas.width / (window.devicePixelRatio || 1));
    var scaleY = this._phoneHeight / (this._canvas.height / (window.devicePixelRatio || 1));
    return {
        x: Math.round(canvasX * scaleX),
        y: Math.round(canvasY * scaleY)
    };
};

PhoneMirror.prototype._getCanvasPos = function _getCanvasPos(e) {
    var rect = this._overlayCanvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
};

PhoneMirror.prototype._getTouchPos = function _getTouchPos(touch) {
    var rect = this._overlayCanvas.getBoundingClientRect();
    return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
    };
};

// ---- Internal: Control hit detection --------------------------------------

PhoneMirror.prototype._isControlHit = function _isControlHit(e) {
    if (!this._controlsVisible) {
        return false;
    }

    var pos = this._getCanvasPos(e);
    var dpr = window.devicePixelRatio || 1;
    var mx = pos.x * dpr;
    var my = pos.y * dpr;

    var keys = ['back', 'home', 'recent', 'screenshot'];
    for (var i = 0; i < keys.length; i++) {
        var ctrl = this._controls[keys[i]];
        if (mx >= ctrl.x && mx <= ctrl.x + ctrl.w && my >= ctrl.y && my <= ctrl.y + ctrl.h) {
            this._onControlClick(keys[i]);
            return true;
        }
    }
    return false;
};

PhoneMirror.prototype._onControlClick = function _onControlClick(controlName) {
    switch (controlName) {
        case 'back':
            this._sendCommand('keyevent', { key: 'KEYCODE_BACK' });
            break;
        case 'home':
            this._sendCommand('keyevent', { key: 'KEYCODE_HOME' });
            break;
        case 'recent':
            this._sendCommand('keyevent', { key: 'KEYCODE_APP_SWITCH' });
            break;
        case 'screenshot':
            this._takeScreenshot();
            break;
    }
};

// ---- Internal: Commands ---------------------------------------------------

PhoneMirror.prototype._sendCommand = function _sendCommand(command, params) {
    if (!this._activeDeviceId) {
        return;
    }

    var payload = {
        device_id: this._activeDeviceId,
        command: command,
        params: params || {}
    };

    if (typeof wsManager !== 'undefined') {
        wsManager.send('phone_command', payload);
    }

    // Also try REST API fallback
    try {
        var apiBase = (typeof API_BASE !== 'undefined') ? API_BASE : 'http://localhost:8765/api';
        fetch(apiBase + '/phone/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).catch(function () {
            // Silently fail — WebSocket is primary transport
        });
    } catch (err) {
        // Ignore
    }
};

PhoneMirror.prototype._takeScreenshot = function _takeScreenshot() {
    if (!this._lastScreenshot) {
        console.warn('[PhoneMirror] No screenshot available');
        return;
    }

    // Create download link
    var link = document.createElement('a');
    link.download = 'phone-screenshot-' + Date.now() + '.png';
    if (this._lastScreenshot.indexOf('data:') === 0) {
        link.href = this._lastScreenshot;
    } else {
        link.href = 'data:image/png;base64,' + this._lastScreenshot;
    }
    link.click();
};

// ---- Internal: FPS counter ------------------------------------------------

PhoneMirror.prototype._startFPSCounter = function _startFPSCounter() {
    var self = this;
    var lastCount = 0;

    this._fpsTimer = setInterval(function () {
        self._fpsActual = self._frameCount - lastCount;
        lastCount = self._frameCount;

        var fpsEl = document.getElementById('pm-fps-counter');
        if (fpsEl) {
            fpsEl.textContent = self._fpsActual + ' FPS';
        }
    }, 1000);
};

// ---- Internal: Helpers ----------------------------------------------------

PhoneMirror.prototype._findDevice = function _findDevice(deviceId) {
    for (var i = 0; i < this._devices.length; i++) {
        var d = this._devices[i];
        if ((d.id || d.device_id) === deviceId) {
            return d;
        }
    }
    return null;
};

// ---------------------------------------------------------------------------
//  Singleton
// ---------------------------------------------------------------------------

var phoneMirror = new PhoneMirror();
