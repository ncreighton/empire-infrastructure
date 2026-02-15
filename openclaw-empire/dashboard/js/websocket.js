// =============================================================================
//  WebSocket Manager — OpenClaw Empire Dashboard v2
//  Auto-reconnecting WebSocket client with event bus for real-time updates.
// =============================================================================

'use strict';

var WS_BASE = 'ws://localhost:8765';

// ---------------------------------------------------------------------------
//  EventBus — lightweight pub/sub used internally by WebSocketManager
// ---------------------------------------------------------------------------

function EventBus() {
    this._handlers = {};
}

EventBus.prototype.on = function on(event, callback) {
    if (typeof callback !== 'function') {
        return;
    }
    if (!this._handlers[event]) {
        this._handlers[event] = [];
    }
    // Prevent duplicate registrations of the same function reference
    if (this._handlers[event].indexOf(callback) === -1) {
        this._handlers[event].push(callback);
    }
};

EventBus.prototype.off = function off(event, callback) {
    if (!this._handlers[event]) {
        return;
    }
    if (!callback) {
        // Remove all handlers for this event
        delete this._handlers[event];
        return;
    }
    var idx = this._handlers[event].indexOf(callback);
    if (idx !== -1) {
        this._handlers[event].splice(idx, 1);
    }
    if (this._handlers[event].length === 0) {
        delete this._handlers[event];
    }
};

EventBus.prototype.emit = function emit(event, data) {
    var handlers = this._handlers[event];
    if (!handlers || handlers.length === 0) {
        return;
    }
    // Iterate over a shallow copy so handlers can safely remove themselves
    var snapshot = handlers.slice();
    for (var i = 0; i < snapshot.length; i++) {
        try {
            snapshot[i](data);
        } catch (err) {
            console.error('[EventBus] Handler error for event "' + event + '":', err);
        }
    }
};

EventBus.prototype.once = function once(event, callback) {
    var self = this;
    function wrapper(data) {
        self.off(event, wrapper);
        callback(data);
    }
    this.on(event, wrapper);
};

EventBus.prototype.listenerCount = function listenerCount(event) {
    if (!this._handlers[event]) {
        return 0;
    }
    return this._handlers[event].length;
};

EventBus.prototype.removeAllListeners = function removeAllListeners() {
    this._handlers = {};
};

// ---------------------------------------------------------------------------
//  WebSocketManager
// ---------------------------------------------------------------------------

// Connection state constants (mirror WebSocket readyState)
var WS_STATE = {
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3
};

function WebSocketManager() {
    this._bus = new EventBus();
    this._ws = null;
    this._url = '';
    this._reconnectAttempts = 0;
    this._maxReconnectAttempts = 50;
    this._reconnectTimer = null;
    this._pingTimer = null;
    this._pongTimer = null;
    this._manualClose = false;
    this._messageQueue = [];
    this._subscriptions = [
        '/ws/agent-status',
        '/ws/pipeline-progress',
        '/ws/activity-feed'
    ];
    this._lastPongTime = 0;
    this._connectionId = null;
    this._statusElementId = 'ws-status';
    this._connDotId = 'connDot';
    this._metrics = {
        messagesReceived: 0,
        messagesSent: 0,
        reconnects: 0,
        errors: 0,
        lastConnected: null,
        lastDisconnected: null,
        totalUptime: 0
    };
    this._uptimeStart = null;
}

// ---- Public: Event registration ------------------------------------------

WebSocketManager.prototype.on = function on(event, callback) {
    this._bus.on(event, callback);
};

WebSocketManager.prototype.off = function off(event, callback) {
    this._bus.off(event, callback);
};

WebSocketManager.prototype.once = function once(event, callback) {
    this._bus.once(event, callback);
};

// ---- Public: Connection lifecycle ----------------------------------------

/**
 * Establish a WebSocket connection to the given URL.
 * If no URL is provided, uses WS_BASE.
 */
WebSocketManager.prototype.connect = function connect(url) {
    if (this._ws && (this._ws.readyState === WS_STATE.CONNECTING || this._ws.readyState === WS_STATE.OPEN)) {
        console.warn('[WS] Already connected or connecting — ignoring connect()');
        return;
    }

    this._url = url || WS_BASE;
    this._manualClose = false;
    this._connectionId = this._generateId();

    this._updateStatus('connecting');
    this._bus.emit('connecting', { url: this._url, attempt: this._reconnectAttempts });

    try {
        this._ws = new WebSocket(this._url);
    } catch (err) {
        console.error('[WS] Failed to create WebSocket:', err);
        this._updateStatus('error');
        this._bus.emit('error', { error: err, phase: 'create' });
        this._scheduleReconnect();
        return;
    }

    this._bindSocketEvents();
};

/**
 * Reconnect with exponential backoff: 1s, 2s, 4s, 8s ... max 30s.
 */
WebSocketManager.prototype.reconnect = function reconnect() {
    if (this._manualClose) {
        return;
    }
    this._cleanup();
    this._reconnectAttempts++;
    this._metrics.reconnects++;

    var delay = this._getBackoffDelay();
    console.log('[WS] Reconnecting in ' + delay + 'ms (attempt ' + this._reconnectAttempts + ')');

    this._updateStatus('reconnecting');
    this._bus.emit('reconnecting', { attempt: this._reconnectAttempts, delay: delay });

    var self = this;
    this._reconnectTimer = setTimeout(function () {
        self._reconnectTimer = null;
        self.connect(self._url);
    }, delay);
};

/**
 * Graceful disconnect. Does not trigger auto-reconnect.
 */
WebSocketManager.prototype.close = function close() {
    this._manualClose = true;
    this._cleanup();

    if (this._ws) {
        try {
            this._ws.close(1000, 'Manual close');
        } catch (err) {
            // Ignore — socket might already be closed
        }
        this._ws = null;
    }

    this._updateStatus('closed');
    this._bus.emit('closed', { manual: true });
};

// ---- Public: Messaging ---------------------------------------------------

/**
 * Send a typed JSON message to the server.
 * If the socket is not open, the message is queued and sent on reconnect.
 */
WebSocketManager.prototype.send = function send(type, payload) {
    var message = {
        type: type,
        payload: payload || {},
        timestamp: new Date().toISOString(),
        connection_id: this._connectionId
    };

    if (!this._ws || this._ws.readyState !== WS_STATE.OPEN) {
        // Queue up to 100 messages while disconnected
        if (this._messageQueue.length < 100) {
            this._messageQueue.push(message);
        }
        return false;
    }

    try {
        this._ws.send(JSON.stringify(message));
        this._metrics.messagesSent++;
        return true;
    } catch (err) {
        console.error('[WS] Send error:', err);
        this._messageQueue.push(message);
        this._metrics.errors++;
        return false;
    }
};

/**
 * Send a raw string or object to the server.
 */
WebSocketManager.prototype.sendRaw = function sendRaw(data) {
    if (!this._ws || this._ws.readyState !== WS_STATE.OPEN) {
        return false;
    }
    try {
        var payload = typeof data === 'string' ? data : JSON.stringify(data);
        this._ws.send(payload);
        this._metrics.messagesSent++;
        return true;
    } catch (err) {
        console.error('[WS] Raw send error:', err);
        this._metrics.errors++;
        return false;
    }
};

// ---- Public: State queries -----------------------------------------------

/**
 * Returns the current connection state as a string.
 */
WebSocketManager.prototype.getState = function getState() {
    if (!this._ws) {
        return 'CLOSED';
    }
    switch (this._ws.readyState) {
        case WS_STATE.CONNECTING: return 'CONNECTING';
        case WS_STATE.OPEN:       return 'OPEN';
        case WS_STATE.CLOSING:    return 'CLOSING';
        case WS_STATE.CLOSED:     return 'CLOSED';
        default:                  return 'UNKNOWN';
    }
};

WebSocketManager.prototype.isConnected = function isConnected() {
    return this._ws !== null && this._ws.readyState === WS_STATE.OPEN;
};

WebSocketManager.prototype.getMetrics = function getMetrics() {
    var m = Object.assign({}, this._metrics);
    if (this._uptimeStart && this.isConnected()) {
        m.currentSessionUptime = Date.now() - this._uptimeStart;
    }
    m.queuedMessages = this._messageQueue.length;
    m.reconnectAttempts = this._reconnectAttempts;
    m.state = this.getState();
    return m;
};

WebSocketManager.prototype.getConnectionId = function getConnectionId() {
    return this._connectionId;
};

// ---- Public: Subscriptions -----------------------------------------------

/**
 * Subscribe to a server channel. Sends a subscribe message if connected.
 */
WebSocketManager.prototype.subscribe = function subscribe(channel) {
    if (this._subscriptions.indexOf(channel) === -1) {
        this._subscriptions.push(channel);
    }
    if (this.isConnected()) {
        this.send('subscribe', { channel: channel });
    }
};

/**
 * Unsubscribe from a server channel.
 */
WebSocketManager.prototype.unsubscribe = function unsubscribe(channel) {
    var idx = this._subscriptions.indexOf(channel);
    if (idx !== -1) {
        this._subscriptions.splice(idx, 1);
    }
    if (this.isConnected()) {
        this.send('unsubscribe', { channel: channel });
    }
};

// ---- Internal: Socket event binding --------------------------------------

WebSocketManager.prototype._bindSocketEvents = function _bindSocketEvents() {
    var self = this;

    this._ws.onopen = function () {
        console.log('[WS] Connected to ' + self._url);
        self._reconnectAttempts = 0;
        self._uptimeStart = Date.now();
        self._metrics.lastConnected = new Date().toISOString();

        self._updateStatus('connected');
        self._bus.emit('connected', { url: self._url, connectionId: self._connectionId });

        // Re-subscribe to all channels
        self._resubscribe();

        // Flush queued messages
        self._flushQueue();

        // Start heartbeat
        self._startPing();
    };

    this._ws.onmessage = function (event) {
        self._metrics.messagesReceived++;
        self._handleMessage(event.data);
    };

    this._ws.onerror = function (event) {
        console.error('[WS] Error:', event);
        self._metrics.errors++;
        self._updateStatus('error');
        self._bus.emit('error', { event: event, phase: 'runtime' });
    };

    this._ws.onclose = function (event) {
        var wasClean = event.wasClean;
        var code = event.code;
        var reason = event.reason || 'No reason';

        console.log('[WS] Closed: code=' + code + ' reason=' + reason + ' clean=' + wasClean);

        if (self._uptimeStart) {
            self._metrics.totalUptime += (Date.now() - self._uptimeStart);
            self._uptimeStart = null;
        }
        self._metrics.lastDisconnected = new Date().toISOString();

        self._stopPing();
        self._updateStatus('disconnected');
        self._bus.emit('disconnected', { code: code, reason: reason, wasClean: wasClean });

        if (!self._manualClose) {
            self._scheduleReconnect();
        }
    };
};

// ---- Internal: Message handling ------------------------------------------

WebSocketManager.prototype._handleMessage = function _handleMessage(raw) {
    var data;
    try {
        data = JSON.parse(raw);
    } catch (err) {
        // Not JSON — emit as raw
        this._bus.emit('raw_message', raw);
        return;
    }

    // Server pong response
    if (data.type === 'pong') {
        this._lastPongTime = Date.now();
        this._bus.emit('pong', data);
        return;
    }

    // Route known event types
    var eventType = data.type || data.event || 'unknown';

    switch (eventType) {
        case 'agent_status':
            this._handleAgentStatus(data.payload || data.data || data);
            break;

        case 'pipeline_progress':
            this._handlePipelineProgress(data.payload || data.data || data);
            break;

        case 'phone_mirror':
            this._handlePhoneMirror(data.payload || data.data || data);
            break;

        case 'activity':
            this._handleActivity(data.payload || data.data || data);
            break;

        case 'notification':
            this._bus.emit('notification', data.payload || data.data || data);
            break;

        case 'health_update':
            this._bus.emit('health_update', data.payload || data.data || data);
            break;

        case 'revenue_update':
            this._bus.emit('revenue_update', data.payload || data.data || data);
            break;

        case 'workflow_update':
            this._bus.emit('workflow_update', data.payload || data.data || data);
            break;

        case 'device_update':
            this._bus.emit('device_update', data.payload || data.data || data);
            break;

        case 'error':
            console.error('[WS] Server error:', data.message || data.payload);
            this._bus.emit('server_error', data.payload || data);
            break;

        case 'welcome':
            console.log('[WS] Server welcome:', data.message || '');
            this._bus.emit('welcome', data);
            break;

        default:
            // Emit generic typed event
            this._bus.emit(eventType, data.payload || data.data || data);
            // Also emit a catch-all
            this._bus.emit('message', data);
            break;
    }
};

WebSocketManager.prototype._handleAgentStatus = function _handleAgentStatus(payload) {
    // payload: { mission_id, status, step, progress_percent }
    var normalized = {
        mission_id: payload.mission_id || payload.id || null,
        status: payload.status || 'unknown',
        step: payload.step || payload.current_step || null,
        progress_percent: typeof payload.progress_percent === 'number'
            ? payload.progress_percent
            : (typeof payload.progress === 'number' ? payload.progress : 0),
        timestamp: payload.timestamp || new Date().toISOString(),
        details: payload.details || payload.message || null
    };
    this._bus.emit('agent_status', normalized);
};

WebSocketManager.prototype._handlePipelineProgress = function _handlePipelineProgress(payload) {
    // payload: { pipeline_id, stage, stage_index, total_stages, success }
    var normalized = {
        pipeline_id: payload.pipeline_id || payload.id || null,
        stage: payload.stage || payload.current_stage || 'unknown',
        stage_index: typeof payload.stage_index === 'number' ? payload.stage_index : 0,
        total_stages: typeof payload.total_stages === 'number' ? payload.total_stages : 1,
        success: payload.success !== false,
        site_id: payload.site_id || null,
        title: payload.title || null,
        timestamp: payload.timestamp || new Date().toISOString()
    };
    normalized.progress_percent = Math.round((normalized.stage_index / normalized.total_stages) * 100);
    this._bus.emit('pipeline_progress', normalized);
};

WebSocketManager.prototype._handlePhoneMirror = function _handlePhoneMirror(payload) {
    // payload: { device_id, screenshot_base64, timestamp }
    var normalized = {
        device_id: payload.device_id || payload.deviceId || 'unknown',
        screenshot_base64: payload.screenshot_base64 || payload.screenshot || payload.frame || null,
        timestamp: payload.timestamp || new Date().toISOString(),
        battery: payload.battery || null,
        wifi: payload.wifi || null,
        fps: payload.fps || null
    };
    this._bus.emit('phone_mirror', normalized);
};

WebSocketManager.prototype._handleActivity = function _handleActivity(payload) {
    // payload: { type, message, module, timestamp, severity }
    var normalized = {
        type: payload.type || 'info',
        message: payload.message || payload.text || '',
        module: payload.module || payload.source || null,
        timestamp: payload.timestamp || new Date().toISOString(),
        severity: payload.severity || payload.level || 'info',
        site_id: payload.site_id || payload.site || null,
        details: payload.details || null
    };
    this._bus.emit('activity', normalized);
};

// ---- Internal: Reconnection backoff --------------------------------------

WebSocketManager.prototype._getBackoffDelay = function _getBackoffDelay() {
    // Exponential: 1s, 2s, 4s, 8s, 16s, max 30s, plus jitter
    var base = Math.min(1000 * Math.pow(2, this._reconnectAttempts - 1), 30000);
    var jitter = Math.floor(Math.random() * 500);
    return base + jitter;
};

WebSocketManager.prototype._scheduleReconnect = function _scheduleReconnect() {
    if (this._manualClose) {
        return;
    }
    if (this._reconnectAttempts >= this._maxReconnectAttempts) {
        console.error('[WS] Max reconnect attempts (' + this._maxReconnectAttempts + ') reached. Giving up.');
        this._updateStatus('failed');
        this._bus.emit('max_reconnects_reached', { attempts: this._reconnectAttempts });
        return;
    }
    this.reconnect();
};

// ---- Internal: Heartbeat -------------------------------------------------

WebSocketManager.prototype._startPing = function _startPing() {
    this._stopPing();
    var self = this;

    this._pingTimer = setInterval(function () {
        if (self.isConnected()) {
            self.send('ping', { timestamp: Date.now() });
        }
    }, 25000); // Ping every 25 seconds

    // Check for pong timeouts
    this._pongTimer = setInterval(function () {
        if (self.isConnected() && self._lastPongTime > 0) {
            var elapsed = Date.now() - self._lastPongTime;
            if (elapsed > 60000) {
                console.warn('[WS] Pong timeout — server unresponsive. Reconnecting.');
                self._ws.close(4000, 'Pong timeout');
            }
        }
    }, 30000);
};

WebSocketManager.prototype._stopPing = function _stopPing() {
    if (this._pingTimer) {
        clearInterval(this._pingTimer);
        this._pingTimer = null;
    }
    if (this._pongTimer) {
        clearInterval(this._pongTimer);
        this._pongTimer = null;
    }
};

// ---- Internal: Queue management ------------------------------------------

WebSocketManager.prototype._flushQueue = function _flushQueue() {
    if (this._messageQueue.length === 0) {
        return;
    }
    console.log('[WS] Flushing ' + this._messageQueue.length + ' queued messages');
    var queue = this._messageQueue.slice();
    this._messageQueue = [];

    for (var i = 0; i < queue.length; i++) {
        try {
            this._ws.send(JSON.stringify(queue[i]));
            this._metrics.messagesSent++;
        } catch (err) {
            console.error('[WS] Failed to flush message:', err);
            // Re-queue remaining messages
            this._messageQueue = this._messageQueue.concat(queue.slice(i));
            break;
        }
    }
};

WebSocketManager.prototype._resubscribe = function _resubscribe() {
    for (var i = 0; i < this._subscriptions.length; i++) {
        this.send('subscribe', { channel: this._subscriptions[i] });
    }
};

// ---- Internal: Cleanup ---------------------------------------------------

WebSocketManager.prototype._cleanup = function _cleanup() {
    this._stopPing();

    if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
    }
};

// ---- Internal: Status DOM update -----------------------------------------

WebSocketManager.prototype._updateStatus = function _updateStatus(status) {
    // Update #ws-status element if it exists
    var el = document.getElementById(this._statusElementId);
    if (el) {
        el.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        el.className = 'ws-status ws-status-' + status;
    }

    // Update connection dot in header
    var dot = document.getElementById(this._connDotId);
    if (dot) {
        switch (status) {
            case 'connected':
                dot.className = 'status-dot green';
                dot.title = 'WebSocket Connected';
                break;
            case 'connecting':
            case 'reconnecting':
                dot.className = 'status-dot yellow';
                dot.title = 'WebSocket ' + status.charAt(0).toUpperCase() + status.slice(1);
                break;
            case 'disconnected':
            case 'error':
            case 'failed':
            case 'closed':
                dot.className = 'status-dot red';
                dot.title = 'WebSocket ' + status.charAt(0).toUpperCase() + status.slice(1);
                break;
            default:
                break;
        }
    }

    // Update offline banner
    var banner = document.getElementById('offlineBanner');
    if (banner) {
        if (status === 'disconnected' || status === 'error' || status === 'failed') {
            banner.classList.add('visible');
        } else if (status === 'connected') {
            banner.classList.remove('visible');
        }
    }
};

// ---- Internal: Utility ---------------------------------------------------

WebSocketManager.prototype._generateId = function _generateId() {
    return 'ws-' + Date.now().toString(36) + '-' + Math.random().toString(36).substr(2, 6);
};

// ---------------------------------------------------------------------------
//  Singleton instance
// ---------------------------------------------------------------------------

var wsManager = new WebSocketManager();
