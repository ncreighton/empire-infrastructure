// =============================================================================
//  Workflow Builder — OpenClaw Empire Dashboard v2
//  Canvas 2D drag-and-drop visual node editor for automation workflows.
//  Renders on <canvas id="workflow-canvas">, manages a node graph with
//  connections, supports pan/zoom, undo/redo, copy/paste, and execution.
// =============================================================================

'use strict';

// ---------------------------------------------------------------------------
//  Node Type Definitions
// ---------------------------------------------------------------------------

var WorkflowNodeType = {
    TRIGGER:   { key: 'TRIGGER',   color: '#4CAF50', icon: '\u23F1', label: 'Trigger',    inputs: 0, outputs: 1, description: 'Schedule, webhook, or manual start' },
    CONTENT:   { key: 'CONTENT',   color: '#2196F3', icon: '\u270D', label: 'Content',    inputs: 1, outputs: 2, description: 'Generate article or newsletter' },
    QUALITY:   { key: 'QUALITY',   color: '#FF9800', icon: '\u2714', label: 'Quality',    inputs: 1, outputs: 2, description: 'Quality check with pass/fail outputs' },
    PUBLISH:   { key: 'PUBLISH',   color: '#9C27B0', icon: '\uD83D\uDCE4', label: 'Publish',  inputs: 1, outputs: 1, description: 'WordPress or Substack publish' },
    SOCIAL:    { key: 'SOCIAL',    color: '#E91E63', icon: '\uD83D\uDCE3', label: 'Social',   inputs: 1, outputs: 1, description: 'Social campaign creation' },
    SEO:       { key: 'SEO',       color: '#00BCD4', icon: '\uD83D\uDD0D', label: 'SEO',      inputs: 1, outputs: 1, description: 'SEO audit and optimization' },
    DEVICE:    { key: 'DEVICE',    color: '#795548', icon: '\uD83D\uDCF1', label: 'Device',   inputs: 1, outputs: 1, description: 'Phone or device task execution' },
    NOTIFY:    { key: 'NOTIFY',    color: '#607D8B', icon: '\uD83D\uDD14', label: 'Notify',   inputs: 1, outputs: 0, description: 'Send notification (n8n, email)' },
    CONDITION: { key: 'CONDITION', color: '#FFC107', icon: '\u2753', label: 'Condition', inputs: 1, outputs: 2, description: 'Branch based on a condition' },
    TRANSFORM: { key: 'TRANSFORM', color: '#8BC34A', icon: '\u21C4', label: 'Transform', inputs: 1, outputs: 1, description: 'Data transformation' }
};

var NODE_TYPE_KEYS = Object.keys(WorkflowNodeType);

// ---------------------------------------------------------------------------
//  Node Config Field Definitions per type
// ---------------------------------------------------------------------------

var NODE_CONFIG_FIELDS = {
    TRIGGER: [
        { key: 'trigger_type', label: 'Trigger Type', type: 'select', options: ['schedule', 'webhook', 'manual'] },
        { key: 'cron', label: 'Cron Expression', type: 'text', placeholder: '0 9 * * 1-5' },
        { key: 'webhook_url', label: 'Webhook URL', type: 'text', placeholder: '/webhook/trigger-name' }
    ],
    CONTENT: [
        { key: 'site_id', label: 'Site', type: 'select', options: [] },
        { key: 'content_type', label: 'Content Type', type: 'select', options: ['article', 'newsletter', 'social_post', 'product_review'] },
        { key: 'word_count', label: 'Word Count', type: 'number', placeholder: '1500', min: 200, max: 10000 }
    ],
    QUALITY: [
        { key: 'min_score', label: 'Min Quality Score', type: 'range', min: 0, max: 100, step: 5 },
        { key: 'retry_count', label: 'Retry Count', type: 'number', placeholder: '2', min: 0, max: 5 }
    ],
    PUBLISH: [
        { key: 'site_id', label: 'Site', type: 'select', options: [] },
        { key: 'post_status', label: 'Status', type: 'select', options: ['draft', 'publish', 'pending', 'future'] },
        { key: 'category', label: 'Category', type: 'text', placeholder: 'tutorials' }
    ],
    SOCIAL: [
        { key: 'platforms', label: 'Platforms', type: 'checkboxes', options: ['twitter', 'facebook', 'instagram', 'pinterest', 'linkedin'] },
        { key: 'post_count', label: 'Posts per Platform', type: 'number', placeholder: '3', min: 1, max: 20 }
    ],
    SEO: [
        { key: 'audit_type', label: 'Audit Type', type: 'select', options: ['full', 'on_page', 'technical', 'backlinks'] },
        { key: 'fix_auto', label: 'Auto-Fix Issues', type: 'toggle' }
    ],
    DEVICE: [
        { key: 'device_id', label: 'Device', type: 'text', placeholder: 'android-01' },
        { key: 'task_description', label: 'Task', type: 'textarea', placeholder: 'Describe the task...' }
    ],
    NOTIFY: [
        { key: 'channel', label: 'Channel', type: 'select', options: ['n8n_webhook', 'email', 'slack', 'telegram', 'discord'] },
        { key: 'recipients', label: 'Recipients', type: 'text', placeholder: 'team@example.com' }
    ],
    CONDITION: [
        { key: 'field', label: 'Field', type: 'text', placeholder: 'data.quality_score' },
        { key: 'operator', label: 'Operator', type: 'select', options: ['==', '!=', '>', '<', '>=', '<=', 'contains', 'not_contains'] },
        { key: 'value', label: 'Value', type: 'text', placeholder: '80' }
    ],
    TRANSFORM: [
        { key: 'transform_type', label: 'Type', type: 'select', options: ['map', 'filter', 'reduce', 'template', 'json_path'] },
        { key: 'expression', label: 'Expression', type: 'textarea', placeholder: '$.data.title' }
    ]
};


// ---------------------------------------------------------------------------
//  WorkflowNode
// ---------------------------------------------------------------------------

function WorkflowNode(type, x, y, config) {
    var typeDef = WorkflowNodeType[type];
    if (!typeDef) {
        throw new Error('Unknown node type: ' + type);
    }

    this.id = 'wn-' + Date.now().toString(36) + '-' + Math.random().toString(36).substr(2, 6);
    this.type = type;
    this.x = x || 0;
    this.y = y || 0;
    this.width = 180;
    this.height = 80;
    this.title = config && config.title ? config.title : typeDef.label;
    this.config = config || {};
    this.inputs = [];
    this.outputs = [];
    this.selected = false;
    this.dragging = false;
    this.executionStatus = null; // null | 'running' | 'success' | 'error'

    for (var i = 0; i < typeDef.inputs; i++) {
        this.inputs.push({ id: this.id + '-in-' + i, connected: false, label: typeDef.inputs === 1 ? 'in' : 'in ' + i });
    }
    for (var o = 0; o < typeDef.outputs; o++) {
        var outLabel = 'out';
        if (typeDef.outputs === 2) {
            outLabel = o === 0 ? (type === 'QUALITY' || type === 'CONDITION' ? 'pass' : 'out 0') : (type === 'QUALITY' || type === 'CONDITION' ? 'fail' : 'out 1');
        }
        this.outputs.push({ id: this.id + '-out-' + o, connected: false, label: outLabel });
    }
}


// ---------------------------------------------------------------------------
//  WorkflowEdge
// ---------------------------------------------------------------------------

function WorkflowEdge(fromNode, fromPort, toNode, toPort) {
    this.id = 'we-' + Date.now().toString(36) + '-' + Math.random().toString(36).substr(2, 6);
    this.fromNode = fromNode;
    this.fromPort = fromPort;
    this.toNode = toNode;
    this.toPort = toPort;
    this.color = null;
}


// ---------------------------------------------------------------------------
//  WorkflowBuilder — Main class
// ---------------------------------------------------------------------------

function WorkflowBuilder(canvasId, options) {
    options = options || {};

    this.canvasId = canvasId;
    this.canvas = null;
    this.ctx = null;

    // Data model
    this.nodes = new Map();
    this.edges = new Map();
    this.selectedNodes = new Set();
    this.selectedEdge = null;

    // Drag state machine
    this.dragState = {
        type: 'none',       // 'none' | 'node' | 'edge' | 'select' | 'pan'
        startX: 0,
        startY: 0,
        currentX: 0,
        currentY: 0,
        nodeOffsets: null,   // Map<nodeId, {dx,dy}> for multi-select drag
        edgeFrom: null       // {nodeId, portIndex, isOutput}
    };

    // View transform (pan and zoom)
    this.viewTransform = { x: 0, y: 0, zoom: 1 };

    // Configuration
    this.gridSize = options.gridSize || 20;
    this.snapToGrid = options.snapToGrid !== false;
    this.showMinimap = options.showMinimap !== false;
    this.showGrid = options.showGrid !== false;
    this.apiBase = options.apiBase || (typeof API_BASE !== 'undefined' ? API_BASE : 'http://localhost:8765/api');

    // Clipboard and undo/redo
    this.clipboard = [];
    this.undoStack = [];
    this.redoStack = [];
    this.maxUndoSteps = 50;

    // Hover state
    this.hoveredNode = null;
    this.hoveredPort = null; // {nodeId, portIndex, isOutput}

    // Animation
    this._animFrame = null;
    this._needsRender = true;

    // Workflow metadata
    this.workflowId = null;
    this.workflowName = 'Untitled Workflow';
    this.workflowCreated = new Date().toISOString();
    this.workflowModified = new Date().toISOString();

    // Execution tracking
    this._executionId = null;
    this._executionRunning = false;

    // Config modal reference
    this._configModal = null;
    this._configNode = null;

    // Palette reference
    this._paletteContainer = null;

    // Bound event handlers (for clean removeEventListener)
    this._boundMouseDown = this._onMouseDown.bind(this);
    this._boundMouseMove = this._onMouseMove.bind(this);
    this._boundMouseUp = this._onMouseUp.bind(this);
    this._boundWheel = this._onWheel.bind(this);
    this._boundKeyDown = this._onKeyDown.bind(this);
    this._boundDblClick = this._onDoubleClick.bind(this);
    this._boundContextMenu = this._onContextMenu.bind(this);

    // DPI scale
    this._dpr = window.devicePixelRatio || 1;
}


// ---------------------------------------------------------------------------
//  Initialization & Teardown
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.init = function init() {
    this.canvas = document.getElementById(this.canvasId);
    if (!this.canvas) {
        console.error('[WorkflowBuilder] Canvas element not found: ' + this.canvasId);
        return;
    }
    this.ctx = this.canvas.getContext('2d');
    this._resizeCanvas();

    this.canvas.addEventListener('mousedown', this._boundMouseDown);
    this.canvas.addEventListener('mousemove', this._boundMouseMove);
    this.canvas.addEventListener('mouseup', this._boundMouseUp);
    this.canvas.addEventListener('wheel', this._boundWheel, { passive: false });
    this.canvas.addEventListener('dblclick', this._boundDblClick);
    this.canvas.addEventListener('contextmenu', this._boundContextMenu);
    document.addEventListener('keydown', this._boundKeyDown);

    var self = this;
    this._resizeObserver = new ResizeObserver(function () {
        self._resizeCanvas();
        self._requestRender();
    });
    this._resizeObserver.observe(this.canvas.parentElement || this.canvas);

    // Listen for WebSocket execution updates
    if (typeof wsManager !== 'undefined') {
        wsManager.on('workflow_update', function (data) {
            self.onExecutionUpdate(data);
        });
    }

    this._startRenderLoop();
    console.log('[WorkflowBuilder] Initialized on canvas: ' + this.canvasId);
};


WorkflowBuilder.prototype.destroy = function destroy() {
    if (this._animFrame) { cancelAnimationFrame(this._animFrame); this._animFrame = null; }
    if (this._resizeObserver) { this._resizeObserver.disconnect(); }
    if (this.canvas) {
        this.canvas.removeEventListener('mousedown', this._boundMouseDown);
        this.canvas.removeEventListener('mousemove', this._boundMouseMove);
        this.canvas.removeEventListener('mouseup', this._boundMouseUp);
        this.canvas.removeEventListener('wheel', this._boundWheel);
        this.canvas.removeEventListener('dblclick', this._boundDblClick);
        this.canvas.removeEventListener('contextmenu', this._boundContextMenu);
    }
    document.removeEventListener('keydown', this._boundKeyDown);
};


WorkflowBuilder.prototype._resizeCanvas = function _resizeCanvas() {
    if (!this.canvas) return;
    var rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    this.canvas.width = rect.width * this._dpr;
    this.canvas.height = rect.height * this._dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    this.ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    this._needsRender = true;
};


// ---------------------------------------------------------------------------
//  Render Loop
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._startRenderLoop = function _startRenderLoop() {
    var self = this;
    function loop() {
        if (self._needsRender) { self.render(); self._needsRender = false; }
        self._animFrame = requestAnimationFrame(loop);
    }
    this._animFrame = requestAnimationFrame(loop);
};

WorkflowBuilder.prototype._requestRender = function _requestRender() { this._needsRender = true; };

WorkflowBuilder.prototype.render = function render() {
    var ctx = this.ctx;
    var w = this.canvas.width / this._dpr;
    var h = this.canvas.height / this._dpr;

    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    ctx.translate(this.viewTransform.x, this.viewTransform.y);
    ctx.scale(this.viewTransform.zoom, this.viewTransform.zoom);

    if (this.showGrid) { this._drawGrid(ctx, w, h); }

    // Edges
    var edgeIter = this.edges.values();
    for (var eRes = edgeIter.next(); !eRes.done; eRes = edgeIter.next()) {
        this._drawEdge(ctx, eRes.value);
    }

    // Temp edge during drag
    if (this.dragState.type === 'edge' && this.dragState.edgeFrom) {
        this._drawTempEdge(ctx);
    }

    // Nodes
    var nodeIter = this.nodes.values();
    for (var nRes = nodeIter.next(); !nRes.done; nRes = nodeIter.next()) {
        this._drawNode(ctx, nRes.value);
    }

    // Selection box
    if (this.dragState.type === 'select') { this._drawSelectionBox(ctx); }

    ctx.restore();

    // Minimap (screen-space)
    if (this.showMinimap && this.nodes.size > 0) { this._drawMinimap(ctx, w, h); }

    // Zoom indicator
    this._drawZoomIndicator(ctx, w, h);
};


// ---------------------------------------------------------------------------
//  Grid Drawing
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._drawGrid = function _drawGrid(ctx, canvasW, canvasH) {
    var zoom = this.viewTransform.zoom;
    var tx = this.viewTransform.x;
    var ty = this.viewTransform.y;
    var gs = this.gridSize;

    var worldLeft = -tx / zoom;
    var worldTop = -ty / zoom;
    var worldRight = (canvasW - tx) / zoom;
    var worldBottom = (canvasH - ty) / zoom;

    var startX = Math.floor(worldLeft / gs) * gs;
    var startY = Math.floor(worldTop / gs) * gs;
    var endX = Math.ceil(worldRight / gs) * gs;
    var endY = Math.ceil(worldBottom / gs) * gs;

    ctx.fillStyle = '#2a2a3e';
    for (var x = startX; x <= endX; x += gs) {
        for (var y = startY; y <= endY; y += gs) {
            ctx.beginPath();
            ctx.arc(x, y, 1, 0, Math.PI * 2);
            ctx.fill();
        }
    }
};


// ---------------------------------------------------------------------------
//  Node Drawing
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._drawNode = function _drawNode(ctx, node) {
    var typeDef = WorkflowNodeType[node.type];
    if (!typeDef) return;

    var x = node.x, y = node.y, w = node.width, h = node.height;
    var r = 8;
    var titleBarH = 28;
    var color = typeDef.color;

    // Execution glow
    if (node.executionStatus) {
        var glowColors = { running: '#FFC107', success: '#4CAF50', error: '#ef4444' };
        var gc = glowColors[node.executionStatus];
        if (gc) {
            ctx.save();
            ctx.shadowColor = gc;
            ctx.shadowBlur = 16;
            this._roundRect(ctx, x - 2, y - 2, w + 4, h + 4, r + 2);
            ctx.fillStyle = this._hexToRgba(gc, 0.06);
            ctx.fill();
            ctx.restore();
        }
    }

    // Selection glow
    if (node.selected) {
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;
        this._roundRect(ctx, x - 1, y - 1, w + 2, h + 2, r + 1);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
    }

    // Body
    ctx.save();
    this._roundRect(ctx, x, y, w, h, r);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fill();
    ctx.strokeStyle = node.selected ? color : 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // Title bar fill
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + titleBarH);
    ctx.lineTo(x, y + titleBarH);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fillStyle = this._hexToRgba(color, 0.2);
    ctx.fill();
    ctx.restore();

    // Title bar divider
    ctx.beginPath();
    ctx.moveTo(x, y + titleBarH);
    ctx.lineTo(x + w, y + titleBarH);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Icon
    ctx.font = '13px sans-serif';
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(typeDef.icon, x + 8, y + titleBarH / 2);

    // Title text (truncated)
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillStyle = '#e0e0e0';
    ctx.textAlign = 'left';
    var titleMaxW = w - 36;
    var titleText = node.title;
    if (ctx.measureText(titleText).width > titleMaxW) {
        while (ctx.measureText(titleText + '...').width > titleMaxW && titleText.length > 0) {
            titleText = titleText.slice(0, -1);
        }
        titleText += '...';
    }
    ctx.fillText(titleText, x + 26, y + titleBarH / 2);

    // Subtitle: type key + optional site_id
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#aaaaaa';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    var subtitle = typeDef.key;
    if (node.config && node.config.site_id) { subtitle += ' | ' + node.config.site_id; }
    ctx.fillText(subtitle, x + 8, y + titleBarH + 6);

    // Config summary
    var summary = this._getConfigSummary(node);
    if (summary) {
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillStyle = '#777777';
        var sMaxW = w - 16;
        if (ctx.measureText(summary).width > sMaxW) {
            while (ctx.measureText(summary + '...').width > sMaxW && summary.length > 0) {
                summary = summary.slice(0, -1);
            }
            summary += '...';
        }
        ctx.fillText(summary, x + 8, y + titleBarH + 20);
    }

    // Execution status dot
    if (node.executionStatus) {
        var sc = { running: '#FFC107', success: '#4CAF50', error: '#ef4444' };
        ctx.beginPath();
        ctx.arc(x + w - 12, y + 14, 4, 0, Math.PI * 2);
        ctx.fillStyle = sc[node.executionStatus] || '#607D8B';
        ctx.fill();
        if (node.executionStatus === 'running') {
            ctx.beginPath();
            ctx.arc(x + w - 12, y + 14, 6, 0, Math.PI * 2);
            ctx.strokeStyle = this._hexToRgba(sc.running, 0.4);
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    // Ports
    this._drawPorts(ctx, node);
};


WorkflowBuilder.prototype._drawPorts = function _drawPorts(ctx, node) {
    var typeDef = WorkflowNodeType[node.type];
    var pr = 6;
    var hr = 8;

    // Input ports (left)
    for (var i = 0; i < node.inputs.length; i++) {
        var inp = node.inputs[i];
        var pos = this.getPortPosition(node, i, false);
        var hovered = this.hoveredPort && this.hoveredPort.nodeId === node.id && this.hoveredPort.portIndex === i && !this.hoveredPort.isOutput;
        var radius = hovered ? hr : pr;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        if (inp.connected) { ctx.fillStyle = typeDef.color; }
        else { ctx.fillStyle = '#1a1a2e'; ctx.strokeStyle = typeDef.color; ctx.lineWidth = 2; ctx.stroke(); }
        ctx.fill();

        if (hovered) {
            ctx.beginPath(); ctx.arc(pos.x, pos.y, radius + 3, 0, Math.PI * 2);
            ctx.strokeStyle = this._hexToRgba(typeDef.color, 0.3); ctx.lineWidth = 1; ctx.stroke();
        }
    }

    // Output ports (right)
    for (var o = 0; o < node.outputs.length; o++) {
        var out = node.outputs[o];
        var oPos = this.getPortPosition(node, o, true);
        var oHov = this.hoveredPort && this.hoveredPort.nodeId === node.id && this.hoveredPort.portIndex === o && this.hoveredPort.isOutput;
        var oRad = oHov ? hr : pr;

        var portColor = typeDef.color;
        if ((node.type === 'CONDITION' || node.type === 'QUALITY') && node.outputs.length === 2) {
            portColor = o === 0 ? '#4CAF50' : '#ef4444';
        }

        ctx.beginPath();
        ctx.arc(oPos.x, oPos.y, oRad, 0, Math.PI * 2);
        if (out.connected) { ctx.fillStyle = portColor; }
        else { ctx.fillStyle = '#1a1a2e'; ctx.strokeStyle = portColor; ctx.lineWidth = 2; ctx.stroke(); }
        ctx.fill();

        if (oHov) {
            ctx.beginPath(); ctx.arc(oPos.x, oPos.y, oRad + 3, 0, Math.PI * 2);
            ctx.strokeStyle = this._hexToRgba(portColor, 0.3); ctx.lineWidth = 1; ctx.stroke();
        }

        // Label for multi-output
        if (node.outputs.length > 1) {
            ctx.font = '8px JetBrains Mono, monospace';
            ctx.fillStyle = '#666666';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(out.label, oPos.x - 10, oPos.y);
        }
    }
};


// ---------------------------------------------------------------------------
//  Edge Drawing
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._drawEdge = function _drawEdge(ctx, edge) {
    var fromNode = this.nodes.get(edge.fromNode);
    var toNode = this.nodes.get(edge.toNode);
    if (!fromNode || !toNode) return;

    var fromPos = this.getPortPosition(fromNode, edge.fromPort, true);
    var toPos = this.getPortPosition(toNode, edge.toPort, false);
    var fromType = WorkflowNodeType[fromNode.type];
    var toType = WorkflowNodeType[toNode.type];
    var fromColor = fromType ? fromType.color : '#607D8B';
    var toColor = toType ? toType.color : '#607D8B';

    if ((fromNode.type === 'CONDITION' || fromNode.type === 'QUALITY') && fromNode.outputs.length === 2) {
        fromColor = edge.fromPort === 0 ? '#4CAF50' : '#ef4444';
    }

    var gradient = ctx.createLinearGradient(fromPos.x, fromPos.y, toPos.x, toPos.y);
    gradient.addColorStop(0, this._hexToRgba(fromColor, 0.8));
    gradient.addColorStop(1, this._hexToRgba(toColor, 0.8));

    var dx = Math.abs(toPos.x - fromPos.x);
    var cp = Math.max(dx * 0.5, 50);

    ctx.beginPath();
    ctx.moveTo(fromPos.x, fromPos.y);
    ctx.bezierCurveTo(fromPos.x + cp, fromPos.y, toPos.x - cp, toPos.y, toPos.x, toPos.y);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = this.selectedEdge === edge.id ? 3 : 2;
    ctx.stroke();

    if (this.selectedEdge === edge.id) {
        ctx.strokeStyle = this._hexToRgba(fromColor, 0.3);
        ctx.lineWidth = 6;
        ctx.stroke();
    }

    this._drawArrowHead(ctx, toPos.x - cp * 0.3, toPos.y, toPos.x, toPos.y, toColor);
};


WorkflowBuilder.prototype._drawTempEdge = function _drawTempEdge(ctx) {
    var fromInfo = this.dragState.edgeFrom;
    var fromNode = this.nodes.get(fromInfo.nodeId);
    if (!fromNode) return;

    var fromPos = this.getPortPosition(fromNode, fromInfo.portIndex, fromInfo.isOutput);
    var toPos = this.screenToWorld(this.dragState.currentX, this.dragState.currentY);
    var fromType = WorkflowNodeType[fromNode.type];
    var color = fromType ? fromType.color : '#607D8B';

    var dx = Math.abs(toPos.x - fromPos.x);
    var cp = Math.max(dx * 0.5, 50);

    var sx, sy, ex, ey;
    if (fromInfo.isOutput) { sx = fromPos.x; sy = fromPos.y; ex = toPos.x; ey = toPos.y; }
    else { sx = toPos.x; sy = toPos.y; ex = fromPos.x; ey = fromPos.y; }

    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.bezierCurveTo(sx + cp, sy, ex - cp, ey, ex, ey);
    ctx.strokeStyle = this._hexToRgba(color, 0.5);
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
};


WorkflowBuilder.prototype._drawArrowHead = function _drawArrowHead(ctx, fromX, fromY, toX, toY, color) {
    var angle = Math.atan2(toY - fromY, toX - fromX);
    var len = 8;
    ctx.save();
    ctx.translate(toX, toY);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-len, len * 0.5);
    ctx.lineTo(-len, -len * 0.5);
    ctx.closePath();
    ctx.fillStyle = this._hexToRgba(color, 0.8);
    ctx.fill();
    ctx.restore();
};


// ---------------------------------------------------------------------------
//  Selection Box & Minimap Drawing
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._drawSelectionBox = function _drawSelectionBox(ctx) {
    var s = this.dragState;
    var ws = this.screenToWorld(s.startX, s.startY);
    var we = this.screenToWorld(s.currentX, s.currentY);
    var x = Math.min(ws.x, we.x), y = Math.min(ws.y, we.y);
    var bw = Math.abs(we.x - ws.x), bh = Math.abs(we.y - ws.y);

    ctx.fillStyle = 'rgba(99,102,241,0.08)';
    ctx.fillRect(x, y, bw, bh);
    ctx.strokeStyle = 'rgba(99,102,241,0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x, y, bw, bh);
    ctx.setLineDash([]);
};


WorkflowBuilder.prototype._drawMinimap = function _drawMinimap(ctx, cw, ch) {
    var mmW = 150, mmH = 100;
    var mmX = cw - mmW - 12, mmY = ch - mmH - 12, pad = 4;

    ctx.fillStyle = 'rgba(10,10,15,0.85)';
    this._roundRect(ctx, mmX, mmY, mmW, mmH, 6);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();

    var bounds = this._getWorldBounds();
    if (!bounds) return;

    var bW = bounds.maxX - bounds.minX + 100;
    var bH = bounds.maxY - bounds.minY + 100;
    var scale = Math.min((mmW - pad * 2) / bW, (mmH - pad * 2) / bH, 1);

    var nIter = this.nodes.values();
    for (var nr = nIter.next(); !nr.done; nr = nIter.next()) {
        var n = nr.value;
        var td = WorkflowNodeType[n.type];
        ctx.fillStyle = this._hexToRgba(td ? td.color : '#607D8B', 0.6);
        ctx.fillRect(
            mmX + pad + (n.x - bounds.minX) * scale,
            mmY + pad + (n.y - bounds.minY) * scale,
            Math.max(n.width * scale, 3),
            Math.max(n.height * scale, 2)
        );
    }

    // Viewport rect
    var vl = -this.viewTransform.x / this.viewTransform.zoom;
    var vt = -this.viewTransform.y / this.viewTransform.zoom;
    var vr = vl + cw / this.viewTransform.zoom;
    var vb = vt + ch / this.viewTransform.zoom;
    var vx = Math.max(mmX + pad, mmX + pad + (vl - bounds.minX) * scale);
    var vy = Math.max(mmY + pad, mmY + pad + (vt - bounds.minY) * scale);
    var vw = Math.min((vr - vl) * scale, mmW - pad * 2);
    var vh = Math.min((vb - vt) * scale, mmH - pad * 2);

    ctx.strokeStyle = 'rgba(99,102,241,0.6)';
    ctx.lineWidth = 1;
    ctx.strokeRect(vx, vy, vw, vh);
    ctx.fillStyle = 'rgba(99,102,241,0.06)';
    ctx.fillRect(vx, vy, vw, vh);
};


WorkflowBuilder.prototype._drawZoomIndicator = function _drawZoomIndicator(ctx, cw, ch) {
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#475569';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(Math.round(this.viewTransform.zoom * 100) + '%', 12, ch - 12);
};


// ---------------------------------------------------------------------------
//  Mouse Events
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._onMouseDown = function _onMouseDown(e) {
    e.preventDefault();
    var rect = this.canvas.getBoundingClientRect();
    var sx = e.clientX - rect.left;
    var sy = e.clientY - rect.top;
    var world = this.screenToWorld(sx, sy);

    this.dragState.startX = sx;
    this.dragState.startY = sy;
    this.dragState.currentX = sx;
    this.dragState.currentY = sy;

    // Port hit?
    var portHit = this.getPortAtPoint(world.x, world.y);
    if (portHit) {
        this.dragState.type = 'edge';
        this.dragState.edgeFrom = { nodeId: portHit.node.id, portIndex: portHit.portIndex, isOutput: portHit.isOutput };
        this._requestRender();
        return;
    }

    // Node hit?
    var nodeHit = this.getNodeAtPoint(world.x, world.y);
    if (nodeHit) {
        this.selectedEdge = null;
        if (e.shiftKey) {
            nodeHit.selected = !nodeHit.selected;
            if (nodeHit.selected) this.selectedNodes.add(nodeHit.id);
            else this.selectedNodes.delete(nodeHit.id);
        } else if (!nodeHit.selected) {
            this._clearSelection();
            nodeHit.selected = true;
            this.selectedNodes.add(nodeHit.id);
        }
        this.dragState.type = 'node';
        this.dragState.nodeOffsets = new Map();
        var self = this;
        this.selectedNodes.forEach(function (nid) {
            var n = self.nodes.get(nid);
            if (n) self.dragState.nodeOffsets.set(nid, { dx: world.x - n.x, dy: world.y - n.y });
        });
        this._requestRender();
        return;
    }

    // Edge hit?
    var edgeHit = this._getEdgeAtPoint(world.x, world.y);
    if (edgeHit) {
        this._clearSelection();
        this.selectedEdge = edgeHit.id;
        this._requestRender();
        return;
    }

    // Empty space
    this._clearSelection();
    this.selectedEdge = null;
    this.dragState.type = (e.button === 1 || e.altKey || e.metaKey) ? 'pan' : 'select';
    this._requestRender();
};


WorkflowBuilder.prototype._onMouseMove = function _onMouseMove(e) {
    var rect = this.canvas.getBoundingClientRect();
    var sx = e.clientX - rect.left;
    var sy = e.clientY - rect.top;
    var world = this.screenToWorld(sx, sy);

    this.dragState.currentX = sx;
    this.dragState.currentY = sy;

    switch (this.dragState.type) {
        case 'node':
            this._handleNodeDrag(world);
            break;
        case 'edge':
            this._requestRender();
            break;
        case 'select':
            this._updateRubberBand();
            this._requestRender();
            break;
        case 'pan':
            this.viewTransform.x += sx - this.dragState.startX;
            this.viewTransform.y += sy - this.dragState.startY;
            this.dragState.startX = sx;
            this.dragState.startY = sy;
            this._requestRender();
            break;
        case 'none':
            this._updateHover(world);
            break;
    }
};


WorkflowBuilder.prototype._onMouseUp = function _onMouseUp(e) {
    var rect = this.canvas.getBoundingClientRect();
    var world = this.screenToWorld(e.clientX - rect.left, e.clientY - rect.top);

    switch (this.dragState.type) {
        case 'node': this._finalizeNodeDrag(); break;
        case 'edge': this._finalizeEdgeDrag(world); break;
    }

    this.dragState.type = 'none';
    this.dragState.edgeFrom = null;
    this.dragState.nodeOffsets = null;
    this._requestRender();
};


WorkflowBuilder.prototype._onWheel = function _onWheel(e) {
    e.preventDefault();
    var rect = this.canvas.getBoundingClientRect();
    var sx = e.clientX - rect.left;
    var sy = e.clientY - rect.top;

    var factor = e.deltaY < 0 ? 1.1 : 0.9;
    var newZoom = Math.max(0.15, Math.min(4, this.viewTransform.zoom * factor));
    var worldBefore = this.screenToWorld(sx, sy);
    this.viewTransform.zoom = newZoom;
    var worldAfter = this.screenToWorld(sx, sy);

    this.viewTransform.x += (worldAfter.x - worldBefore.x) * newZoom;
    this.viewTransform.y += (worldAfter.y - worldBefore.y) * newZoom;
    this._requestRender();
};


WorkflowBuilder.prototype._onDoubleClick = function _onDoubleClick(e) {
    var rect = this.canvas.getBoundingClientRect();
    var world = this.screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    var nodeHit = this.getNodeAtPoint(world.x, world.y);
    if (nodeHit) this.showConfigModal(nodeHit);
};


WorkflowBuilder.prototype._onContextMenu = function _onContextMenu(e) { e.preventDefault(); };


WorkflowBuilder.prototype._onKeyDown = function _onKeyDown(e) {
    var active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.tagName === 'SELECT')) return;

    var ctrl = e.ctrlKey || e.metaKey;

    switch (e.key) {
        case 'Delete': case 'Backspace': e.preventDefault(); this._deleteSelected(); break;
        case 'z': if (ctrl && !e.shiftKey) { e.preventDefault(); this.undo(); } break;
        case 'Z': case 'y': if (ctrl) { e.preventDefault(); this.redo(); } break;
        case 'c': if (ctrl) { e.preventDefault(); this._copySelected(); } break;
        case 'v': if (ctrl) { e.preventDefault(); this._pasteClipboard(); } break;
        case 'a': if (ctrl) { e.preventDefault(); this._selectAll(); } break;
        case 'd': if (ctrl) { e.preventDefault(); this._duplicateSelected(); } break;
        case 'Escape':
            this._clearSelection();
            this.selectedEdge = null;
            this.dragState.type = 'none';
            this._requestRender();
            break;
    }
};


// ---------------------------------------------------------------------------
//  Drag Handlers
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._handleNodeDrag = function _handleNodeDrag(world) {
    var self = this;
    this.selectedNodes.forEach(function (nid) {
        var n = self.nodes.get(nid);
        var off = self.dragState.nodeOffsets ? self.dragState.nodeOffsets.get(nid) : null;
        if (n && off) {
            var nx = world.x - off.dx;
            var ny = world.y - off.dy;
            if (self.snapToGrid) {
                nx = Math.round(nx / self.gridSize) * self.gridSize;
                ny = Math.round(ny / self.gridSize) * self.gridSize;
            }
            n.x = nx;
            n.y = ny;
            n.dragging = true;
        }
    });
    this._requestRender();
};


WorkflowBuilder.prototype._finalizeNodeDrag = function _finalizeNodeDrag() {
    var moved = false;
    this.selectedNodes.forEach(function (nid) {
        var n = this.nodes.get(nid);
        if (n) { if (n.dragging) moved = true; n.dragging = false; }
    }.bind(this));
    if (moved) {
        this.pushUndo({ type: 'move_node', nodeIds: Array.from(this.selectedNodes) });
        this.workflowModified = new Date().toISOString();
    }
};


WorkflowBuilder.prototype._finalizeEdgeDrag = function _finalizeEdgeDrag(world) {
    if (!this.dragState.edgeFrom) return;
    var portHit = this.getPortAtPoint(world.x, world.y);
    if (!portHit) return;

    var from = this.dragState.edgeFrom;
    if (from.nodeId === portHit.node.id) return;
    if (from.isOutput === portHit.isOutput) return;

    var fromId, fromPort, toId, toPort;
    if (from.isOutput) { fromId = from.nodeId; fromPort = from.portIndex; toId = portHit.node.id; toPort = portHit.portIndex; }
    else { fromId = portHit.node.id; fromPort = portHit.portIndex; toId = from.nodeId; toPort = from.portIndex; }

    if (!this.isPortCompatible(this.nodes.get(fromId), this.nodes.get(toId))) return;

    var dup = false;
    this.edges.forEach(function (ex) {
        if (ex.fromNode === fromId && ex.fromPort === fromPort && ex.toNode === toId && ex.toPort === toPort) dup = true;
    });
    if (!dup) this.connectNodes(fromId, fromPort, toId, toPort);
};


WorkflowBuilder.prototype._updateRubberBand = function _updateRubberBand() {
    var s = this.dragState;
    var ws = this.screenToWorld(s.startX, s.startY);
    var we = this.screenToWorld(s.currentX, s.currentY);
    var l = Math.min(ws.x, we.x), t = Math.min(ws.y, we.y);
    var r = Math.max(ws.x, we.x), b = Math.max(ws.y, we.y);

    var self = this;
    this.nodes.forEach(function (node) {
        var hit = node.x + node.width > l && node.x < r && node.y + node.height > t && node.y < b;
        node.selected = hit;
        if (hit) self.selectedNodes.add(node.id);
        else self.selectedNodes.delete(node.id);
    });
};


WorkflowBuilder.prototype._updateHover = function _updateHover(world) {
    var prevNode = this.hoveredNode;
    var prevPort = this.hoveredPort;

    var portHit = this.getPortAtPoint(world.x, world.y);
    if (portHit) {
        this.hoveredPort = { nodeId: portHit.node.id, portIndex: portHit.portIndex, isOutput: portHit.isOutput };
        this.hoveredNode = portHit.node.id;
        this.canvas.style.cursor = 'crosshair';
    } else {
        this.hoveredPort = null;
        var nodeHit = this.getNodeAtPoint(world.x, world.y);
        this.hoveredNode = nodeHit ? nodeHit.id : null;
        this.canvas.style.cursor = nodeHit ? 'move' : 'grab';
    }

    if (this.hoveredNode !== prevNode || JSON.stringify(this.hoveredPort) !== JSON.stringify(prevPort)) {
        this._requestRender();
    }
};


// ---------------------------------------------------------------------------
//  Hit Testing
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.getNodeAtPoint = function getNodeAtPoint(wx, wy) {
    var arr = Array.from(this.nodes.values()).reverse();
    for (var i = 0; i < arr.length; i++) {
        var n = arr[i];
        if (wx >= n.x && wx <= n.x + n.width && wy >= n.y && wy <= n.y + n.height) return n;
    }
    return null;
};


WorkflowBuilder.prototype.getPortAtPoint = function getPortAtPoint(wx, wy) {
    var hitR2 = 100; // 10px squared
    var arr = Array.from(this.nodes.values()).reverse();
    for (var i = 0; i < arr.length; i++) {
        var n = arr[i];
        for (var inp = 0; inp < n.inputs.length; inp++) {
            var p = this.getPortPosition(n, inp, false);
            if ((wx - p.x) * (wx - p.x) + (wy - p.y) * (wy - p.y) <= hitR2) return { node: n, portIndex: inp, isOutput: false };
        }
        for (var out = 0; out < n.outputs.length; out++) {
            var op = this.getPortPosition(n, out, true);
            if ((wx - op.x) * (wx - op.x) + (wy - op.y) * (wy - op.y) <= hitR2) return { node: n, portIndex: out, isOutput: true };
        }
    }
    return null;
};


WorkflowBuilder.prototype._getEdgeAtPoint = function _getEdgeAtPoint(wx, wy) {
    var hitD2 = 64; // 8px squared
    var arr = Array.from(this.edges.values());
    for (var i = 0; i < arr.length; i++) {
        var edge = arr[i];
        var fn = this.nodes.get(edge.fromNode), tn = this.nodes.get(edge.toNode);
        if (!fn || !tn) continue;
        var fp = this.getPortPosition(fn, edge.fromPort, true);
        var tp = this.getPortPosition(tn, edge.toPort, false);
        var dx = Math.abs(tp.x - fp.x);
        var cp = Math.max(dx * 0.5, 50);

        for (var t = 0; t <= 1; t += 0.05) {
            var mt = 1 - t;
            var bx = mt * mt * mt * fp.x + 3 * mt * mt * t * (fp.x + cp) + 3 * mt * t * t * (tp.x - cp) + t * t * t * tp.x;
            var by = mt * mt * mt * fp.y + 3 * mt * mt * t * fp.y + 3 * mt * t * t * tp.y + t * t * t * tp.y;
            if ((wx - bx) * (wx - bx) + (wy - by) * (wy - by) <= hitD2) return edge;
        }
    }
    return null;
};


// ---------------------------------------------------------------------------
//  Port Geometry & Validation
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.getPortPosition = function getPortPosition(node, portIndex, isOutput) {
    var count = isOutput ? node.outputs.length : node.inputs.length;
    var spacing = node.height / (count + 1);
    return { x: isOutput ? node.x + node.width : node.x, y: node.y + spacing * (portIndex + 1) };
};


WorkflowBuilder.prototype.isPortCompatible = function isPortCompatible(fromNode, toNode) {
    return fromNode && toNode && fromNode.id !== toNode.id;
};


// ---------------------------------------------------------------------------
//  Coordinate Transforms
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.screenToWorld = function screenToWorld(sx, sy) {
    return { x: (sx - this.viewTransform.x) / this.viewTransform.zoom, y: (sy - this.viewTransform.y) / this.viewTransform.zoom };
};

WorkflowBuilder.prototype.worldToScreen = function worldToScreen(wx, wy) {
    return { x: wx * this.viewTransform.zoom + this.viewTransform.x, y: wy * this.viewTransform.zoom + this.viewTransform.y };
};


// ---------------------------------------------------------------------------
//  Node Management
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.addNode = function addNode(type, x, y, config) {
    var node = new WorkflowNode(type, x, y, config);
    this.nodes.set(node.id, node);
    this.pushUndo({ type: 'add_node', nodeId: node.id });
    this.workflowModified = new Date().toISOString();
    this._requestRender();
    return node;
};


WorkflowBuilder.prototype.removeNode = function removeNode(id) {
    var node = this.nodes.get(id);
    if (!node) return;

    var toRemove = [];
    this.edges.forEach(function (edge) {
        if (edge.fromNode === id || edge.toNode === id) toRemove.push(edge.id);
    });
    var self = this;
    toRemove.forEach(function (eid) { self._removeEdgeRaw(eid); });

    this.nodes.delete(id);
    this.selectedNodes.delete(id);
    this.pushUndo({ type: 'remove_node', nodeId: id });
    this.workflowModified = new Date().toISOString();
    this._requestRender();
};


WorkflowBuilder.prototype.duplicateNode = function duplicateNode(id) {
    var src = this.nodes.get(id);
    if (!src) return null;
    var cfg = JSON.parse(JSON.stringify(src.config || {}));
    cfg.title = src.title;
    var n = this.addNode(src.type, src.x + 40, src.y + 40, cfg);
    n.title = src.title + ' (copy)';
    return n;
};


WorkflowBuilder.prototype.connectNodes = function connectNodes(fromId, fromPort, toId, toPort) {
    var fn = this.nodes.get(fromId), tn = this.nodes.get(toId);
    if (!fn || !tn) return null;
    if (fromPort < 0 || fromPort >= fn.outputs.length) return null;
    if (toPort < 0 || toPort >= tn.inputs.length) return null;

    // Remove existing edges to the same input port
    var rem = [];
    this.edges.forEach(function (e) { if (e.toNode === toId && e.toPort === toPort) rem.push(e.id); });
    var self = this;
    rem.forEach(function (eid) { self._removeEdgeRaw(eid); });

    var edge = new WorkflowEdge(fromId, fromPort, toId, toPort);
    this.edges.set(edge.id, edge);
    fn.outputs[fromPort].connected = true;
    tn.inputs[toPort].connected = true;

    this.pushUndo({ type: 'add_edge', edgeId: edge.id });
    this.workflowModified = new Date().toISOString();
    this._requestRender();
    return edge;
};


WorkflowBuilder.prototype.disconnectEdge = function disconnectEdge(edgeId) {
    this._removeEdgeRaw(edgeId);
    this.pushUndo({ type: 'remove_edge', edgeId: edgeId });
    this.workflowModified = new Date().toISOString();
    this._requestRender();
};


WorkflowBuilder.prototype._removeEdgeRaw = function _removeEdgeRaw(edgeId) {
    var edge = this.edges.get(edgeId);
    if (!edge) return;

    var fn = this.nodes.get(edge.fromNode);
    var tn = this.nodes.get(edge.toNode);

    if (fn && fn.outputs[edge.fromPort]) {
        var still = false;
        this.edges.forEach(function (e) { if (e.id !== edgeId && e.fromNode === edge.fromNode && e.fromPort === edge.fromPort) still = true; });
        if (!still) fn.outputs[edge.fromPort].connected = false;
    }
    if (tn && tn.inputs[edge.toPort]) { tn.inputs[edge.toPort].connected = false; }

    this.edges.delete(edgeId);
    if (this.selectedEdge === edgeId) this.selectedEdge = null;
};


// ---------------------------------------------------------------------------
//  Selection Helpers
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._clearSelection = function _clearSelection() {
    var self = this;
    this.selectedNodes.forEach(function (nid) { var n = self.nodes.get(nid); if (n) n.selected = false; });
    this.selectedNodes.clear();
};

WorkflowBuilder.prototype._selectAll = function _selectAll() {
    var self = this;
    this.nodes.forEach(function (n) { n.selected = true; self.selectedNodes.add(n.id); });
    this._requestRender();
};

WorkflowBuilder.prototype._deleteSelected = function _deleteSelected() {
    if (this.selectedEdge) { this.disconnectEdge(this.selectedEdge); return; }
    if (this.selectedNodes.size === 0) return;
    var ids = Array.from(this.selectedNodes);
    var self = this;
    ids.forEach(function (nid) { self.removeNode(nid); });
};

WorkflowBuilder.prototype._copySelected = function _copySelected() {
    this.clipboard = [];
    var self = this;
    this.selectedNodes.forEach(function (nid) {
        var n = self.nodes.get(nid);
        if (n) self.clipboard.push({ type: n.type, x: n.x, y: n.y, title: n.title, config: JSON.parse(JSON.stringify(n.config || {})) });
    });
};

WorkflowBuilder.prototype._pasteClipboard = function _pasteClipboard() {
    if (this.clipboard.length === 0) return;
    this._clearSelection();
    var self = this;
    this.clipboard.forEach(function (item) {
        var n = self.addNode(item.type, item.x + 40, item.y + 40, JSON.parse(JSON.stringify(item.config)));
        n.title = item.title;
        n.selected = true;
        self.selectedNodes.add(n.id);
    });
    this._requestRender();
};

WorkflowBuilder.prototype._duplicateSelected = function _duplicateSelected() {
    if (this.selectedNodes.size === 0) return;
    var newSel = [];
    var self = this;
    this.selectedNodes.forEach(function (nid) { var nn = self.duplicateNode(nid); if (nn) newSel.push(nn.id); });
    this._clearSelection();
    newSel.forEach(function (nid) { var n = self.nodes.get(nid); if (n) { n.selected = true; self.selectedNodes.add(nid); } });
    this._requestRender();
};


// ---------------------------------------------------------------------------
//  Undo / Redo
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.pushUndo = function pushUndo(action) {
    action.snapshot = this._createSnapshot();
    this.undoStack.push(action);
    if (this.undoStack.length > this.maxUndoSteps) this.undoStack.shift();
    this.redoStack = [];
};

WorkflowBuilder.prototype.undo = function undo() {
    if (this.undoStack.length === 0) return;
    var act = this.undoStack.pop();
    this.redoStack.push({ type: 'redo', snapshot: this._createSnapshot() });
    if (act.snapshot) this._restoreSnapshot(act.snapshot);
    this._requestRender();
};

WorkflowBuilder.prototype.redo = function redo() {
    if (this.redoStack.length === 0) return;
    var act = this.redoStack.pop();
    this.undoStack.push({ type: 'undo_from_redo', snapshot: this._createSnapshot() });
    if (act.snapshot) this._restoreSnapshot(act.snapshot);
    this._requestRender();
};

WorkflowBuilder.prototype._createSnapshot = function _createSnapshot() {
    var nodesData = [], edgesData = [];
    this.nodes.forEach(function (n) {
        nodesData.push({ id: n.id, type: n.type, x: n.x, y: n.y, width: n.width, height: n.height, title: n.title,
            config: JSON.parse(JSON.stringify(n.config || {})), inputs: JSON.parse(JSON.stringify(n.inputs)), outputs: JSON.parse(JSON.stringify(n.outputs)) });
    });
    this.edges.forEach(function (e) {
        edgesData.push({ id: e.id, fromNode: e.fromNode, fromPort: e.fromPort, toNode: e.toNode, toPort: e.toPort });
    });
    return { nodes: nodesData, edges: edgesData };
};

WorkflowBuilder.prototype._restoreSnapshot = function _restoreSnapshot(snap) {
    this.nodes.clear(); this.edges.clear(); this.selectedNodes.clear(); this.selectedEdge = null;
    var self = this;
    snap.nodes.forEach(function (d) {
        var n = new WorkflowNode(d.type, d.x, d.y, d.config);
        n.id = d.id; n.width = d.width; n.height = d.height; n.title = d.title;
        n.inputs = d.inputs; n.outputs = d.outputs; n.selected = false; n.dragging = false; n.executionStatus = null;
        self.nodes.set(n.id, n);
    });
    snap.edges.forEach(function (d) {
        var e = new WorkflowEdge(d.fromNode, d.fromPort, d.toNode, d.toPort);
        e.id = d.id; self.edges.set(e.id, e);
    });
};


// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.toJSON = function toJSON() {
    var na = [], ea = [];
    this.nodes.forEach(function (n) { na.push({ id: n.id, type: n.type, x: n.x, y: n.y, width: n.width, height: n.height, title: n.title, config: n.config || {} }); });
    this.edges.forEach(function (e) { ea.push({ id: e.id, fromNode: e.fromNode, fromPort: e.fromPort, toNode: e.toNode, toPort: e.toPort }); });
    return { nodes: na, edges: ea, metadata: { id: this.workflowId, name: this.workflowName, created: this.workflowCreated, modified: this.workflowModified, nodeCount: na.length, edgeCount: ea.length } };
};

WorkflowBuilder.prototype.fromJSON = function fromJSON(data) {
    if (!data) return;
    this.nodes.clear(); this.edges.clear(); this.selectedNodes.clear(); this.selectedEdge = null;
    this.undoStack = []; this.redoStack = [];
    var self = this;
    var meta = data.metadata || {};
    this.workflowId = meta.id || null;
    this.workflowName = meta.name || 'Untitled Workflow';
    this.workflowCreated = meta.created || new Date().toISOString();
    this.workflowModified = meta.modified || new Date().toISOString();

    (data.nodes || []).forEach(function (nd) {
        var n = new WorkflowNode(nd.type, nd.x, nd.y, nd.config);
        n.id = nd.id; if (nd.width) n.width = nd.width; if (nd.height) n.height = nd.height; if (nd.title) n.title = nd.title;
        self.nodes.set(n.id, n);
    });
    (data.edges || []).forEach(function (ed) {
        var e = new WorkflowEdge(ed.fromNode, ed.fromPort, ed.toNode, ed.toPort);
        e.id = ed.id; self.edges.set(e.id, e);
        var fn = self.nodes.get(ed.fromNode), tn = self.nodes.get(ed.toNode);
        if (fn && fn.outputs[ed.fromPort]) fn.outputs[ed.fromPort].connected = true;
        if (tn && tn.inputs[ed.toPort]) tn.inputs[ed.toPort].connected = true;
    });
    this.zoomToFit();
    this._requestRender();
};

// Alias for app.js compatibility
WorkflowBuilder.prototype.load = function load(data) { this.fromJSON(data); };


// ---------------------------------------------------------------------------
//  API Integration: Save / Load / Execute
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.saveWorkflow = function saveWorkflow(name) {
    if (name) this.workflowName = name;
    this.workflowModified = new Date().toISOString();
    var payload = this.toJSON();
    var method = this.workflowId ? 'PUT' : 'POST';
    var endpoint = this.workflowId ? this.apiBase + '/workflows/' + this.workflowId : this.apiBase + '/workflows';
    var self = this;

    return fetch(endpoint, { method: method, headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify(payload) })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { if (d.id) self.workflowId = d.id; console.log('[WorkflowBuilder] Saved: ' + self.workflowName); return d; })
        .catch(function (err) { console.error('[WorkflowBuilder] Save failed:', err); throw err; });
};

WorkflowBuilder.prototype.loadWorkflow = function loadWorkflow(id) {
    var self = this;
    return fetch(this.apiBase + '/workflows/' + id, { headers: { 'Accept': 'application/json' } })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { self.fromJSON(d); console.log('[WorkflowBuilder] Loaded: ' + (d.metadata ? d.metadata.name : id)); return d; })
        .catch(function (err) { console.error('[WorkflowBuilder] Load failed:', err); throw err; });
};

WorkflowBuilder.prototype.listWorkflows = function listWorkflows() {
    return fetch(this.apiBase + '/workflows', { headers: { 'Accept': 'application/json' } })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) { return d.workflows || d.data || d || []; })
        .catch(function () { return []; });
};

WorkflowBuilder.prototype.executeWorkflow = function executeWorkflow() {
    var payload = this.toJSON();
    var self = this;
    this.nodes.forEach(function (n) { n.executionStatus = null; });
    this._executionRunning = true;
    var endpoint = this.workflowId ? this.apiBase + '/workflows/' + this.workflowId + '/execute' : this.apiBase + '/workflows/execute';

    return fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify(payload) })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (d) {
            self._executionId = d.execution_id || d.id || null;
            console.log('[WorkflowBuilder] Execution started: ' + self._executionId);
            if (typeof wsManager === 'undefined' || !wsManager.isConnected()) self._pollExecution();
            return d;
        })
        .catch(function (err) { console.error('[WorkflowBuilder] Execution failed:', err); self._executionRunning = false; throw err; });
};

WorkflowBuilder.prototype.highlightExecution = function highlightExecution(nodeId, status) {
    var n = this.nodes.get(nodeId);
    if (n) { n.executionStatus = status; this._requestRender(); }
};

WorkflowBuilder.prototype.onExecutionUpdate = function onExecutionUpdate(data) {
    if (!data) return;
    var nodeId = data.node_id || data.nodeId;
    var status = data.status || data.state;
    if (nodeId) this.highlightExecution(nodeId, status);
    if (data.workflow_complete || data.finished) { this._executionRunning = false; console.log('[WorkflowBuilder] Execution complete'); }
    this._requestRender();
};

WorkflowBuilder.prototype._pollExecution = function _pollExecution() {
    if (!this._executionId || !this._executionRunning) return;
    var self = this;
    var timer = setInterval(function () {
        if (!self._executionRunning) { clearInterval(timer); return; }
        fetch(self.apiBase + '/workflows/execution/' + self._executionId, { headers: { 'Accept': 'application/json' } })
            .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(function (d) {
                if (d.node_statuses) { Object.keys(d.node_statuses).forEach(function (nid) { self.highlightExecution(nid, d.node_statuses[nid]); }); }
                if (d.status === 'complete' || d.status === 'failed' || d.finished) { self._executionRunning = false; clearInterval(timer); }
            })
            .catch(function () { self._executionRunning = false; clearInterval(timer); });
    }, 2000);
};


// ---------------------------------------------------------------------------
//  Layout Utilities
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.zoomToFit = function zoomToFit() {
    if (this.nodes.size === 0) { this.viewTransform = { x: 0, y: 0, zoom: 1 }; this._requestRender(); return; }
    var b = this._getWorldBounds();
    if (!b) return;
    var cw = this.canvas.width / this._dpr, ch = this.canvas.height / this._dpr, pad = 60;
    var bw = b.maxX - b.minX + pad * 2, bh = b.maxY - b.minY + pad * 2;
    var zoom = Math.max(0.15, Math.min(Math.min(cw / bw, ch / bh, 1.5), 4));
    var cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
    this.viewTransform.zoom = zoom;
    this.viewTransform.x = cw / 2 - cx * zoom;
    this.viewTransform.y = ch / 2 - cy * zoom;
    this._requestRender();
};

WorkflowBuilder.prototype.autoLayout = function autoLayout() {
    if (this.nodes.size === 0) return;
    var adj = {}, inDeg = {};
    this.nodes.forEach(function (n) { adj[n.id] = []; inDeg[n.id] = 0; });
    this.edges.forEach(function (e) { if (adj[e.fromNode]) adj[e.fromNode].push(e.toNode); if (inDeg[e.toNode] !== undefined) inDeg[e.toNode]++; });

    var queue = [];
    Object.keys(inDeg).forEach(function (nid) { if (inDeg[nid] === 0) queue.push(nid); });

    var layers = [], visited = new Set();
    while (queue.length > 0) {
        var cur = queue.slice();
        layers.push(cur);
        var next = [];
        cur.forEach(function (nid) {
            visited.add(nid);
            (adj[nid] || []).forEach(function (nb) { inDeg[nb]--; if (inDeg[nb] === 0 && !visited.has(nb)) next.push(nb); });
        });
        queue = next;
    }

    var self = this;
    this.nodes.forEach(function (n) { if (!visited.has(n.id)) { if (layers.length === 0) layers.push([]); layers[layers.length - 1].push(n.id); } });

    this.pushUndo({ type: 'auto_layout' });
    var layerGap = 260, nodeGap = 120, startX = 100;

    for (var col = 0; col < layers.length; col++) {
        var layer = layers[col];
        for (var row = 0; row < layer.length; row++) {
            var n = this.nodes.get(layer[row]);
            if (n) {
                n.x = startX + col * layerGap;
                n.y = 200 + row * nodeGap - ((layer.length - 1) * nodeGap) / 2;
                if (self.snapToGrid) { n.x = Math.round(n.x / self.gridSize) * self.gridSize; n.y = Math.round(n.y / self.gridSize) * self.gridSize; }
            }
        }
    }
    this.workflowModified = new Date().toISOString();
    this.zoomToFit();
};

WorkflowBuilder.prototype.clear = function clear() {
    this.pushUndo({ type: 'clear' });
    this.nodes.clear(); this.edges.clear(); this.selectedNodes.clear(); this.selectedEdge = null;
    this.workflowModified = new Date().toISOString();
    this._requestRender();
};


// ---------------------------------------------------------------------------
//  Config Summary Helper
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._getConfigSummary = function _getConfigSummary(node) {
    var c = node.config || {}, p = [];
    switch (node.type) {
        case 'TRIGGER': if (c.trigger_type) p.push(c.trigger_type); if (c.cron) p.push(c.cron); break;
        case 'CONTENT': if (c.content_type) p.push(c.content_type); if (c.word_count) p.push(c.word_count + ' words'); break;
        case 'QUALITY': if (c.min_score) p.push('min: ' + c.min_score); break;
        case 'PUBLISH': if (c.post_status) p.push(c.post_status); break;
        case 'SOCIAL': if (c.platforms && Array.isArray(c.platforms)) p.push(c.platforms.join(', ')); break;
        case 'SEO': if (c.audit_type) p.push(c.audit_type); break;
        case 'DEVICE': if (c.device_id) p.push(c.device_id); break;
        case 'NOTIFY': if (c.channel) p.push(c.channel); break;
        case 'CONDITION': if (c.field && c.operator) p.push(c.field + ' ' + c.operator + ' ' + (c.value || '')); break;
        case 'TRANSFORM': if (c.transform_type) p.push(c.transform_type); break;
    }
    return p.length > 0 ? p.join(' | ') : '';
};


// ---------------------------------------------------------------------------
//  World Bounds & Drawing Helpers
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype._getWorldBounds = function _getWorldBounds() {
    if (this.nodes.size === 0) return null;
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    this.nodes.forEach(function (n) {
        if (n.x < minX) minX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.x + n.width > maxX) maxX = n.x + n.width;
        if (n.y + n.height > maxY) maxY = n.y + n.height;
    });
    return { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
};

WorkflowBuilder.prototype._roundRect = function _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
};

WorkflowBuilder.prototype._hexToRgba = function _hexToRgba(hex, alpha) {
    if (!hex) return 'rgba(96,125,139,' + alpha + ')';
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    var r = parseInt(hex.substr(0, 2), 16), g = parseInt(hex.substr(2, 2), 16), b = parseInt(hex.substr(4, 2), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
};


// ---------------------------------------------------------------------------
//  NodePalette: sidebar component for drag-to-create nodes
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.renderPalette = function renderPalette(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    this._paletteContainer = container;
    var self = this;

    var html = '<div class="node-palette">';
    NODE_TYPE_KEYS.forEach(function (key) {
        var td = WorkflowNodeType[key];
        html += '<div class="node-palette-item" data-node-type="' + key + '" draggable="true" title="' + td.description + '">' +
            '<span class="palette-icon" style="color:' + td.color + ';">' + td.icon + '</span>' +
            '<span class="palette-label">' + td.label + '</span></div>';
    });
    html += '</div>';
    container.innerHTML = html;

    var items = container.querySelectorAll('.node-palette-item');
    for (var i = 0; i < items.length; i++) {
        (function (item) {
            item.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', item.getAttribute('data-node-type')); e.dataTransfer.effectAllowed = 'copy'; });
            item.addEventListener('click', function () {
                var type = item.getAttribute('data-node-type');
                var cw = self.canvas.width / self._dpr, ch = self.canvas.height / self._dpr;
                var center = self.screenToWorld(cw / 2, ch / 2);
                self.addNode(type, center.x + (Math.random() - 0.5) * 100, center.y + (Math.random() - 0.5) * 100);
            });
        })(items[i]);
    }

    this.canvas.addEventListener('dragover', function (e) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
    this.canvas.addEventListener('drop', function (e) {
        e.preventDefault();
        var type = e.dataTransfer.getData('text/plain');
        if (!WorkflowNodeType[type]) return;
        var rect = self.canvas.getBoundingClientRect();
        var world = self.screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
        if (self.snapToGrid) { world.x = Math.round(world.x / self.gridSize) * self.gridSize; world.y = Math.round(world.y / self.gridSize) * self.gridSize; }
        self.addNode(type, world.x, world.y);
    });
};


// ---------------------------------------------------------------------------
//  Node Config Modal
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.showConfigModal = function showConfigModal(node) {
    if (!node) return;
    this._configNode = node;
    var fields = NODE_CONFIG_FIELDS[node.type] || [];
    var typeDef = WorkflowNodeType[node.type];

    // Inject site options from global SITES
    var siteOpts = [];
    if (typeof SITES !== 'undefined' && Array.isArray(SITES)) siteOpts = SITES.map(function (s) { return s.id; });
    fields = fields.map(function (f) { return (f.key === 'site_id' && f.options && f.options.length === 0) ? Object.assign({}, f, { options: siteOpts }) : f; });

    var overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.id = 'wf-config-modal';

    var modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.maxWidth = '520px';

    var html = '<h2 style="display:flex;align-items:center;gap:0.5rem;"><span style="color:' + typeDef.color + ';">' + typeDef.icon + '</span> Configure ' + typeDef.label + '</h2>' +
        '<p style="color:var(--text-secondary);font-size:0.82rem;margin-bottom:1rem;">' + typeDef.description + '</p>';

    // Title field
    html += '<label style="display:block;margin-bottom:0.75rem;"><span style="display:block;font-size:0.78rem;font-weight:500;margin-bottom:0.3rem;">Title</span>' +
        '<input type="text" id="wf-cfg-title" value="' + (node.title || '').replace(/"/g, '&quot;') + '" ' +
        'style="width:100%;padding:0.5rem 0.7rem;border-radius:6px;border:1px solid var(--card-border);background:var(--bg);color:var(--text);font-family:var(--sans);font-size:0.85rem;"></label>';

    // Dynamic fields
    fields.forEach(function (f) {
        var cv = node.config ? node.config[f.key] : '';
        if (cv === undefined || cv === null) cv = '';

        html += '<label style="display:block;margin-bottom:0.75rem;"><span style="display:block;font-size:0.78rem;font-weight:500;margin-bottom:0.3rem;">' + f.label + '</span>';
        var inputCss = 'width:100%;padding:0.5rem 0.7rem;border-radius:6px;border:1px solid var(--card-border);background:var(--bg);color:var(--text);font-family:var(--mono);font-size:0.82rem;';

        switch (f.type) {
            case 'text':
                html += '<input type="text" id="wf-cfg-' + f.key + '" value="' + String(cv).replace(/"/g, '&quot;') + '" placeholder="' + (f.placeholder || '') + '" style="' + inputCss + '">'; break;
            case 'number':
                html += '<input type="number" id="wf-cfg-' + f.key + '" value="' + (cv || '') + '" placeholder="' + (f.placeholder || '') + '"' +
                    (f.min !== undefined ? ' min="' + f.min + '"' : '') + (f.max !== undefined ? ' max="' + f.max + '"' : '') + ' style="' + inputCss + '">'; break;
            case 'select':
                html += '<select id="wf-cfg-' + f.key + '" style="' + inputCss + '"><option value="">-- Select --</option>';
                (f.options || []).forEach(function (opt) { html += '<option value="' + opt + '"' + (String(cv) === String(opt) ? ' selected' : '') + '>' + opt + '</option>'; });
                html += '</select>'; break;
            case 'textarea':
                html += '<textarea id="wf-cfg-' + f.key + '" rows="3" placeholder="' + (f.placeholder || '') + '" style="' + inputCss + 'resize:vertical;">' + String(cv) + '</textarea>'; break;
            case 'range':
                html += '<div style="display:flex;align-items:center;gap:0.5rem;"><input type="range" id="wf-cfg-' + f.key + '" min="' + (f.min || 0) + '" max="' + (f.max || 100) + '" step="' + (f.step || 1) + '" value="' + (cv || f.min || 0) + '" style="flex:1;">' +
                    '<span id="wf-cfg-' + f.key + '-val" style="font-family:var(--mono);font-size:0.82rem;min-width:30px;">' + (cv || f.min || 0) + '</span></div>'; break;
            case 'toggle':
                var isOn = cv === true || cv === 'true' || cv === 1;
                html += '<div class="toggle' + (isOn ? ' active' : '') + '" id="wf-cfg-' + f.key + '" data-value="' + (isOn ? 'true' : 'false') + '"></div>'; break;
            case 'checkboxes':
                var checked = Array.isArray(cv) ? cv : [];
                html += '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;">';
                (f.options || []).forEach(function (opt) { html += '<label style="display:flex;align-items:center;gap:0.3rem;font-size:0.82rem;cursor:pointer;"><input type="checkbox" data-group="' + f.key + '" value="' + opt + '"' + (checked.indexOf(opt) !== -1 ? ' checked' : '') + '>' + opt + '</label>'; });
                html += '</div>'; break;
        }
        html += '</label>';
    });

    html += '<div class="modal-actions" style="margin-top:1rem;"><button class="btn btn-ghost" id="wf-cfg-cancel">Cancel</button><button class="btn btn-primary" id="wf-cfg-save">Save</button></div>';

    modal.innerHTML = html;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    var titleIn = document.getElementById('wf-cfg-title');
    if (titleIn) titleIn.focus();

    // Range slider live update
    fields.forEach(function (f) {
        if (f.type === 'range') {
            var sl = document.getElementById('wf-cfg-' + f.key);
            var vd = document.getElementById('wf-cfg-' + f.key + '-val');
            if (sl && vd) sl.addEventListener('input', function () { vd.textContent = sl.value; });
        }
    });

    // Toggle click
    var toggles = overlay.querySelectorAll('.toggle');
    for (var t = 0; t < toggles.length; t++) {
        toggles[t].addEventListener('click', function () { var a = this.classList.contains('active'); this.classList.toggle('active'); this.setAttribute('data-value', a ? 'false' : 'true'); });
    }

    var self = this;
    document.getElementById('wf-cfg-cancel').addEventListener('click', function () { self.hideConfigModal(); });
    overlay.addEventListener('click', function (e) { if (e.target === overlay) self.hideConfigModal(); });
    document.getElementById('wf-cfg-save').addEventListener('click', function () { self._saveConfigModal(fields); });
    overlay.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') { e.preventDefault(); self._saveConfigModal(fields); }
        if (e.key === 'Escape') self.hideConfigModal();
    });

    this._configModal = overlay;
};


WorkflowBuilder.prototype._saveConfigModal = function _saveConfigModal(fields) {
    var node = this._configNode;
    if (!node) return;

    var titleIn = document.getElementById('wf-cfg-title');
    if (titleIn) node.title = titleIn.value.trim() || node.title;
    if (!node.config) node.config = {};

    fields.forEach(function (f) {
        var el;
        switch (f.type) {
            case 'toggle': el = document.getElementById('wf-cfg-' + f.key); if (el) node.config[f.key] = el.getAttribute('data-value') === 'true'; break;
            case 'checkboxes':
                var cbs = document.querySelectorAll('input[data-group="' + f.key + '"]');
                var v = []; for (var i = 0; i < cbs.length; i++) { if (cbs[i].checked) v.push(cbs[i].value); }
                node.config[f.key] = v; break;
            case 'number': case 'range':
                el = document.getElementById('wf-cfg-' + f.key); if (el && el.value !== '') node.config[f.key] = parseFloat(el.value); break;
            default:
                el = document.getElementById('wf-cfg-' + f.key); if (el) node.config[f.key] = el.value; break;
        }
    });

    this.pushUndo({ type: 'config_change', nodeId: node.id });
    this.workflowModified = new Date().toISOString();
    this.hideConfigModal();
    this._requestRender();
};


WorkflowBuilder.prototype.hideConfigModal = function hideConfigModal() {
    if (this._configModal) { document.body.removeChild(this._configModal); this._configModal = null; this._configNode = null; }
};


// ---------------------------------------------------------------------------
//  Toolbar Rendering
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.renderToolbar = function renderToolbar(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    var self = this;

    var btns = [
        { id: 'wf-tb-zoomin', icon: '+', title: 'Zoom In', fn: function () { self._zoomBy(1.2); } },
        { id: 'wf-tb-zoomout', icon: '-', title: 'Zoom Out', fn: function () { self._zoomBy(0.8); } },
        { id: 'wf-tb-zoomfit', icon: '\u25A3', title: 'Zoom to Fit', fn: function () { self.zoomToFit(); } },
        { sep: true },
        { id: 'wf-tb-undo', icon: '\u21B6', title: 'Undo (Ctrl+Z)', fn: function () { self.undo(); } },
        { id: 'wf-tb-redo', icon: '\u21B7', title: 'Redo (Ctrl+Y)', fn: function () { self.redo(); } },
        { sep: true },
        { id: 'wf-tb-layout', icon: '\u2630', title: 'Auto Layout', fn: function () { self.autoLayout(); } },
        { id: 'wf-tb-clear', icon: '\u2716', title: 'Clear All', fn: function () { if (confirm('Clear all nodes?')) self.clear(); } },
        { sep: true },
        { id: 'wf-tb-save', icon: '\u2193', title: 'Save Workflow', fn: function () { var n = prompt('Workflow name:', self.workflowName); if (n) self.saveWorkflow(n); } },
        { id: 'wf-tb-exec', icon: '\u25B6', title: 'Execute Workflow', fn: function () { self.executeWorkflow(); } }
    ];

    var html = '<div class="workflow-toolbar">';
    btns.forEach(function (b) {
        if (b.sep) html += '<div class="workflow-toolbar-separator"></div>';
        else html += '<button class="workflow-toolbar-btn" id="' + b.id + '" title="' + b.title + '">' + b.icon + '</button>';
    });
    html += '</div>';
    container.innerHTML = html;

    btns.forEach(function (b) {
        if (b.sep) return;
        var el = document.getElementById(b.id);
        if (el) el.addEventListener('click', function (e) { e.preventDefault(); b.fn(); });
    });
};


WorkflowBuilder.prototype._zoomBy = function _zoomBy(factor) {
    var cw = this.canvas.width / this._dpr, ch = this.canvas.height / this._dpr;
    var cx = cw / 2, cy = ch / 2;
    var newZoom = Math.max(0.15, Math.min(4, this.viewTransform.zoom * factor));
    var wb = this.screenToWorld(cx, cy);
    this.viewTransform.zoom = newZoom;
    var wa = this.screenToWorld(cx, cy);
    this.viewTransform.x += (wa.x - wb.x) * newZoom;
    this.viewTransform.y += (wa.y - wb.y) * newZoom;
    this._requestRender();
};


// ---------------------------------------------------------------------------
//  Demo Workflow
// ---------------------------------------------------------------------------

WorkflowBuilder.prototype.loadDemoWorkflow = function loadDemoWorkflow() {
    this.clear();
    this.undoStack = [];

    var trigger = this.addNode('TRIGGER', 60, 200, { trigger_type: 'schedule', cron: '0 9 * * 1-5' });
    trigger.title = 'Daily 9 AM';
    var content = this.addNode('CONTENT', 320, 160, { site_id: 'witchcraft', content_type: 'article', word_count: 1500 });
    content.title = 'Generate Article';
    var quality = this.addNode('QUALITY', 580, 160, { min_score: 80, retry_count: 2 });
    quality.title = 'Quality Gate';
    var publish = this.addNode('PUBLISH', 840, 100, { site_id: 'witchcraft', post_status: 'publish' });
    publish.title = 'Publish to WP';
    var social = this.addNode('SOCIAL', 1100, 100, { platforms: ['twitter', 'pinterest'], post_count: 3 });
    social.title = 'Social Campaign';
    var notify = this.addNode('NOTIFY', 1100, 280, { channel: 'slack', recipients: '#content-team' });
    notify.title = 'Alert: Failed QA';
    var seo = this.addNode('SEO', 840, 280, { audit_type: 'on_page', fix_auto: true });
    seo.title = 'SEO Optimize';

    this.connectNodes(trigger.id, 0, content.id, 0);
    this.connectNodes(content.id, 0, quality.id, 0);
    this.connectNodes(quality.id, 0, publish.id, 0);
    this.connectNodes(quality.id, 1, notify.id, 0);
    this.connectNodes(publish.id, 0, social.id, 0);
    this.connectNodes(content.id, 1, seo.id, 0);

    this.undoStack = [];
    this.redoStack = [];
    this.workflowName = 'Daily Content Pipeline';
    this.zoomToFit();
};


// ---------------------------------------------------------------------------
//  Global Registration
// ---------------------------------------------------------------------------

var workflowBuilder = new WorkflowBuilder('workflow-canvas');

window.WorkflowBuilder = WorkflowBuilder;
window.WorkflowNodeType = WorkflowNodeType;
window.workflowBuilder = workflowBuilder;

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
        if (document.getElementById('workflow-canvas')) workflowBuilder.init();
    });
} else {
    if (document.getElementById('workflow-canvas')) workflowBuilder.init();
}
