// =============================================================================
//  Empire App — OpenClaw Empire Dashboard v2
//  Main application controller: state, tab routing, API calls, rendering.
// =============================================================================

'use strict';

var API_BASE = 'http://localhost:8765/api';

// ---------------------------------------------------------------------------
//  Site registry (matches site-registry.json and the inline SITES array)
// ---------------------------------------------------------------------------

var SITES = [
    { id: 'witchcraft', domain: 'witchcraftforbeginners.com', color: '#4A1C6F', freq: 'daily', name: 'Witchcraft', niche: 'witchcraft-spirituality' },
    { id: 'smarthome', domain: 'smarthomewizards.com', color: '#0066CC', freq: '3x/wk', name: 'Smart Home', niche: 'smart-home-tech' },
    { id: 'aiaction', domain: 'aiinactionhub.com', color: '#00F0FF', freq: 'daily', name: 'AI Action', niche: 'ai-technology' },
    { id: 'aidiscovery', domain: 'aidiscoverydigest.com', color: '#E94560', freq: '3x/wk', name: 'AI Discovery', niche: 'ai-discovery' },
    { id: 'wealthai', domain: 'wealthfromai.com', color: '#00C853', freq: '3x/wk', name: 'Wealth AI', niche: 'ai-money' },
    { id: 'family', domain: 'family-flourish.com', color: '#E8887C', freq: '3x/wk', name: 'Family Flourish', niche: 'family-wellness' },
    { id: 'mythical', domain: 'mythicalarchives.com', color: '#8B4513', freq: '2x/wk', name: 'Mythical', niche: 'mythology' },
    { id: 'bulletjournals', domain: 'bulletjournals.net', color: '#FF6B6B', freq: '2x/wk', name: 'Bullet Journals', niche: 'productivity-journaling' },
    { id: 'crystalwitchcraft', domain: 'crystalwitchcraft.com', color: '#9B59B6', freq: '2x/wk', name: 'Crystal', niche: 'crystal-magic' },
    { id: 'herbalwitchery', domain: 'herbalwitchery.com', color: '#2ECC71', freq: '2x/wk', name: 'Herbal', niche: 'herbal-magic' },
    { id: 'moonphasewitch', domain: 'moonphasewitch.com', color: '#C0C0C0', freq: '2x/wk', name: 'Moon Phase', niche: 'lunar-magic' },
    { id: 'tarotbeginners', domain: 'tarotforbeginners.net', color: '#FFD700', freq: '2x/wk', name: 'Tarot', niche: 'tarot-divination' },
    { id: 'spellsrituals', domain: 'spellsandrituals.com', color: '#8B0000', freq: '2x/wk', name: 'Spells', niche: 'spells-rituals' },
    { id: 'paganpathways', domain: 'paganpathways.net', color: '#556B2F', freq: '2x/wk', name: 'Pagan', niche: 'pagan-spirituality' },
    { id: 'witchyhomedecor', domain: 'witchyhomedecor.com', color: '#DDA0DD', freq: '2x/wk', name: 'Witchy Decor', niche: 'witchy-decor' },
    { id: 'seasonalwitchcraft', domain: 'seasonalwitchcraft.com', color: '#FF8C00', freq: '2x/wk', name: 'Seasonal', niche: 'seasonal-wheel-of-year' }
];

// ---------------------------------------------------------------------------
//  Mission templates (from workflow_templates.py)
// ---------------------------------------------------------------------------

var MISSION_TYPES = [
    { type: 'content_publish', label: 'Content Publish', color: '#3b82f6', icon: '\u270D\uFE0F', description: 'Generate, optimize, and publish an article' },
    { type: 'social_growth', label: 'Social Growth', color: '#22c55e', icon: '\uD83D\uDCC8', description: 'Engage, post, and analyze on social platforms' },
    { type: 'account_creation', label: 'Account Creation', color: '#8b5cf6', icon: '\uD83D\uDC64', description: 'Create identities, emails, and platform signups' },
    { type: 'app_exploration', label: 'App Exploration', color: '#f97316', icon: '\uD83D\uDD0D', description: 'Explore apps and build playbooks' },
    { type: 'monetization', label: 'Monetization', color: '#eab308', icon: '\uD83D\uDCB0', description: 'Check revenue, optimize, and report' },
    { type: 'site_maintenance', label: 'Site Maintenance', color: '#6b7280', icon: '\uD83D\uDD27', description: 'Backup, audit, and health-check sites' },
    { type: 'revenue_check', label: 'Revenue Check', color: '#10b981', icon: '\uD83D\uDCCA', description: 'Forecast, detect anomalies, generate reports' },
    { type: 'device_maintenance', label: 'Device Maintenance', color: '#64748b', icon: '\uD83D\uDCF1', description: 'Health-check, update, and clean devices' },
    { type: 'substack_daily', label: 'Substack Daily', color: '#ec4899', icon: '\uD83D\uDCE8', description: 'Write, publish, and promote a newsletter' }
];

// ---------------------------------------------------------------------------
//  Utility helpers
// ---------------------------------------------------------------------------

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function fmt$(n) {
    return '$' + Number(n || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtDate(d) {
    if (!d) return '--';
    var dt = new Date(d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function fmtTime(d) {
    if (!d) return '--:--';
    var dt = new Date(d);
    return dt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function fmtDateTime(d) {
    if (!d) return '--';
    var dt = new Date(d);
    return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
           dt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function relTime(d) {
    if (!d) return '--';
    var diff = Date.now() - new Date(d).getTime();
    if (diff < 0) {
        var abs = Math.abs(diff);
        if (abs < 3600000) return 'in ' + Math.ceil(abs / 60000) + 'm';
        if (abs < 86400000) return 'in ' + Math.ceil(abs / 3600000) + 'h';
        return 'in ' + Math.ceil(abs / 86400000) + 'd';
    }
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
    if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
    return Math.floor(diff / 86400000) + 'd ago';
}

function animateValue(el, end, prefix, suffix, duration) {
    prefix = prefix || '';
    suffix = suffix || '';
    duration = duration || 800;
    var startTime = performance.now();
    var isInt = Number.isInteger(end);
    function step(now) {
        var elapsed = now - startTime;
        var progress = Math.min(elapsed / duration, 1);
        var eased = 1 - Math.pow(1 - progress, 3);
        var current = end * eased;
        el.textContent = prefix + (isInt ? Math.round(current).toLocaleString() : current.toFixed(2)) + suffix;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function truncate(str, maxLen) {
    if (!str) return '';
    if (str.length <= maxLen) return str;
    return str.substring(0, maxLen - 3) + '...';
}

function findSite(siteId) {
    for (var i = 0; i < SITES.length; i++) {
        if (SITES[i].id === siteId) return SITES[i];
    }
    return null;
}

// ---------------------------------------------------------------------------
//  EmpireApp class
// ---------------------------------------------------------------------------

function EmpireApp() {
    // State
    this.currentTab = 'overview';
    this.services = {};
    this.stats = {};
    this.recentActivity = [];
    this.devices = [];
    this.pipelines = [];
    this.workflows = [];
    this.abTests = [];
    this.calendarEntries = [];
    this.notifications = [];
    this.filterSite = null;
    this.token = localStorage.getItem('openclaw_token') || '';
    this.apiOnline = false;
    this.charts = {};
    this.unreadNotifs = 0;

    // Timers
    this._healthTimer = null;
    this._statsTimer = null;
    this._activityTimer = null;
    this._chartsTimer = null;
}

// ---- Initialization -------------------------------------------------------

EmpireApp.prototype.init = function init() {
    var self = this;

    // Register tab handlers
    this.registerTabHandlers();

    // Register global UI handlers
    this._registerUIHandlers();

    // Connect WebSocket
    this._connectWebSocket();

    // Initial data load
    this._updateTimestamp();
    this._initialLoad();

    // Auto-refresh timers
    this._healthTimer = setInterval(function () { self.fetchHealth(); }, 30000);
    this._statsTimer = setInterval(function () { self.fetchStats(); }, 60000);
    this._activityTimer = setInterval(function () {
        self.fetchActivity();
        self.fetchNotifications();
    }, 60000);
    this._chartsTimer = setInterval(function () {
        self.fetchRevenue(30);
        self.fetchContentCalendar(null, 7);
    }, 300000);

    console.log('[App] OpenClaw Empire Dashboard initialized');
};

EmpireApp.prototype._initialLoad = function _initialLoad() {
    var self = this;
    Promise.allSettled([
        this.fetchHealth(),
        this.fetchStats(),
        this.fetchDevices(),
        this.fetchPipelines(),
        this.fetchRevenue(30),
        this.fetchContentCalendar(null, 7),
        this.fetchWorkflows(),
        this.fetchABTests(),
        this.fetchActivity(),
        this.fetchNotifications()
    ]).then(function () {
        self.renderOverview();
        self._updateTimestamp();
    }).catch(function () {
        // Errors handled individually in each fetch
    });
};

// ---- Tab Routing ----------------------------------------------------------

EmpireApp.prototype.registerTabHandlers = function registerTabHandlers() {
    var self = this;
    var tabLinks = $$('[data-tab]');
    for (var i = 0; i < tabLinks.length; i++) {
        tabLinks[i].addEventListener('click', function (e) {
            e.preventDefault();
            var tabName = this.getAttribute('data-tab');
            self.switchTab(tabName);
        });
    }
};

EmpireApp.prototype.switchTab = function switchTab(tabName) {
    this.currentTab = tabName;

    // Update nav active states
    var links = $$('[data-tab]');
    for (var i = 0; i < links.length; i++) {
        if (links[i].getAttribute('data-tab') === tabName) {
            links[i].classList.add('active');
        } else {
            links[i].classList.remove('active');
        }
    }

    // Show/hide tab panels
    var panels = $$('.tab-panel');
    for (var j = 0; j < panels.length; j++) {
        if (panels[j].id === 'panel-' + tabName) {
            panels[j].style.display = '';
            panels[j].classList.add('active');
        } else {
            panels[j].style.display = 'none';
            panels[j].classList.remove('active');
        }
    }

    // Load tab-specific data
    switch (tabName) {
        case 'overview':
            this.renderOverview();
            break;
        case 'content':
            this.fetchContentCalendar(null, 7);
            this.fetchPipelines();
            break;
        case 'revenue':
            this.fetchRevenue(30);
            break;
        case 'devices':
            this.fetchDevices();
            break;
        case 'missions':
            this.renderMissionTemplates();
            break;
        case 'workflows':
            this.fetchWorkflows();
            break;
        case 'ab-tests':
            this.fetchABTests();
            break;
        case 'phone':
            if (typeof phoneMirror !== 'undefined') {
                phoneMirror.init('phone-mirror-canvas');
                phoneMirror.showControls();
            }
            break;
    }
};

// ---- API Methods ----------------------------------------------------------

EmpireApp.prototype._apiCall = function _apiCall(endpoint, options) {
    var self = this;
    options = options || {};
    var method = options.method || 'GET';
    var body = options.body || null;
    var timeout = options.timeout || 10000;

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, timeout);

    var headers = {
        'Accept': 'application/json'
    };
    if (this.token) {
        headers['Authorization'] = 'Bearer ' + this.token;
    }
    if (body && typeof body === 'object') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify(body);
    }

    return fetch(API_BASE + endpoint, {
        method: method,
        headers: headers,
        body: body,
        signal: controller.signal
    }).then(function (resp) {
        clearTimeout(timer);
        if (!resp.ok) {
            throw new Error('HTTP ' + resp.status + ' ' + resp.statusText);
        }
        return resp.json();
    }).then(function (data) {
        self._setOnline(true);
        return data;
    }).catch(function (err) {
        clearTimeout(timer);
        if (err.name === 'AbortError') {
            err = new Error('Request timeout: ' + endpoint);
        }
        self._setOnline(false);
        throw err;
    });
};

EmpireApp.prototype.fetchHealth = function fetchHealth() {
    var self = this;
    var t0 = performance.now();
    return this._apiCall('/health').then(function (data) {
        var latency = Math.round(performance.now() - t0);
        self.services = data;
        self._setOnline(true);
        self._renderHealthCard(data, latency);
        return data;
    }).catch(function (err) {
        self._renderHealthCardError();
        throw err;
    });
};

EmpireApp.prototype.fetchStats = function fetchStats() {
    var self = this;
    return this._apiCall('/stats').then(function (data) {
        self.stats = data;
        self._renderQuickStatsFromData(data);
        return data;
    }).catch(function () {
        // Non-critical — stats may not be available
    });
};

EmpireApp.prototype.fetchDevices = function fetchDevices() {
    var self = this;
    return this._apiCall('/devices').then(function (data) {
        self.devices = data.devices || data.data || data || [];
        self.renderDeviceFleet(self.devices);
        return self.devices;
    }).catch(function (err) {
        self.devices = [];
        self.renderDeviceFleet([]);
        throw err;
    });
};

EmpireApp.prototype.fetchPipelines = function fetchPipelines() {
    var self = this;
    return this._apiCall('/pipelines').then(function (data) {
        self.pipelines = data.pipelines || data.data || data || [];
        self.renderContentPipelines(self.pipelines);
        return self.pipelines;
    }).catch(function () {
        self.pipelines = [];
        self.renderContentPipelines([]);
    });
};

EmpireApp.prototype.fetchRevenue = function fetchRevenue(days) {
    days = days || 30;
    var self = this;
    return this._apiCall('/revenue?days=' + days).then(function (data) {
        self.renderRevenueCharts(data);
        return data;
    }).catch(function () {
        // Revenue chart will show error state
    });
};

EmpireApp.prototype.fetchContentCalendar = function fetchContentCalendar(site, days) {
    days = days || 7;
    var endpoint = '/content/calendar?days=' + days;
    if (site) endpoint += '&site=' + site;
    var self = this;
    return this._apiCall(endpoint).then(function (data) {
        self.calendarEntries = data.items || data.scheduled || data.posts || data || [];
        self.renderCalendar(self.calendarEntries);
        return self.calendarEntries;
    }).catch(function () {
        self.calendarEntries = [];
        self.renderCalendar([]);
    });
};

EmpireApp.prototype.fetchWorkflows = function fetchWorkflows() {
    var self = this;
    return this._apiCall('/workflows').then(function (data) {
        self.workflows = data.workflows || data.data || data || [];
        self.renderWorkflowList(self.workflows);
        return self.workflows;
    }).catch(function () {
        self.workflows = [];
        self.renderWorkflowList([]);
    });
};

EmpireApp.prototype.fetchABTests = function fetchABTests() {
    var self = this;
    return this._apiCall('/ab-tests').then(function (data) {
        self.abTests = data.tests || data.experiments || data.data || data || [];
        self.renderABTests(self.abTests);
        return self.abTests;
    }).catch(function () {
        self.abTests = [];
        self.renderABTests([]);
    });
};

EmpireApp.prototype.fetchActivity = function fetchActivity() {
    var self = this;
    return this._apiCall('/activity?limit=50').then(function (data) {
        var events = data.events || data.activity || data.data || data || [];
        self.recentActivity = events.slice(0, 50);
        self.renderActivityFeed(self.recentActivity);
        return self.recentActivity;
    }).catch(function () {
        // Try combining from other sources
        self._fetchActivityFallback();
    });
};

EmpireApp.prototype.fetchNotifications = function fetchNotifications() {
    var self = this;
    return this._apiCall('/notifications').then(function (data) {
        self.notifications = data.notifications || data.data || data || [];
        self._renderNotifications(self.notifications);
        return self.notifications;
    }).catch(function () {
        // Non-critical
    });
};

EmpireApp.prototype.triggerPipeline = function triggerPipeline(siteId, title) {
    var self = this;
    return this._apiCall('/pipelines/trigger', {
        method: 'POST',
        body: { site_id: siteId, title: title }
    }).then(function (data) {
        self._pushActivity({
            type: 'pipeline',
            message: 'Pipeline triggered for ' + siteId + ': ' + (title || 'untitled'),
            timestamp: new Date().toISOString(),
            severity: 'info'
        });
        self.fetchPipelines();
        return data;
    }).catch(function (err) {
        console.error('[App] Failed to trigger pipeline:', err);
        throw err;
    });
};

EmpireApp.prototype.triggerMission = function triggerMission(type, params) {
    var self = this;
    return this._apiCall('/missions/execute', {
        method: 'POST',
        body: { type: type, params: params || {} }
    }).then(function (data) {
        self._pushActivity({
            type: 'mission',
            message: 'Mission started: ' + type,
            timestamp: new Date().toISOString(),
            severity: 'info'
        });
        return data;
    }).catch(function (err) {
        console.error('[App] Failed to trigger mission:', err);
        throw err;
    });
};

EmpireApp.prototype.sendPhoneCommand = function sendPhoneCommand(deviceId, command) {
    return this._apiCall('/phone/command', {
        method: 'POST',
        body: { device_id: deviceId, command: command }
    });
};

// ---- Rendering: Overview --------------------------------------------------

EmpireApp.prototype.renderOverview = function renderOverview() {
    this.renderServiceCards(this.services);
    this.renderActivityFeed(this.recentActivity);
    this.renderQuickStats(this.stats);
    this.renderContentPipelines(this.pipelines);
};

EmpireApp.prototype.renderServiceCards = function renderServiceCards(services) {
    var container = document.getElementById('serviceCards');
    if (!container) return;

    var svcList = services.services || services.components || [];
    if (!Array.isArray(svcList)) {
        // Try to convert from object
        if (typeof services === 'object') {
            svcList = Object.keys(services).filter(function (k) {
                return typeof services[k] === 'object' && services[k] !== null;
            }).map(function (k) {
                return Object.assign({ name: k }, services[k]);
            });
        }
    }

    if (svcList.length === 0) {
        container.innerHTML = '<div class="card-loading">No services data</div>';
        return;
    }

    container.innerHTML = svcList.map(function (svc) {
        var name = svc.name || svc.id || 'Unknown';
        var status = svc.status || svc.state || 'unknown';
        var healthy = status === 'healthy' || status === 'running' || status === 'online' || status === 'ok';
        var dotClass = healthy ? 'green' : (status === 'degraded' ? 'yellow' : 'red');
        var uptime = svc.uptime || svc.uptime_percent || '--';
        var latency = svc.latency || svc.response_time || '--';

        return '<div class="card" style="padding:0.75rem;">' +
            '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.35rem;">' +
                '<span class="status-dot ' + dotClass + '"></span>' +
                '<span style="font-weight:600;font-size:0.85rem;">' + escapeHtml(name) + '</span>' +
            '</div>' +
            '<div style="font-size:0.75rem;color:var(--text-secondary);">' +
                'Status: ' + escapeHtml(status) +
                (latency !== '--' ? ' | ' + latency + 'ms' : '') +
                (uptime !== '--' ? ' | ' + uptime + '% uptime' : '') +
            '</div>' +
        '</div>';
    }).join('');
};

EmpireApp.prototype.renderActivityFeed = function renderActivityFeed(events) {
    var feed = document.getElementById('activityFeed');
    if (!feed) return;

    events = events || [];
    if (events.length === 0) {
        feed.innerHTML = '<li style="color:var(--text-dim);padding:0.5rem 0;">No recent activity</li>';
        return;
    }

    feed.innerHTML = events.slice(0, 30).map(function (ev) {
        var evType = ev.type || 'info';
        var dotColor;
        switch (evType) {
            case 'publish': dotColor = 'var(--success)'; break;
            case 'alert': case 'warning': dotColor = 'var(--warning)'; break;
            case 'error': case 'critical': dotColor = 'var(--critical)'; break;
            default: dotColor = 'var(--primary)'; break;
        }
        var siteId = ev.site_id || ev.site || null;
        var siteTag = siteId ? '<span class="feed-site">' + escapeHtml(siteId) + '</span> ' : '';
        var message = ev.message || ev.text || ev.description || '--';
        var time = ev.timestamp || ev.time || null;

        return '<li class="feed-item">' +
            '<span class="feed-dot" style="background:' + dotColor + ';"></span>' +
            '<span class="feed-text">' + siteTag + escapeHtml(message) + '</span>' +
            '<span class="feed-time">' + relTime(time) + '</span>' +
        '</li>';
    }).join('');
};

EmpireApp.prototype.renderQuickStats = function renderQuickStats(stats) {
    if (!stats || typeof stats !== 'object') return;

    // Revenue today
    if (stats.revenue_today !== undefined) {
        var revEl = document.getElementById('revToday');
        if (revEl) animateValue(revEl, stats.revenue_today, '$', '', 900);
    }

    // Revenue change
    if (stats.revenue_change !== undefined) {
        var changeEl = document.getElementById('revChange');
        if (changeEl) {
            var diff = stats.revenue_change;
            changeEl.className = 'metric-change ' + (diff > 0 ? 'up' : diff < 0 ? 'down' : 'flat');
            changeEl.textContent = (diff > 0 ? '+' : '') + diff.toFixed(1) + '%';
        }
    }

    // Posts this week
    if (stats.posts_this_week !== undefined) {
        var postsEl = document.getElementById('postsWeek');
        if (postsEl) animateValue(postsEl, stats.posts_this_week, '', '', 700);
    }

    // Active tasks
    if (stats.active_tasks !== undefined) {
        var tasksEl = document.getElementById('activeTasks');
        if (tasksEl) animateValue(tasksEl, stats.active_tasks, '', '', 500);
    }
};

EmpireApp.prototype.renderContentPipelines = function renderContentPipelines(pipelines) {
    var container = document.getElementById('pipelineList');
    if (!container) return;

    pipelines = pipelines || [];
    if (pipelines.length === 0) {
        container.innerHTML = '<div style="color:var(--text-dim);padding:1rem;text-align:center;">No active pipelines</div>';
        return;
    }

    container.innerHTML = pipelines.map(function (p) {
        var id = p.pipeline_id || p.id || '--';
        var site = p.site_id || p.site || '--';
        var stage = p.stage || p.current_stage || 'pending';
        var stageIdx = typeof p.stage_index === 'number' ? p.stage_index : 0;
        var totalStages = typeof p.total_stages === 'number' ? p.total_stages : 5;
        var pct = Math.round((stageIdx / totalStages) * 100);
        var status = p.status || (p.success === false ? 'failed' : 'running');
        var barColor = status === 'failed' ? 'var(--critical)' : status === 'complete' ? 'var(--success)' : 'var(--primary)';
        var siteObj = findSite(site);
        var siteColor = siteObj ? siteObj.color : '#6366f1';

        return '<div style="padding:0.6rem 0;border-bottom:1px solid rgba(30,30,46,0.5);">' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">' +
                '<span style="font-weight:500;font-size:0.82rem;">' +
                    '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:' + siteColor + ';margin-right:0.4rem;"></span>' +
                    escapeHtml(truncate(p.title || id, 40)) +
                '</span>' +
                '<span style="font-family:var(--mono);font-size:0.72rem;color:var(--text-secondary);">' + escapeHtml(stage) + ' (' + pct + '%)</span>' +
            '</div>' +
            '<div class="forge-bar"><div class="forge-bar-fill" style="width:' + pct + '%;background:' + barColor + ';"></div></div>' +
        '</div>';
    }).join('');
};

EmpireApp.prototype.renderCalendar = function renderCalendar(entries) {
    var body = document.getElementById('calendarBody');
    if (!body) return;

    var today = new Date();
    var dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    var days = [];

    for (var i = 0; i < 7; i++) {
        var d = new Date(today);
        d.setDate(d.getDate() + i);
        var dateStr = d.toISOString().split('T')[0];
        var postsOnDay = (entries || []).filter(function (p) {
            var pDate = (p.date || p.scheduled || '').split('T')[0];
            return pDate === dateStr;
        });
        days.push({ date: d, posts: postsOnDay, isToday: i === 0 });
    }

    var html = '<div class="cal-days">';
    for (var j = 0; j < days.length; j++) {
        var day = days[j];
        var ct = day.posts.length;
        var cls = 'cal-day';
        if (day.isToday) cls += ' today';
        if (ct > 0) cls += ' has-posts';
        if (ct >= 3) cls += ' many-posts';

        html += '<div class="' + cls + '" title="' + ct + ' post(s) scheduled">' +
            '<div class="cal-day-label">' + dayNames[day.date.getDay()] + '</div>' +
            '<div>' + day.date.getDate() + '</div>' +
            (ct > 0 ? '<div class="cal-post-count">' + ct + '</div>' : '') +
        '</div>';
    }
    html += '</div>';

    // Upcoming entries list
    if (entries && entries.length > 0) {
        html += '<div style="margin-top:0.75rem;font-size:0.78rem;color:var(--text-secondary);">';
        var shown = entries.slice(0, 8);
        for (var k = 0; k < shown.length; k++) {
            var p = shown[k];
            var siteId = p.site_id || p.site || null;
            var siteObj = siteId ? findSite(siteId) : null;
            var siteColor = siteObj ? siteObj.color : 'var(--primary)';

            html += '<div style="padding:0.3rem 0;border-bottom:1px solid rgba(30,30,46,0.5);display:flex;justify-content:space-between;align-items:center;">' +
                '<span style="display:flex;align-items:center;gap:0.3rem;">';
            if (siteObj) {
                html += '<span style="width:6px;height:6px;border-radius:50%;background:' + siteColor + ';flex-shrink:0;"></span>';
            }
            html += '<span style="color:var(--text);font-weight:500;">' + escapeHtml(truncate(p.title || p.name || '--', 45)) + '</span>' +
                '</span>' +
                '<span style="font-family:var(--mono);font-size:0.7rem;white-space:nowrap;margin-left:0.5rem;">' + fmtDate(p.date || p.scheduled) + '</span>' +
            '</div>';
        }
        html += '</div>';
    }

    body.innerHTML = html;
};

EmpireApp.prototype.renderDeviceFleet = function renderDeviceFleet(devices) {
    var container = document.getElementById('deviceGrid');
    if (!container) return;

    devices = devices || [];
    if (devices.length === 0) {
        container.innerHTML = '<div style="color:var(--text-dim);padding:2rem;text-align:center;">No devices registered. Connect an Android device with Termux to begin.</div>';
        return;
    }

    var self = this;
    container.innerHTML = devices.map(function (device) {
        var id = device.id || device.device_id || 'unknown';
        var name = device.name || device.model || id;
        var online = device.online || device.status === 'online' || device.connected;
        var dotClass = online ? 'green' : 'red';
        var battery = device.battery !== undefined ? device.battery + '%' : '--';
        var os = device.os || device.android_version || '--';
        var lastSeen = device.last_seen || device.last_active || null;

        return '<div class="card" style="cursor:pointer;" data-device-id="' + escapeHtml(id) + '">' +
            '<div class="card-header">' +
                '<div class="card-title">' +
                    '<span class="status-dot ' + dotClass + '"></span>' +
                    escapeHtml(name) +
                '</div>' +
            '</div>' +
            '<div class="card-body" style="font-size:0.8rem;">' +
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;">' +
                    '<div><span style="color:var(--text-dim);">Battery:</span> ' + escapeHtml(battery) + '</div>' +
                    '<div><span style="color:var(--text-dim);">OS:</span> ' + escapeHtml(os) + '</div>' +
                    '<div style="grid-column:span 2;"><span style="color:var(--text-dim);">Last seen:</span> ' + relTime(lastSeen) + '</div>' +
                '</div>' +
                '<div style="margin-top:0.5rem;display:flex;gap:0.35rem;">' +
                    '<button class="btn btn-ghost" style="font-size:0.72rem;padding:0.25rem 0.6rem;" onclick="app.onDeviceClick(\'' + escapeHtml(id) + '\')">Mirror</button>' +
                    '<button class="btn btn-ghost" style="font-size:0.72rem;padding:0.25rem 0.6rem;" onclick="app.sendPhoneCommand(\'' + escapeHtml(id) + '\', \'screenshot\')">Screenshot</button>' +
                '</div>' +
            '</div>' +
        '</div>';
    }).join('');
};

EmpireApp.prototype.renderRevenueCharts = function renderRevenueCharts(data) {
    if (!data) return;

    // Revenue today card
    var todayRev = data.today || data.revenue_today || 0;
    var yesterdayRev = data.yesterday || data.revenue_yesterday || 0;
    var revEl = document.getElementById('revToday');
    if (revEl) animateValue(revEl, todayRev, '$', '', 900);

    var changeEl = document.getElementById('revChange');
    if (changeEl && yesterdayRev > 0) {
        var diff = ((todayRev - yesterdayRev) / yesterdayRev) * 100;
        changeEl.className = 'metric-change ' + (diff > 0 ? 'up' : diff < 0 ? 'down' : 'flat');
        changeEl.textContent = (diff > 0 ? '+' : '') + diff.toFixed(1) + '%';
    }

    // Revenue sparkline
    if (data.recent && data.recent.length > 1) {
        this._drawSparkLine('revSpark', data.recent);
    }

    // Delegate to chart factory for the main chart
    var daily = data.daily || data.data || data.days || [];
    if (daily.length > 0 && typeof chartFactory !== 'undefined') {
        var labels = daily.map(function (d) {
            return d.date ? new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : (d.label || '');
        });
        var values = daily.map(function (d) { return d.total || d.revenue || d.amount || 0; });

        chartFactory.createRevenueChart('revenueChart', { labels: labels, values: values });
    }

    // Forecast chart if data is available
    if (data.forecast && typeof chartFactory !== 'undefined') {
        chartFactory.createForecastChart('forecastChart', data.forecast);
    }
};

EmpireApp.prototype.renderABTests = function renderABTests(tests) {
    var container = document.getElementById('abTestList');
    if (!container) return;

    tests = tests || [];
    if (tests.length === 0) {
        container.innerHTML = '<div style="color:var(--text-dim);padding:2rem;text-align:center;">No active A/B tests</div>';
        return;
    }

    container.innerHTML = tests.map(function (test) {
        var name = test.name || test.title || 'Untitled Test';
        var status = test.status || 'running';
        var statusColor = status === 'complete' ? 'var(--success)' : status === 'running' ? 'var(--primary)' : 'var(--text-dim)';
        var winner = test.winner || null;
        var confidence = test.confidence || test.statistical_significance || null;
        var variants = test.variants || [];

        var variantHtml = variants.map(function (v) {
            var varName = v.name || v.variant || '--';
            var conversion = v.conversion_rate || v.rate || 0;
            var isWinner = winner && varName === winner;
            return '<div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;' +
                (isWinner ? 'color:var(--success);font-weight:600;' : '') + '">' +
                '<span>' + escapeHtml(varName) + (isWinner ? ' (Winner)' : '') + '</span>' +
                '<span style="font-family:var(--mono);font-size:0.78rem;">' + (conversion * 100).toFixed(2) + '%</span>' +
            '</div>';
        }).join('');

        return '<div class="card">' +
            '<div class="card-header">' +
                '<div class="card-title">' + escapeHtml(name) + '</div>' +
                '<span style="font-size:0.72rem;color:' + statusColor + ';font-weight:500;">' + escapeHtml(status) + '</span>' +
            '</div>' +
            '<div class="card-body" style="font-size:0.82rem;">' +
                (confidence ? '<div style="margin-bottom:0.5rem;color:var(--text-secondary);font-size:0.75rem;">Confidence: ' + (confidence * 100).toFixed(1) + '%</div>' : '') +
                variantHtml +
            '</div>' +
        '</div>';
    }).join('');
};

EmpireApp.prototype.renderWorkflowList = function renderWorkflowList(workflows) {
    var container = document.getElementById('workflowGrid');
    if (!container) return;

    workflows = workflows || [];
    if (workflows.length === 0) {
        container.innerHTML = '<div style="color:var(--text-dim);padding:2rem;text-align:center;">No saved workflows. Use the Workflow Builder to create one.</div>';
        return;
    }

    container.innerHTML = workflows.map(function (wf) {
        var name = wf.name || wf.title || 'Untitled';
        var description = wf.description || '';
        var nodeCount = wf.node_count || (wf.nodes ? wf.nodes.length : 0);
        var lastRun = wf.last_run || wf.last_executed || null;
        var status = wf.status || 'saved';
        var statusColor = status === 'running' ? 'var(--primary)' : status === 'complete' ? 'var(--success)' : 'var(--text-dim)';

        return '<div class="card">' +
            '<div class="card-header">' +
                '<div class="card-title">' + escapeHtml(name) + '</div>' +
                '<span style="font-size:0.72rem;color:' + statusColor + ';">' + escapeHtml(status) + '</span>' +
            '</div>' +
            '<div class="card-body" style="font-size:0.82rem;">' +
                (description ? '<div style="color:var(--text-secondary);margin-bottom:0.4rem;">' + escapeHtml(truncate(description, 100)) + '</div>' : '') +
                '<div style="display:flex;justify-content:space-between;color:var(--text-dim);font-size:0.75rem;">' +
                    '<span>' + nodeCount + ' nodes</span>' +
                    '<span>Last run: ' + (lastRun ? relTime(lastRun) : 'Never') + '</span>' +
                '</div>' +
                '<div style="margin-top:0.5rem;display:flex;gap:0.35rem;">' +
                    '<button class="btn btn-primary" style="font-size:0.72rem;padding:0.25rem 0.6rem;" onclick="app._executeWorkflow(\'' + escapeHtml(wf.id || '') + '\')">Execute</button>' +
                    '<button class="btn btn-ghost" style="font-size:0.72rem;padding:0.25rem 0.6rem;" onclick="app._editWorkflow(\'' + escapeHtml(wf.id || '') + '\')">Edit</button>' +
                '</div>' +
            '</div>' +
        '</div>';
    }).join('');
};

EmpireApp.prototype.renderMissionTemplates = function renderMissionTemplates() {
    var container = document.getElementById('missionGrid');
    if (!container) return;

    var self = this;
    container.innerHTML = MISSION_TYPES.map(function (m) {
        return '<div class="card" style="cursor:pointer;border-left:3px solid ' + m.color + ';" data-mission-type="' + m.type + '">' +
            '<div class="card-body">' +
                '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;">' +
                    '<span style="font-size:1.2rem;">' + m.icon + '</span>' +
                    '<span style="font-weight:600;font-size:0.9rem;">' + escapeHtml(m.label) + '</span>' +
                '</div>' +
                '<div style="color:var(--text-secondary);font-size:0.78rem;margin-bottom:0.6rem;">' + escapeHtml(m.description) + '</div>' +
                '<button class="btn btn-primary" style="font-size:0.72rem;padding:0.25rem 0.8rem;background:' + m.color + ';" ' +
                    'onclick="app.onMissionStart(\'' + m.type + '\')">Launch</button>' +
            '</div>' +
        '</div>';
    }).join('');
};

// ---- Event Handlers -------------------------------------------------------

EmpireApp.prototype.onWebSocketEvent = function onWebSocketEvent(eventType, data) {
    switch (eventType) {
        case 'agent_status':
            this._pushActivity({
                type: 'mission',
                message: 'Mission ' + (data.mission_id || '--') + ': ' + (data.status || 'update') + ' (' + (data.step || '--') + ')',
                timestamp: data.timestamp || new Date().toISOString(),
                severity: data.status === 'failed' ? 'error' : 'info'
            });
            break;

        case 'pipeline_progress':
            this._updatePipelineInList(data);
            break;

        case 'phone_mirror':
            if (typeof phoneMirror !== 'undefined') {
                phoneMirror.updateFrame(data.screenshot_base64);
            }
            break;

        case 'activity':
            this._pushActivity(data);
            break;

        case 'health_update':
            this.services = data;
            this.renderServiceCards(data);
            break;

        case 'revenue_update':
            this.renderRevenueCharts(data);
            break;

        case 'notification':
            this._pushNotification(data);
            break;
    }
};

EmpireApp.prototype.onPipelineTrigger = function onPipelineTrigger(siteId) {
    var site = findSite(siteId);
    var title = prompt('Article title for ' + (site ? site.name : siteId) + ':');
    if (!title) return;
    this.triggerPipeline(siteId, title);
};

EmpireApp.prototype.onMissionStart = function onMissionStart(type) {
    var mission = MISSION_TYPES.find(function (m) { return m.type === type; });
    if (!mission) {
        console.error('[App] Unknown mission type:', type);
        return;
    }

    // Build a simple params object — for now just site_id for relevant missions
    var params = {};
    if (type === 'content_publish' || type === 'social_growth' || type === 'site_maintenance') {
        var siteId = prompt('Site ID (e.g., witchcraft, smarthome):');
        if (!siteId) return;
        params.site_id = siteId;

        if (type === 'content_publish') {
            var title = prompt('Article title:');
            if (!title) return;
            params.title = title;
        }
    }

    this.triggerMission(type, params);
};

EmpireApp.prototype.onDeviceClick = function onDeviceClick(deviceId) {
    if (typeof phoneMirror !== 'undefined') {
        phoneMirror.setDevice(deviceId);
        phoneMirror.startStream(deviceId);
    }
    // Switch to phone tab if it exists
    var phoneTab = document.querySelector('[data-tab="phone"]');
    if (phoneTab) {
        this.switchTab('phone');
    }
};

// ---- Internal: Health card rendering --------------------------------------

EmpireApp.prototype._renderHealthCard = function _renderHealthCard(data, latency) {
    var dot = document.getElementById('apiDot');
    var statusEl = document.getElementById('apiStatus');
    var latencyEl = document.getElementById('apiLatency');
    var uptimeEl = document.getElementById('apiUptime');

    var status = data.status || 'healthy';
    if (dot) {
        dot.className = 'status-dot ' + (status === 'healthy' ? 'green' : status === 'degraded' ? 'yellow' : 'red');
    }
    if (statusEl) {
        statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    if (latencyEl) {
        latencyEl.textContent = 'Response: ' + latency + 'ms';
    }
    if (uptimeEl) {
        uptimeEl.textContent = 'Uptime: ' + (data.uptime_percent || data.uptime || '99.9') + '%';
    }
};

EmpireApp.prototype._renderHealthCardError = function _renderHealthCardError() {
    var dot = document.getElementById('apiDot');
    if (dot) dot.className = 'status-dot red';
    var statusEl = document.getElementById('apiStatus');
    if (statusEl) statusEl.textContent = 'Offline';
    var latencyEl = document.getElementById('apiLatency');
    if (latencyEl) latencyEl.textContent = 'Response: --ms';
    var uptimeEl = document.getElementById('apiUptime');
    if (uptimeEl) uptimeEl.textContent = 'Uptime: --%';
};

EmpireApp.prototype._renderQuickStatsFromData = function _renderQuickStatsFromData(data) {
    if (!data) return;

    if (data.revenue_today !== undefined || data.revenue) {
        var rev = data.revenue_today || (data.revenue ? data.revenue.today : 0);
        var revEl = document.getElementById('revToday');
        if (revEl && rev) animateValue(revEl, rev, '$', '', 900);
    }

    if (data.posts_this_week !== undefined || data.posts) {
        var posts = data.posts_this_week || (data.posts ? data.posts.this_week : 0);
        var postsEl = document.getElementById('postsWeek');
        if (postsEl && posts) animateValue(postsEl, posts, '', '', 700);
    }

    if (data.active_tasks !== undefined || data.tasks) {
        var tasks = data.active_tasks || (data.tasks ? data.tasks.active : 0);
        var tasksEl = document.getElementById('activeTasks');
        if (tasksEl && tasks !== undefined) animateValue(tasksEl, tasks, '', '', 500);
    }

    if (data.forge) {
        var f = data.forge;
        if (f.tasks_learned) {
            var fEl = document.getElementById('forgeTasks');
            if (fEl) animateValue(fEl, f.tasks_learned, '', '', 600);
        }
        if (f.success_rate) {
            var sEl = document.getElementById('forgeSuccess');
            if (sEl) animateValue(sEl, f.success_rate, '', '%', 600);
        }
    }
};

EmpireApp.prototype._renderNotifications = function _renderNotifications(notifs) {
    notifs = notifs || [];
    var unread = notifs.filter(function (n) { return n.unread !== false; }).length;
    this.unreadNotifs = unread;

    var badge = document.getElementById('notifBadge');
    if (badge) {
        badge.style.display = unread > 0 ? 'flex' : 'none';
        badge.textContent = unread;
    }

    var list = document.getElementById('notifList');
    if (!list) return;

    if (notifs.length === 0) {
        list.innerHTML = '<li style="color:var(--text-dim);padding:0.5rem 0;text-align:center;">No notifications</li>';
        return;
    }

    list.innerHTML = notifs.slice(0, 20).map(function (n) {
        var severity = n.severity || n.level || 'info';
        var unreadClass = n.unread !== false ? 'unread' : '';
        return '<li class="notif-item ' + severity + ' ' + unreadClass + '">' +
            '<div class="notif-header">' +
                '<span class="notif-title">' + escapeHtml(n.title || 'Notification') + '</span>' +
                '<span class="notif-time">' + relTime(n.time || n.timestamp) + '</span>' +
            '</div>' +
            '<div class="notif-body">' + escapeHtml(n.body || n.message || '') + '</div>' +
        '</li>';
    }).join('');
};

// ---- Internal: Activity management ----------------------------------------

EmpireApp.prototype._pushActivity = function _pushActivity(event) {
    this.recentActivity.unshift(event);
    if (this.recentActivity.length > 50) {
        this.recentActivity = this.recentActivity.slice(0, 50);
    }
    this.renderActivityFeed(this.recentActivity);
};

EmpireApp.prototype._pushNotification = function _pushNotification(notif) {
    this.notifications.unshift(notif);
    if (this.notifications.length > 50) {
        this.notifications = this.notifications.slice(0, 50);
    }
    this._renderNotifications(this.notifications);
};

EmpireApp.prototype._updatePipelineInList = function _updatePipelineInList(data) {
    var found = false;
    for (var i = 0; i < this.pipelines.length; i++) {
        var pid = this.pipelines[i].pipeline_id || this.pipelines[i].id;
        if (pid === data.pipeline_id) {
            this.pipelines[i].stage = data.stage;
            this.pipelines[i].stage_index = data.stage_index;
            this.pipelines[i].total_stages = data.total_stages;
            this.pipelines[i].success = data.success;
            found = true;
            break;
        }
    }
    if (!found && data.pipeline_id) {
        this.pipelines.unshift(data);
    }
    this.renderContentPipelines(this.pipelines);
};

EmpireApp.prototype._fetchActivityFallback = function _fetchActivityFallback() {
    // Try to build activity from WordPress dashboard and scheduler endpoints
    var self = this;
    var events = [];

    Promise.allSettled([
        this._apiCall('/wordpress/dashboard').then(function (data) {
            if (data.recent_activity) {
                events = events.concat(data.recent_activity);
            }
        }),
        this._apiCall('/scheduler/jobs').then(function (data) {
            var jobs = data.jobs || data || [];
            jobs.filter(function (j) { return j.last_run; }).forEach(function (j) {
                events.push({
                    type: 'task',
                    message: 'Job "' + (j.name || j.id) + '" completed',
                    site_id: j.site || null,
                    timestamp: j.last_run,
                    severity: j.last_result === 'failed' ? 'error' : 'info'
                });
            });
        })
    ]).then(function () {
        events.sort(function (a, b) { return new Date(b.timestamp || 0) - new Date(a.timestamp || 0); });
        self.recentActivity = events.slice(0, 50);
        self.renderActivityFeed(self.recentActivity);
    });
};

// ---- Internal: WebSocket integration --------------------------------------

EmpireApp.prototype._connectWebSocket = function _connectWebSocket() {
    if (typeof wsManager === 'undefined') return;

    var self = this;

    wsManager.on('connected', function () {
        self._setOnline(true);
        console.log('[App] WebSocket connected');
    });

    wsManager.on('disconnected', function () {
        console.log('[App] WebSocket disconnected');
    });

    // Route all typed events
    var eventTypes = ['agent_status', 'pipeline_progress', 'phone_mirror', 'activity', 'health_update', 'revenue_update', 'notification', 'device_update'];
    eventTypes.forEach(function (evType) {
        wsManager.on(evType, function (data) {
            self.onWebSocketEvent(evType, data);
        });
    });

    // Connect
    wsManager.connect();
};

// ---- Internal: Connection status ------------------------------------------

EmpireApp.prototype._setOnline = function _setOnline(ok) {
    this.apiOnline = ok;
    var dot = document.getElementById('connDot');
    var banner = document.getElementById('offlineBanner');

    if (ok) {
        if (dot) dot.className = 'status-dot green';
        if (banner) banner.classList.remove('visible');
    } else {
        if (dot) dot.className = 'status-dot red';
        if (banner) banner.classList.add('visible');
    }
};

// ---- Internal: UI handlers ------------------------------------------------

EmpireApp.prototype._registerUIHandlers = function _registerUIHandlers() {
    var self = this;

    // Token modal
    var btnSettings = document.getElementById('btnSettings');
    if (btnSettings) {
        btnSettings.addEventListener('click', function () {
            var modal = document.getElementById('tokenModal');
            if (modal) modal.classList.remove('hidden');
            var input = document.getElementById('tokenInput');
            if (input) input.focus();
        });
    }

    var btnTokenCancel = document.getElementById('btnTokenCancel');
    if (btnTokenCancel) {
        btnTokenCancel.addEventListener('click', function () {
            var modal = document.getElementById('tokenModal');
            if (modal) modal.classList.add('hidden');
        });
    }

    var btnTokenSave = document.getElementById('btnTokenSave');
    if (btnTokenSave) {
        btnTokenSave.addEventListener('click', function () {
            var input = document.getElementById('tokenInput');
            if (input && input.value.trim()) {
                self.token = input.value.trim();
                localStorage.setItem('openclaw_token', self.token);
            }
            var modal = document.getElementById('tokenModal');
            if (modal) modal.classList.add('hidden');
            self._refreshAll();
        });
    }

    var tokenInput = document.getElementById('tokenInput');
    if (tokenInput) {
        tokenInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && btnTokenSave) btnTokenSave.click();
        });
    }

    // Refresh button
    var btnRefresh = document.getElementById('btnRefresh');
    if (btnRefresh) {
        btnRefresh.addEventListener('click', function () { self._refreshAll(); });
    }

    // Notification bell
    var btnNotif = document.getElementById('btnNotif');
    if (btnNotif) {
        btnNotif.addEventListener('click', function () {
            var notifCard = document.getElementById('cardNotifications');
            if (notifCard) notifCard.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // Filter bar
    var btnClearFilter = document.getElementById('btnClearFilter');
    if (btnClearFilter) {
        btnClearFilter.addEventListener('click', function () { self._clearFilter(); });
    }

    // Card expand buttons
    var expandBtns = $$('.card-expand-btn');
    for (var i = 0; i < expandBtns.length; i++) {
        expandBtns[i].addEventListener('click', function () {
            var id = this.getAttribute('data-expand');
            var card = document.getElementById(id);
            if (card) card.classList.toggle('expanded');
            this.textContent = card && card.classList.contains('expanded') ? '\u2923' : '\u2922';
        });
    }
};

EmpireApp.prototype._refreshAll = function _refreshAll() {
    this._updateTimestamp();
    this._initialLoad();
};

EmpireApp.prototype._updateTimestamp = function _updateTimestamp() {
    var el = document.getElementById('lastRefresh');
    if (el) {
        el.textContent = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
};

EmpireApp.prototype._clearFilter = function _clearFilter() {
    this.filterSite = null;
    var bar = document.getElementById('filterBar');
    if (bar) bar.classList.remove('visible');
};

EmpireApp.prototype._setFilter = function _setFilter(siteId) {
    this.filterSite = siteId;
    var site = findSite(siteId);
    if (site) {
        var nameEl = document.getElementById('filterSiteName');
        if (nameEl) nameEl.textContent = site.name + ' (' + site.domain + ')';
        var bar = document.getElementById('filterBar');
        if (bar) bar.classList.add('visible');
    }
};

// ---- Internal: Workflow execution -----------------------------------------

EmpireApp.prototype._executeWorkflow = function _executeWorkflow(workflowId) {
    if (!workflowId) return;
    this._apiCall('/workflows/' + workflowId + '/execute', { method: 'POST' })
        .then(function () {
            console.log('[App] Workflow execution started:', workflowId);
        })
        .catch(function (err) {
            console.error('[App] Workflow execution failed:', err);
        });
};

EmpireApp.prototype._editWorkflow = function _editWorkflow(workflowId) {
    if (!workflowId) return;
    // Load workflow into builder
    if (typeof workflowBuilder !== 'undefined') {
        this._apiCall('/workflows/' + workflowId).then(function (data) {
            workflowBuilder.load(data);
        }).catch(function (err) {
            console.error('[App] Failed to load workflow:', err);
        });
    }
};

// ---- Internal: Sparkline drawing ------------------------------------------

EmpireApp.prototype._drawSparkLine = function _drawSparkLine(canvasId, data) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.offsetWidth;
    var h = canvas.offsetHeight;
    if (w === 0 || h === 0) return;

    canvas.width = w * 2;
    canvas.height = h * 2;
    ctx.scale(2, 2);

    var values = data.map(function (d) { return typeof d === 'number' ? d : (d.value || d.total || 0); });
    var max = Math.max.apply(null, values.concat([1]));
    var min = Math.min.apply(null, values.concat([0]));
    var range = max - min || 1;
    var step = w / (values.length - 1);

    ctx.beginPath();
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1.5;
    values.forEach(function (v, i) {
        var x = i * step;
        var y = h - ((v - min) / range) * (h - 4) - 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill gradient below line
    ctx.lineTo((values.length - 1) * step, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    var grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(99,102,241,0.2)');
    grad.addColorStop(1, 'rgba(99,102,241,0)');
    ctx.fillStyle = grad;
    ctx.fill();
};

// ---------------------------------------------------------------------------
//  Global setFilter function (used by inline onclick in health table)
// ---------------------------------------------------------------------------

function setFilter(siteId) {
    if (typeof app !== 'undefined') {
        app._setFilter(siteId);
    }
}

// ---------------------------------------------------------------------------
//  Singleton and initialization
// ---------------------------------------------------------------------------

var app = new EmpireApp();

document.addEventListener('DOMContentLoaded', function () {
    app.init();
});
