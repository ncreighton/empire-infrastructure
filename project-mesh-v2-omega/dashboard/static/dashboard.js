const API = '';
let allProjects = [];

async function fetchJSON(url) {
    try {
        const resp = await fetch(API + url);
        return await resp.json();
    } catch (e) {
        console.error('Fetch error:', url, e);
        return null;
    }
}

async function loadStatus() {
    const data = await fetchJSON('/api/status');
    if (!data) return;

    const badge = document.getElementById('status-badge');
    const status = data.daemon?.state || 'unknown';
    badge.textContent = status === 'idle' ? 'Daemon Running' : status;
    badge.className = data.status === 'healthy' ? 'healthy' : 'degraded';
}

async function loadServices() {
    const data = await fetchJSON('/api/services');
    if (!data || !data.services) return;

    const grid = document.getElementById('services-grid');
    grid.innerHTML = '';

    for (const [id, svc] of Object.entries(data.services)) {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="name">${svc.name || id}</div>
            <div class="meta">:${svc.port || '?'}</div>
            <div class="status ${svc.status || 'unknown'}">${svc.status || 'unknown'}
                ${svc.response_time_ms ? ` (${svc.response_time_ms}ms)` : ''}
            </div>
        `;
        grid.appendChild(card);
    }
}

async function loadGraphStats() {
    const data = await fetchJSON('/api/knowledge');
    if (!data || !data.stats) return;

    const grid = document.getElementById('graph-stats');
    grid.innerHTML = '';

    const labels = {
        projects: 'Projects', functions: 'Functions', classes: 'Classes',
        api_endpoints: 'Endpoints', knowledge_entries: 'Knowledge', patterns: 'Patterns',
        dependencies: 'Dependencies', code_snippets: 'Snippets',
        configs: 'Configs', api_keys_used: 'API Keys'
    };

    for (const [key, count] of Object.entries(data.stats)) {
        const stat = document.createElement('div');
        stat.className = 'stat';
        stat.innerHTML = `
            <div class="value">${count.toLocaleString()}</div>
            <div class="label">${labels[key] || key}</div>
        `;
        grid.appendChild(stat);
    }
}

async function loadProjects() {
    const data = await fetchJSON('/api/projects');
    if (!data) return;

    allProjects = data.projects || [];
    document.getElementById('project-count').textContent = `(${allProjects.length})`;
    renderProjects(allProjects);
}

function renderProjects(projects) {
    const grid = document.getElementById('projects-grid');
    grid.innerHTML = '';

    for (const p of projects) {
        const card = document.createElement('div');
        card.className = `card priority-${p.priority}`;
        card.setAttribute('data-category', p.category);
        card.setAttribute('data-type', p.project_type);

        const portBadge = p.port ? `<span class="meta">:${p.port}</span>` : '';
        card.innerHTML = `
            <div class="name">${p.name}</div>
            <div class="meta">${p.category} ${portBadge}</div>
        `;
        grid.appendChild(card);
    }
}

function filterProjects(filter, btn) {
    document.querySelectorAll('.filter').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (filter === 'all') {
        renderProjects(allProjects);
    } else if (filter === 'wordpress') {
        renderProjects(allProjects.filter(p => !p.project_type || p.project_type === 'wordpress'));
    } else {
        renderProjects(allProjects.filter(p => p.category === filter));
    }
}

async function loadEvents() {
    const data = await fetchJSON('/api/events?count=20');
    if (!data || !data.events) return;

    const feed = document.getElementById('events-feed');
    feed.innerHTML = '';

    for (const ev of data.events.reverse().slice(0, 15)) {
        const div = document.createElement('div');
        div.className = 'event';
        const time = ev.timestamp ? new Date(ev.timestamp).toLocaleString() : '';
        div.innerHTML = `
            <span class="time">${time}</span>
            <span class="type-badge">${ev.type}</span>
            <span>${ev.source || ''}</span>
        `;
        feed.appendChild(div);
    }

    if (data.events.length === 0) {
        feed.innerHTML = '<div class="event"><span class="meta">No events yet. Start the daemon to generate events.</span></div>';
    }
}

async function doSearch() {
    const input = document.getElementById('search-input');
    const q = input.value.trim();
    if (!q) return;

    const data = await fetchJSON(`/api/search?q=${encodeURIComponent(q)}`);
    const container = document.getElementById('search-results');

    if (!data || !data.results || data.results.length === 0) {
        container.innerHTML = '<div class="result">No results found.</div>';
        return;
    }

    container.innerHTML = data.results.map(r => `
        <div class="result">
            <span class="type">${r.type}</span>
            <strong>${r.name}</strong>
            ${r.project ? ` <span class="meta">[${r.project}]</span>` : ''}
            ${r.detail ? `<br><span class="meta">${r.detail}</span>` : ''}
        </div>
    `).join('');
}

async function triggerAction(action) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Running...';

    try {
        const resp = await fetch(`/api/${action}`, { method: 'POST' });
        const data = await resp.json();
        btn.textContent = data.status === 'ok' ? 'Done!' : 'Error';
        setTimeout(() => { btn.disabled = false; btn.textContent = btn.dataset.original || action; }, 2000);
        refreshAll();
    } catch (e) {
        btn.textContent = 'Error';
        setTimeout(() => { btn.disabled = false; }, 2000);
    }
}

async function refreshAll() {
    await Promise.all([
        loadStatus(),
        loadServices(),
        loadGraphStats(),
        loadProjects(),
        loadEvents(),
    ]);
    document.getElementById('last-refresh').textContent = `Last refresh: ${new Date().toLocaleTimeString()}`;
}

// Initial load
refreshAll();

// Auto-refresh every 30s
setInterval(refreshAll, 30000);
