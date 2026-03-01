const API = '';
let allProjects = [];
let projectDnaCache = {};

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

// -- Knowledge Base -------------------------------------------

async function loadKnowledge() {
    const data = await fetchJSON('/api/knowledge/entries?limit=50');
    if (!data) return;

    // Render category counts
    const catContainer = document.getElementById('knowledge-categories');
    if (data.categories && data.categories.length > 0) {
        catContainer.innerHTML = '<div class="kb-cat-header">Categories (' + data.total + ' total entries)</div>' +
            data.categories.map(c => `
                <div class="kb-cat-pill" onclick="filterKnowledge('${escapeHtml(c.category)}')">
                    <span class="kb-cat-name">${escapeHtml(c.category)}</span>
                    <span class="kb-cat-count">${c.count}</span>
                </div>
            `).join('');
    } else {
        catContainer.innerHTML = '<div class="meta">No knowledge entries indexed yet. Run a scan first.</div>';
    }

    // Render recent entries
    const entryContainer = document.getElementById('knowledge-entries');
    if (data.entries && data.entries.length > 0) {
        entryContainer.innerHTML = data.entries.slice(0, 20).map(e => `
            <div class="kb-entry">
                <span class="kb-entry-cat">${escapeHtml(e.category || 'general')}</span>
                <span class="kb-entry-text">${escapeHtml(e.text)}</span>
                ${e.source_project ? `<span class="kb-entry-project">[${escapeHtml(e.source_project)}]</span>` : ''}
            </div>
        `).join('');
    } else {
        entryContainer.innerHTML = '';
    }
}

function filterKnowledge(category) {
    // Highlight selected category and scroll to entries
    const pills = document.querySelectorAll('.kb-cat-pill');
    pills.forEach(p => p.classList.remove('active'));
    const clicked = [...pills].find(p => p.querySelector('.kb-cat-name').textContent === category);
    if (clicked) clicked.classList.add('active');

    // Fetch filtered entries via search
    fetchJSON(`/api/search?q=${encodeURIComponent(category)}`).then(data => {
        const entryContainer = document.getElementById('knowledge-entries');
        if (!data || !data.results || data.results.length === 0) {
            entryContainer.innerHTML = '<div class="meta">No entries in this category.</div>';
            return;
        }
        entryContainer.innerHTML = data.results.map(r => `
            <div class="kb-entry">
                <span class="kb-entry-cat">${escapeHtml(r.type)}</span>
                <span class="kb-entry-text">${escapeHtml(r.name)}</span>
                ${r.project ? `<span class="kb-entry-project">[${escapeHtml(r.project)}]</span>` : ''}
                ${r.detail ? `<br><span class="meta" style="margin-left:10px">${escapeHtml(r.detail)}</span>` : ''}
            </div>
        `).join('');
    });
}

// -- Detected Patterns ----------------------------------------

async function loadPatterns() {
    const data = await fetchJSON('/api/patterns');
    if (!data) return;

    const container = document.getElementById('patterns-section');
    if (!data.patterns || data.patterns.length === 0) {
        container.innerHTML = '<div class="meta">No patterns detected yet. Run a code scan to discover patterns.</div>';
        return;
    }

    container.innerHTML = data.patterns.map(p => {
        const projects = Array.isArray(p.used_by_projects) ? p.used_by_projects : [];
        const files = Array.isArray(p.implementation_files) ? p.implementation_files : [];
        return `
            <div class="pattern-card">
                <div class="pattern-header">
                    <span class="pattern-name">${escapeHtml(p.name)}</span>
                    <span class="pattern-type-badge">${escapeHtml(p.pattern_type || 'code')}</span>
                </div>
                ${p.description ? `<div class="pattern-desc">${escapeHtml(p.description)}</div>` : ''}
                ${projects.length > 0 ? `
                    <div class="pattern-projects">
                        Used by: ${projects.map(pr => `<span class="cap-badge cap-automation">${escapeHtml(pr)}</span>`).join(' ')}
                    </div>
                ` : ''}
                ${files.length > 0 ? `
                    <div class="pattern-files meta">
                        Files: ${files.slice(0, 3).map(f => escapeHtml(f)).join(', ')}${files.length > 3 ? ` +${files.length - 3} more` : ''}
                    </div>
                ` : ''}
                ${p.canonical_source ? `<div class="meta">Canonical: ${escapeHtml(p.canonical_source)}</div>` : ''}
            </div>
        `;
    }).join('');
}

// -- Dependencies ---------------------------------------------

async function loadDependencies() {
    const data = await fetchJSON('/api/dependencies');
    if (!data) return;

    // Render summary grid (most connected projects)
    const summaryContainer = document.getElementById('dep-summary');
    if (data.project_summary && Object.keys(data.project_summary).length > 0) {
        const sorted = Object.entries(data.project_summary)
            .map(([slug, counts]) => ({ slug, ...counts, total: counts.depends_on + counts.depended_by }))
            .sort((a, b) => b.total - a.total)
            .slice(0, 10);

        summaryContainer.innerHTML = sorted.map(p => `
            <div class="dep-summary-card">
                <div class="dep-proj-name">${escapeHtml(p.slug)}</div>
                <div class="dep-counts">
                    <span class="dep-out" title="Depends on">${p.depends_on} out</span>
                    <span class="dep-in" title="Depended by">${p.depended_by} in</span>
                </div>
            </div>
        `).join('');
    } else {
        summaryContainer.innerHTML = '<div class="meta">No dependencies mapped yet.</div>';
    }

    // Render full dependency list
    const listContainer = document.getElementById('dep-list');
    if (data.dependencies && data.dependencies.length > 0) {
        listContainer.innerHTML = '<div class="dep-list-header">' + data.total + ' dependency links</div>' +
            data.dependencies.slice(0, 30).map(d => `
                <div class="dep-row">
                    <span class="dep-from">${escapeHtml(d.from_project)}</span>
                    <span class="dep-arrow">&rarr;</span>
                    <span class="dep-to">${escapeHtml(d.to_project)}</span>
                    <span class="dep-type-badge">${escapeHtml(d.dependency_type)}</span>
                    ${d.details ? `<span class="meta">${escapeHtml(d.details)}</span>` : ''}
                </div>
            `).join('') +
            (data.dependencies.length > 30 ? `<div class="meta">...and ${data.dependencies.length - 30} more</div>` : '');
    } else {
        listContainer.innerHTML = '';
    }
}

// -- Top APIs -------------------------------------------------

async function loadApiStats() {
    const data = await fetchJSON('/api/stats/summary');
    if (!data) return;

    // API usage grid
    const apiGrid = document.getElementById('api-usage-grid');
    if (data.api_usage && data.api_usage.length > 0) {
        apiGrid.innerHTML = data.api_usage.map(a => `
            <div class="stat">
                <div class="value">${a.project_count}</div>
                <div class="label">${escapeHtml(a.service_name)}</div>
                <div class="meta">${a.total_usages} usage${a.total_usages !== 1 ? 's' : ''}</div>
            </div>
        `).join('');
    } else {
        apiGrid.innerHTML = '<div class="meta">No API usage data yet.</div>';
    }

    // API per project list
    const projList = document.getElementById('api-per-project');
    if (data.api_per_project && data.api_per_project.length > 0) {
        projList.innerHTML = '<div class="api-proj-header">Projects by API Integration Count</div>' +
            data.api_per_project.map(p => {
                const services = (p.services || '').split(',').filter(Boolean);
                return `
                    <div class="api-proj-row">
                        <span class="api-proj-name">${escapeHtml(p.name || p.slug)}</span>
                        <span class="api-proj-count">${p.api_count} APIs</span>
                        <div class="api-proj-services">
                            ${services.map(s => `<span class="cap-badge cap-api">${escapeHtml(s.trim())}</span>`).join(' ')}
                        </div>
                    </div>
                `;
            }).join('');
    } else {
        projList.innerHTML = '';
    }
}

// -- Projects (enhanced with DNA badges) ----------------------

async function loadProjects() {
    const data = await fetchJSON('/api/projects');
    if (!data) return;

    allProjects = data.projects || [];
    document.getElementById('project-count').textContent = `(${allProjects.length})`;

    // Pre-fetch DNA profiles for all projects (in background, non-blocking)
    for (const p of allProjects) {
        if (!projectDnaCache[p.slug]) {
            fetchJSON(`/api/dna/${p.slug}`).then(dna => {
                if (dna && !dna.error) {
                    projectDnaCache[p.slug] = dna;
                    // Re-render the specific card if it exists
                    const card = document.querySelector(`[data-slug="${p.slug}"]`);
                    if (card) {
                        updateCardDnaBadges(card, dna);
                    }
                }
            });
        }
    }

    renderProjects(allProjects);
}

function renderProjects(projects) {
    const grid = document.getElementById('projects-grid');
    grid.innerHTML = '';

    for (const p of projects) {
        const card = document.createElement('div');
        card.className = `card project-card priority-${p.priority}`;
        card.setAttribute('data-category', p.category);
        card.setAttribute('data-type', p.project_type);
        card.setAttribute('data-slug', p.slug);

        const portBadge = p.port ? `<span class="meta">:${p.port}</span>` : '';
        card.innerHTML = `
            <div class="name">${escapeHtml(p.name)}</div>
            <div class="meta">${escapeHtml(p.category)} ${portBadge}</div>
            <div class="dna-badges"></div>
            <div class="dna-detail" style="display:none"></div>
        `;

        // Add click-to-expand for full DNA profile
        card.addEventListener('click', () => toggleDnaDetail(card, p.slug));

        // Apply cached DNA badges if available
        if (projectDnaCache[p.slug]) {
            updateCardDnaBadges(card, projectDnaCache[p.slug]);
        }

        grid.appendChild(card);
    }
}

function updateCardDnaBadges(card, dna) {
    const badgeContainer = card.querySelector('.dna-badges');
    if (!badgeContainer || !dna.capabilities) return;

    // Show up to 5 capability badges
    const caps = dna.capabilities.slice(0, 5);
    badgeContainer.innerHTML = caps.map(c => {
        const catClass = getCapabilityColorClass(c);
        return `<span class="cap-badge ${catClass}">${escapeHtml(c)}</span>`;
    }).join('') + (dna.capabilities.length > 5 ? `<span class="cap-badge cap-more">+${dna.capabilities.length - 5}</span>` : '');
}

function getCapabilityColorClass(cap) {
    const colorMap = {
        'seo-optimization': 'cap-seo',
        'wordpress-api': 'cap-wordpress',
        'image-generation': 'cap-image',
        'video-creation': 'cap-video',
        'tts-generation': 'cap-audio',
        'content-pipeline': 'cap-content',
        'email-marketing': 'cap-email',
        'social-media': 'cap-social',
        'ecommerce': 'cap-ecommerce',
        'analytics': 'cap-analytics',
        'automation': 'cap-automation',
        'ai-llm': 'cap-ai',
        'database': 'cap-database',
        'api-service': 'cap-api',
        'web-scraping': 'cap-scraping',
        'affiliate': 'cap-affiliate',
    };
    return colorMap[cap] || 'cap-default';
}

async function toggleDnaDetail(card, slug) {
    const detail = card.querySelector('.dna-detail');
    if (!detail) return;

    // Toggle visibility
    if (detail.style.display !== 'none') {
        detail.style.display = 'none';
        return;
    }

    // Fetch DNA if not cached
    if (!projectDnaCache[slug]) {
        detail.innerHTML = '<div class="meta">Loading DNA profile...</div>';
        detail.style.display = 'block';
        const dna = await fetchJSON(`/api/dna/${slug}`);
        if (dna && !dna.error) {
            projectDnaCache[slug] = dna;
            updateCardDnaBadges(card, dna);
        } else {
            detail.innerHTML = '<div class="meta">No DNA profile available.</div>';
            return;
        }
    }

    const dna = projectDnaCache[slug];
    detail.style.display = 'block';
    detail.innerHTML = `
        <div class="dna-section">
            <div class="dna-score">Reuse Score: <strong>${dna.code_reuse_score || 0}/100</strong></div>
            <div class="dna-row">
                <span class="dna-label">Capabilities:</span>
                <div>${(dna.capabilities || []).map(c => `<span class="cap-badge ${getCapabilityColorClass(c)}">${escapeHtml(c)}</span>`).join(' ') || 'None'}</div>
            </div>
            <div class="dna-row">
                <span class="dna-label">Patterns:</span>
                <div>${(dna.patterns || []).map(p => `<span class="cap-badge cap-pattern">${escapeHtml(p)}</span>`).join(' ') || 'None'}</div>
            </div>
            <div class="dna-row">
                <span class="dna-label">APIs:</span>
                <div>${(dna.apis_integrated || []).map(a => `<span class="cap-badge cap-api">${escapeHtml(a)}</span>`).join(' ') || 'None'}</div>
            </div>
            <div class="dna-row">
                <span class="dna-label">Tech Stack:</span>
                <div>${(dna.tech_stack || []).map(t => `<span class="cap-badge cap-default">${escapeHtml(t)}</span>`).join(' ') || 'N/A'}</div>
            </div>
            <div class="dna-row">
                <span class="dna-label">Stats:</span>
                <span class="meta">
                    ${dna.stats ? `${dna.stats.functions} fn | ${dna.stats.classes} cls | ${dna.stats.endpoints} ep | ${dna.stats.configs} cfg` : 'N/A'}
                </span>
            </div>
            <button class="dna-similar-btn" onclick="event.stopPropagation(); loadSimilar('${escapeHtml(slug)}', this)">Find Similar Projects</button>
            <div class="dna-similar-results"></div>
        </div>
    `;
}

async function loadSimilar(slug, btn) {
    const resultsDiv = btn.nextElementSibling;
    btn.disabled = true;
    btn.textContent = 'Loading...';

    const data = await fetchJSON(`/api/dna/similar/${encodeURIComponent(slug)}`);
    btn.textContent = 'Find Similar Projects';
    btn.disabled = false;

    if (!data || !data.similar || data.similar.length === 0) {
        resultsDiv.innerHTML = '<div class="meta">No similar projects found.</div>';
        return;
    }

    resultsDiv.innerHTML = data.similar.map(s => `
        <div class="similar-row">
            <span class="similar-name">${escapeHtml(s.project)}</span>
            <span class="similar-score">${s.similarity}%</span>
            <div class="similar-shared">
                ${(s.shared_capabilities || []).map(c => `<span class="cap-badge cap-default">${escapeHtml(c)}</span>`).join(' ')}
            </div>
        </div>
    `).join('');
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

    const container = document.getElementById('search-results');
    container.innerHTML = '<div class="result search-loading">Searching...</div>';

    const data = await fetchJSON(`/api/search?q=${encodeURIComponent(q)}`);

    if (!data || !data.results || data.results.length === 0) {
        container.innerHTML = '<div class="result search-empty">No results found.</div>';
        return;
    }

    container.innerHTML = `<div class="search-count">${data.results.length} result${data.results.length !== 1 ? 's' : ''} for "${escapeHtml(q)}"</div>` +
        data.results.map(r => `
            <div class="result result-${r.type || 'unknown'}">
                <span class="type">${r.type}</span>
                <strong>${escapeHtml(r.name)}</strong>
                ${r.project ? ` <span class="result-project">[${escapeHtml(r.project)}]</span>` : ''}
                ${r.score ? `<span class="result-score">${r.score.toFixed(1)}</span>` : ''}
                ${r.detail ? `<br><span class="meta result-detail">${escapeHtml(r.detail)}</span>` : ''}
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

// -- Utility --------------------------------------------------

function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// -- Refresh All ----------------------------------------------

async function refreshAll() {
    await Promise.all([
        loadStatus(),
        loadServices(),
        loadGraphStats(),
        loadProjects(),
        loadEvents(),
        loadKnowledge(),
        loadPatterns(),
        loadDependencies(),
        loadApiStats(),
    ]);
    document.getElementById('last-refresh').textContent = `Last refresh: ${new Date().toLocaleTimeString()}`;
}

// Initial load
refreshAll();

// Auto-refresh every 30s
setInterval(refreshAll, 30000);
