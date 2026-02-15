// =============================================================================
//  Chart Factory — OpenClaw Empire Dashboard v2
//  Chart.js factory for revenue, growth, velocity, and forecast charts.
//  Requires Chart.js loaded globally from CDN.
// =============================================================================

'use strict';

// ---------------------------------------------------------------------------
//  Site color palette — 16 distinct brand colors
// ---------------------------------------------------------------------------

var SITE_COLORS = {
    witchcraft:         '#4A1C6F',
    smarthome:          '#0066CC',
    aiaction:           '#00F0FF',
    aidiscovery:        '#E94560',
    wealthai:           '#00C853',
    family:             '#E8887C',
    mythical:           '#8B4513',
    bulletjournals:     '#FF6B6B',
    crystalwitchcraft:  '#9B59B6',
    herbalwitchery:     '#2ECC71',
    moonphasewitch:     '#C0C0C0',
    tarotbeginners:     '#FFD700',
    spellsrituals:      '#8B0000',
    paganpathways:      '#556B2F',
    witchyhomedecor:    '#DDA0DD',
    seasonalwitchcraft: '#FF8C00'
};

var SITE_COLORS_ARRAY = [
    '#4A1C6F', '#0066CC', '#00F0FF', '#E94560', '#00C853', '#E8887C',
    '#8B4513', '#FF6B6B', '#9B59B6', '#2ECC71', '#C0C0C0', '#FFD700',
    '#8B0000', '#556B2F', '#DDA0DD', '#FF8C00'
];

var SITE_NAMES = [
    'Witchcraft', 'Smart Home', 'AI Action', 'AI Discovery', 'Wealth AI',
    'Family Flourish', 'Mythical', 'Bullet Journals', 'Crystal', 'Herbal',
    'Moon Phase', 'Tarot', 'Spells', 'Pagan', 'Witchy Decor', 'Seasonal'
];

// Revenue gradient palette
var REVENUE_GRADIENT_COLORS = ['#22c55e', '#16a34a', '#15803d'];

// Pipeline stage gradient (blue -> green)
var STAGE_GRADIENT = ['#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#22c55e'];

// Engagement doughnut palette
var ENGAGEMENT_COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#94a3b8'];

// ---------------------------------------------------------------------------
//  ChartFactory
// ---------------------------------------------------------------------------

function ChartFactory() {
    this._instances = {};
    this._defaultFont = "'Inter', 'system-ui', sans-serif";
    this._monoFont = "'JetBrains Mono', monospace";
}

// ---- Public: Chart creation -----------------------------------------------

/**
 * Create a stacked bar chart of revenue broken down by site.
 *
 * data: {
 *   labels: string[],                  // date labels
 *   datasets: { site_id: string, values: number[] }[]
 * }
 * OR simplified:
 * data: {
 *   labels: string[],
 *   values: number[]                   // single series
 * }
 */
ChartFactory.prototype.createRevenueChart = function createRevenueChart(canvasId, data) {
    this.destroyChart(canvasId);
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('[Charts] Canvas #' + canvasId + ' not found');
        return null;
    }
    var ctx = canvas.getContext('2d');

    var labels = data.labels || [];
    var datasets;

    if (data.datasets && Array.isArray(data.datasets) && data.datasets.length > 0 && data.datasets[0].site_id) {
        // Multi-site stacked data
        datasets = data.datasets.map(function (ds, idx) {
            var color = SITE_COLORS[ds.site_id] || SITE_COLORS_ARRAY[idx % SITE_COLORS_ARRAY.length];
            return {
                label: ds.name || ds.site_id,
                data: ds.values || ds.data || [],
                backgroundColor: color + 'cc',
                borderColor: color,
                borderWidth: 1,
                borderRadius: 2
            };
        });
    } else if (data.values || (data.datasets && data.datasets.length > 0)) {
        // Single series with gradient
        var gradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 220);
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0.6)');
        gradient.addColorStop(1, 'rgba(34, 197, 94, 0.05)');

        var values = data.values || (data.datasets[0].data || data.datasets[0].values || []);
        datasets = [{
            label: 'Revenue',
            data: values,
            backgroundColor: gradient,
            borderColor: '#22c55e',
            borderWidth: 2,
            borderRadius: 4
        }];
    } else {
        datasets = [];
    }

    var options = this._mergeOptions({
        plugins: {
            legend: {
                display: datasets.length > 1,
                position: 'bottom',
                labels: {
                    color: '#e2e8f0',
                    font: { size: 10, family: this._defaultFont },
                    boxWidth: 10,
                    padding: 12,
                    usePointStyle: true
                }
            },
            tooltip: this._tooltipConfig(function (ctx2) {
                return '$' + Number(ctx2.parsed.y || 0).toFixed(2);
            })
        },
        scales: {
            x: this._xAxisConfig(),
            y: this._yAxisConfig('$')
        }
    });

    if (datasets.length > 1) {
        options.scales.x.stacked = true;
        options.scales.y.stacked = true;
    }

    var chart = new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: datasets },
        options: options
    });

    this._instances[canvasId] = chart;
    return chart;
};

/**
 * Create a bar chart of articles published per week per site.
 *
 * data: {
 *   labels: string[],   // site names
 *   values: number[],   // articles count
 *   colors?: string[]   // optional per-bar colors
 * }
 */
ChartFactory.prototype.createVelocityChart = function createVelocityChart(canvasId, data) {
    this.destroyChart(canvasId);
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('[Charts] Canvas #' + canvasId + ' not found');
        return null;
    }
    var ctx = canvas.getContext('2d');

    var labels = data.labels || [];
    var values = data.values || data.data || [];
    var colors = data.colors || labels.map(function (_, idx) {
        return SITE_COLORS_ARRAY[idx % SITE_COLORS_ARRAY.length];
    });

    var chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Articles / Week',
                data: values,
                backgroundColor: colors.map(function (c) { return c + 'cc'; }),
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 4,
                barThickness: 'flex',
                maxBarThickness: 32
            }]
        },
        options: this._mergeOptions({
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: this._tooltipConfig(function (ctx2) {
                    return ctx2.parsed.x + ' articles';
                })
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        stepSize: 1
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#e2e8f0',
                        font: { size: 11, family: "'Inter', sans-serif" }
                    }
                }
            }
        })
    });

    this._instances[canvasId] = chart;
    return chart;
};

/**
 * Create a line chart with confidence bands (fill area) for forecasting.
 *
 * data: {
 *   labels: string[],
 *   actual: number[],          // actual values (may have nulls for future dates)
 *   forecast: number[],        // predicted values
 *   upper_bound: number[],     // upper confidence band
 *   lower_bound: number[]      // lower confidence band
 * }
 */
ChartFactory.prototype.createForecastChart = function createForecastChart(canvasId, data) {
    this.destroyChart(canvasId);
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('[Charts] Canvas #' + canvasId + ' not found');
        return null;
    }
    var ctx = canvas.getContext('2d');

    var labels = data.labels || [];
    var actual = data.actual || [];
    var forecast = data.forecast || [];
    var upperBound = data.upper_bound || data.upperBound || [];
    var lowerBound = data.lower_bound || data.lowerBound || [];

    // Gradient for confidence band
    var bandGradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 220);
    bandGradient.addColorStop(0, 'rgba(99, 102, 241, 0.15)');
    bandGradient.addColorStop(1, 'rgba(99, 102, 241, 0.02)');

    // Actual line gradient
    var actualGradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 220);
    actualGradient.addColorStop(0, 'rgba(34, 197, 94, 0.25)');
    actualGradient.addColorStop(1, 'rgba(34, 197, 94, 0)');

    var datasets = [
        {
            label: 'Upper Bound',
            data: upperBound,
            borderColor: 'rgba(99, 102, 241, 0.3)',
            backgroundColor: bandGradient,
            fill: '+1',
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.3,
            order: 3
        },
        {
            label: 'Lower Bound',
            data: lowerBound,
            borderColor: 'rgba(99, 102, 241, 0.3)',
            backgroundColor: 'transparent',
            fill: false,
            borderWidth: 1,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.3,
            order: 4
        },
        {
            label: 'Forecast',
            data: forecast,
            borderColor: '#6366f1',
            backgroundColor: 'transparent',
            fill: false,
            borderWidth: 2,
            borderDash: [6, 3],
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: '#6366f1',
            tension: 0.3,
            order: 2
        },
        {
            label: 'Actual',
            data: actual,
            borderColor: '#22c55e',
            backgroundColor: actualGradient,
            fill: true,
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: '#22c55e',
            tension: 0.3,
            order: 1
        }
    ];

    var chart = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: this._mergeOptions({
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 10, family: this._defaultFont },
                        boxWidth: 12,
                        padding: 12,
                        usePointStyle: true,
                        filter: function (item) {
                            return item.text !== 'Upper Bound' && item.text !== 'Lower Bound';
                        }
                    }
                },
                tooltip: this._tooltipConfig(function (ctx2) {
                    return '$' + Number(ctx2.parsed.y || 0).toFixed(2);
                })
            },
            scales: {
                x: this._xAxisConfig(),
                y: this._yAxisConfig('$')
            }
        })
    });

    this._instances[canvasId] = chart;
    return chart;
};

/**
 * Create a line chart of subscriber / traffic growth over time.
 *
 * data: {
 *   labels: string[],
 *   series: { name: string, values: number[], color?: string }[]
 * }
 * OR simplified:
 * data: {
 *   labels: string[],
 *   values: number[]
 * }
 */
ChartFactory.prototype.createGrowthChart = function createGrowthChart(canvasId, data) {
    this.destroyChart(canvasId);
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('[Charts] Canvas #' + canvasId + ' not found');
        return null;
    }
    var ctx = canvas.getContext('2d');

    var labels = data.labels || [];
    var datasets;

    if (data.series && data.series.length > 0) {
        datasets = data.series.map(function (s, idx) {
            var color = s.color || SITE_COLORS_ARRAY[idx % SITE_COLORS_ARRAY.length];
            var gradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 220);
            gradient.addColorStop(0, color + '33');
            gradient.addColorStop(1, color + '00');

            return {
                label: s.name || 'Series ' + (idx + 1),
                data: s.values || s.data || [],
                borderColor: color,
                backgroundColor: idx === 0 ? gradient : 'transparent',
                fill: idx === 0,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: color,
                tension: 0.4
            };
        });
    } else {
        var gradient = ctx.createLinearGradient(0, 0, 0, canvas.height || 220);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.25)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');

        datasets = [{
            label: 'Growth',
            data: data.values || [],
            borderColor: '#6366f1',
            backgroundColor: gradient,
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: '#6366f1',
            tension: 0.4
        }];
    }

    var chart = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: this._mergeOptions({
            plugins: {
                legend: {
                    display: datasets.length > 1,
                    position: 'bottom',
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 10, family: this._defaultFont },
                        boxWidth: 10,
                        padding: 12,
                        usePointStyle: true
                    }
                },
                tooltip: this._tooltipConfig(function (ctx2) {
                    return Number(ctx2.parsed.y || 0).toLocaleString();
                })
            },
            scales: {
                x: this._xAxisConfig(),
                y: this._yAxisConfig('')
            }
        })
    });

    this._instances[canvasId] = chart;
    return chart;
};

/**
 * Create a horizontal bar chart of pipeline stage durations.
 *
 * data: {
 *   stages: string[],      // stage names
 *   durations: number[],   // seconds or ms
 *   unit?: string           // 's' or 'ms' (default 's')
 * }
 */
ChartFactory.prototype.createPipelineChart = function createPipelineChart(canvasId, data) {
    this.destroyChart(canvasId);
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('[Charts] Canvas #' + canvasId + ' not found');
        return null;
    }
    var ctx = canvas.getContext('2d');

    var stages = data.stages || data.labels || [];
    var durations = data.durations || data.values || data.data || [];
    var unit = data.unit || 's';

    // Generate a gradient palette from blue to green
    var stageColors = stages.map(function (_, idx) {
        var t = stages.length > 1 ? idx / (stages.length - 1) : 0;
        var r = Math.round(59 + (34 - 59) * t);
        var g = Math.round(130 + (197 - 130) * t);
        var b = Math.round(246 + (94 - 246) * t);
        return 'rgb(' + r + ',' + g + ',' + b + ')';
    });

    var chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: stages,
            datasets: [{
                label: 'Duration (' + unit + ')',
                data: durations,
                backgroundColor: stageColors.map(function (c) { return c.replace('rgb', 'rgba').replace(')', ',0.7)'); }),
                borderColor: stageColors,
                borderWidth: 1,
                borderRadius: 4,
                barThickness: 'flex',
                maxBarThickness: 28
            }]
        },
        options: this._mergeOptions({
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: this._tooltipConfig(function (ctx2) {
                    var val = ctx2.parsed.x;
                    if (unit === 'ms') {
                        return val < 1000 ? val + 'ms' : (val / 1000).toFixed(1) + 's';
                    }
                    return val < 60 ? val + 's' : Math.floor(val / 60) + 'm ' + (val % 60) + 's';
                })
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    ticks: {
                        color: '#94a3b8',
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                        callback: function (v) {
                            if (unit === 'ms') return v < 1000 ? v + 'ms' : (v / 1000).toFixed(0) + 's';
                            return v + 's';
                        }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#e2e8f0',
                        font: { size: 11, family: "'Inter', sans-serif" }
                    }
                }
            }
        })
    });

    this._instances[canvasId] = chart;
    return chart;
};

/**
 * Create a doughnut chart of engagement levels.
 *
 * data: {
 *   labels: string[],   // e.g. ['High', 'Medium', 'Low', 'Bounce', 'Other']
 *   values: number[],
 *   colors?: string[]
 * }
 */
ChartFactory.prototype.createEngagementChart = function createEngagementChart(canvasId, data) {
    this.destroyChart(canvasId);
    var canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error('[Charts] Canvas #' + canvasId + ' not found');
        return null;
    }
    var ctx = canvas.getContext('2d');

    var labels = data.labels || [];
    var values = data.values || data.data || [];
    var colors = data.colors || ENGAGEMENT_COLORS.slice(0, labels.length);

    var total = values.reduce(function (sum, v) { return sum + v; }, 0);

    var chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(function (c) { return c + 'cc'; }),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 8,
                spacing: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    display: true,
                    position: 'right',
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 11, family: "'Inter', sans-serif" },
                        boxWidth: 12,
                        padding: 10,
                        usePointStyle: true,
                        generateLabels: function (chart2) {
                            var dataset = chart2.data.datasets[0];
                            return chart2.data.labels.map(function (label, i) {
                                var pct = total > 0 ? ((dataset.data[i] / total) * 100).toFixed(1) : '0.0';
                                return {
                                    text: label + ' (' + pct + '%)',
                                    fillStyle: dataset.backgroundColor[i],
                                    strokeStyle: dataset.borderColor[i],
                                    lineWidth: 2,
                                    hidden: false,
                                    index: i,
                                    pointStyle: 'circle'
                                };
                            });
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(18, 18, 26, 0.95)',
                    borderColor: '#1e1e2e',
                    borderWidth: 1,
                    titleColor: '#e2e8f0',
                    bodyColor: '#94a3b8',
                    bodyFont: { family: "'JetBrains Mono', monospace" },
                    padding: 10,
                    cornerRadius: 8,
                    callbacks: {
                        label: function (ctx2) {
                            var pct = total > 0 ? ((ctx2.parsed / total) * 100).toFixed(1) : '0.0';
                            return ' ' + ctx2.label + ': ' + ctx2.parsed.toLocaleString() + ' (' + pct + '%)';
                        }
                    }
                }
            }
        }
    });

    this._instances[canvasId] = chart;
    return chart;
};

// ---- Public: Chart management ---------------------------------------------

/**
 * Destroy an existing chart instance to free memory.
 */
ChartFactory.prototype.destroyChart = function destroyChart(chartId) {
    if (this._instances[chartId]) {
        try {
            this._instances[chartId].destroy();
        } catch (err) {
            console.warn('[Charts] Error destroying chart "' + chartId + '":', err);
        }
        delete this._instances[chartId];
    }
};

/**
 * Update the data of an existing chart without recreating it.
 */
ChartFactory.prototype.updateChart = function updateChart(chartId, newData) {
    var chart = this._instances[chartId];
    if (!chart) {
        console.warn('[Charts] No chart instance "' + chartId + '" to update');
        return;
    }

    try {
        if (newData.labels) {
            chart.data.labels = newData.labels;
        }

        if (newData.datasets) {
            for (var i = 0; i < newData.datasets.length && i < chart.data.datasets.length; i++) {
                if (newData.datasets[i].data) {
                    chart.data.datasets[i].data = newData.datasets[i].data;
                }
                if (newData.datasets[i].values) {
                    chart.data.datasets[i].data = newData.datasets[i].values;
                }
            }
        } else if (newData.values) {
            // Simple update of first dataset
            if (chart.data.datasets.length > 0) {
                chart.data.datasets[0].data = newData.values;
            }
        }

        chart.update('active');
    } catch (err) {
        console.error('[Charts] Error updating chart "' + chartId + '":', err);
    }
};

/**
 * Get the default dark theme options (shared across all charts).
 */
ChartFactory.prototype.getDefaultOptions = function getDefaultOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 800,
            easing: 'easeOutQuart'
        },
        interaction: {
            mode: 'index',
            intersect: false
        },
        elements: {
            line: {
                tension: 0.4
            },
            point: {
                radius: 0,
                hoverRadius: 5
            }
        },
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: 'rgba(18, 18, 26, 0.95)',
                borderColor: '#1e1e2e',
                borderWidth: 1,
                titleColor: '#e2e8f0',
                bodyColor: '#94a3b8',
                bodyFont: { family: "'JetBrains Mono', monospace" },
                padding: 10,
                cornerRadius: 8,
                displayColors: true,
                boxPadding: 4
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(255,255,255,0.06)', drawBorder: false },
                ticks: { color: '#94a3b8', font: { size: 10 }, maxTicksLimit: 12 }
            },
            y: {
                grid: { color: 'rgba(255,255,255,0.06)', drawBorder: false },
                ticks: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace", size: 10 } }
            }
        }
    };
};

/**
 * Get a chart instance by canvas ID.
 */
ChartFactory.prototype.getChart = function getChart(chartId) {
    return this._instances[chartId] || null;
};

/**
 * Destroy all chart instances.
 */
ChartFactory.prototype.destroyAll = function destroyAll() {
    var keys = Object.keys(this._instances);
    for (var i = 0; i < keys.length; i++) {
        this.destroyChart(keys[i]);
    }
};

// ---- Public: Utility helpers ----------------------------------------------

/**
 * Get a site's brand color by its ID.
 */
ChartFactory.prototype.getSiteColor = function getSiteColor(siteId) {
    return SITE_COLORS[siteId] || '#6366f1';
};

/**
 * Generate N evenly-spaced colors from a gradient.
 */
ChartFactory.prototype.generateGradientColors = function generateGradientColors(n, startColor, endColor) {
    startColor = startColor || '#3b82f6';
    endColor = endColor || '#22c55e';

    var start = this._hexToRgb(startColor);
    var end = this._hexToRgb(endColor);
    var colors = [];

    for (var i = 0; i < n; i++) {
        var t = n > 1 ? i / (n - 1) : 0;
        var r = Math.round(start.r + (end.r - start.r) * t);
        var g = Math.round(start.g + (end.g - start.g) * t);
        var b = Math.round(start.b + (end.b - start.b) * t);
        colors.push('rgb(' + r + ',' + g + ',' + b + ')');
    }

    return colors;
};

// ---- Internal: Shared chart configurations --------------------------------

ChartFactory.prototype._mergeOptions = function _mergeOptions(overrides) {
    var base = this.getDefaultOptions();
    return this._deepMerge(base, overrides);
};

ChartFactory.prototype._tooltipConfig = function _tooltipConfig(labelCallback) {
    return {
        backgroundColor: 'rgba(18, 18, 26, 0.95)',
        borderColor: '#1e1e2e',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        titleFont: { weight: '600', size: 12 },
        bodyColor: '#94a3b8',
        bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
        padding: { top: 8, bottom: 8, left: 12, right: 12 },
        cornerRadius: 8,
        displayColors: true,
        boxPadding: 4,
        callbacks: {
            label: labelCallback || function (ctx2) { return ctx2.formattedValue; }
        }
    };
};

ChartFactory.prototype._xAxisConfig = function _xAxisConfig() {
    return {
        grid: { color: 'rgba(255,255,255,0.06)', drawBorder: false },
        ticks: {
            color: '#94a3b8',
            font: { size: 10, family: this._defaultFont },
            maxTicksLimit: 12,
            maxRotation: 0
        }
    };
};

ChartFactory.prototype._yAxisConfig = function _yAxisConfig(prefix) {
    prefix = prefix || '';
    return {
        grid: { color: 'rgba(255,255,255,0.06)', drawBorder: false },
        ticks: {
            color: '#94a3b8',
            font: { family: "'JetBrains Mono', monospace", size: 10 },
            callback: function (value) {
                if (typeof value === 'number') {
                    if (value >= 1000000) return prefix + (value / 1000000).toFixed(1) + 'M';
                    if (value >= 1000) return prefix + (value / 1000).toFixed(1) + 'K';
                    return prefix + value;
                }
                return prefix + value;
            }
        },
        beginAtZero: true
    };
};

// ---- Internal: Utilities --------------------------------------------------

ChartFactory.prototype._hexToRgb = function _hexToRgb(hex) {
    hex = hex.replace(/^#/, '');
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    var num = parseInt(hex, 16);
    return {
        r: (num >> 16) & 255,
        g: (num >> 8) & 255,
        b: num & 255
    };
};

ChartFactory.prototype._deepMerge = function _deepMerge(target, source) {
    var result = {};
    var keys = Object.keys(target);
    for (var i = 0; i < keys.length; i++) {
        result[keys[i]] = target[keys[i]];
    }
    var sourceKeys = Object.keys(source);
    for (var j = 0; j < sourceKeys.length; j++) {
        var key = sourceKeys[j];
        if (
            source[key] !== null &&
            typeof source[key] === 'object' &&
            !Array.isArray(source[key]) &&
            result[key] !== null &&
            typeof result[key] === 'object' &&
            !Array.isArray(result[key])
        ) {
            result[key] = this._deepMerge(result[key], source[key]);
        } else {
            result[key] = source[key];
        }
    }
    return result;
};

// ---------------------------------------------------------------------------
//  Singleton
// ---------------------------------------------------------------------------

var chartFactory = new ChartFactory();
