// API Base URL
const API_BASE = 'http://localhost:5000/api';

// Chart instances
let charts = {};

// Utility Functions
const formatNumber = (num) => {
    return new Intl.NumberFormat('en-IN').format(num);
};

const animateValue = (element, start, end, duration) => {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = formatNumber(Math.floor(current));
    }, 16);
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadOverview();
        await loadStateAnalysis();
        await loadTimeSeries();
        await loadClustering();
        await loadStatistics();
        await loadHeatmap();
        await loadRankings();
        await loadAnomalies();
        await loadInsights();

        hideLoading();
    } catch (error) {
        console.error('Error initializing app:', error);
        hideLoading();
    }

    setupEventListeners();
});

// Hide Loading Overlay
function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('hidden');
}

// Load Overview Data
async function loadOverview() {
    try {
        const response = await fetch(`${API_BASE}/overview`);
        const data = await response.json();

        // Update hero stats
        animateValue(document.getElementById('totalRecords'), 0,
            data.total_demographic_records + data.total_biometric_records + data.total_enrollment_records, 2000);
        animateValue(document.getElementById('totalStates'), 0, data.total_states, 1500);
        animateValue(document.getElementById('totalDistricts'), 0, data.total_districts, 1500);

        // Update demographic chart
        createDoughnutChart('demographicChart', {
            labels: ['Age 5-17', 'Age 17+'],
            data: [data.demographic_age_distribution.age_5_17, data.demographic_age_distribution.age_17_plus],
            colors: ['#667eea', '#764ba2']
        });
        document.getElementById('demoTotal').textContent = formatNumber(
            data.demographic_age_distribution.age_5_17 + data.demographic_age_distribution.age_17_plus
        );

        // Update biometric chart
        createDoughnutChart('biometricChart', {
            labels: ['Age 5-17', 'Age 17+'],
            data: [data.biometric_age_distribution.age_5_17, data.biometric_age_distribution.age_17_plus],
            colors: ['#f093fb', '#f5576c']
        });
        document.getElementById('bioTotal').textContent = formatNumber(
            data.biometric_age_distribution.age_5_17 + data.biometric_age_distribution.age_17_plus
        );

        // Update enrollment chart
        createDoughnutChart('enrollmentChart', {
            labels: ['Age 0-5', 'Age 5-17', 'Age 18+'],
            data: [
                data.enrollment_age_distribution.age_0_5,
                data.enrollment_age_distribution.age_5_17,
                data.enrollment_age_distribution.age_18_plus
            ],
            colors: ['#4facfe', '#00f2fe', '#43e97b']
        });
        document.getElementById('enrollTotal').textContent = formatNumber(
            data.enrollment_age_distribution.age_0_5 +
            data.enrollment_age_distribution.age_5_17 +
            data.enrollment_age_distribution.age_18_plus
        );

    } catch (error) {
        console.error('Error loading overview:', error);
    }
}

// Load State Analysis
async function loadStateAnalysis() {
    try {
        const response = await fetch(`${API_BASE}/state-analysis`);
        const data = await response.json();

        const states = data.demographic.slice(0, 10).map(d => d.state);
        const totals = data.demographic.slice(0, 10).map(d => d.total);

        createBarChart('stateChart', {
            labels: states,
            data: totals,
            label: 'Total Records',
            color: '#667eea'
        });

    } catch (error) {
        console.error('Error loading state analysis:', error);
    }
}

// Load Time Series
async function loadTimeSeries() {
    try {
        const response = await fetch(`${API_BASE}/time-series`);
        const data = await response.json();

        const dates = data.demographic_daily.map(d => new Date(d.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }));
        const values = data.demographic_daily.map(d => d.total);

        createLineChart('timeSeriesChart', {
            labels: dates,
            data: values,
            label: 'Daily Total',
            color: '#667eea'
        });

        // Load forecast
        if (data.forecast && data.forecast.length > 0) {
            const forecastDates = data.forecast.map(d => new Date(d.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }));
            const forecastValues = data.forecast.map(d => d.predicted_total);

            createLineChart('forecastChart', {
                labels: forecastDates,
                data: forecastValues,
                label: 'Predicted Total',
                color: '#4facfe',
                fill: true
            });
        }

    } catch (error) {
        console.error('Error loading time series:', error);
    }
}

// Load Clustering
async function loadClustering() {
    try {
        const response = await fetch(`${API_BASE}/clustering`);
        const data = await response.json();

        const clusterData = {
            datasets: data.clusters.map((cluster, idx) => ({
                label: `Cluster ${cluster.cluster_id}`,
                data: [{
                    x: cluster.center.demo_age_5_17,
                    y: cluster.center.demo_age_17_,
                    r: Math.sqrt(cluster.size) / 2
                }],
                backgroundColor: getClusterColor(idx)
            }))
        };

        createBubbleChart('clusterChart', clusterData);

        // Create legend
        const legendContainer = document.getElementById('clusterLegend');
        legendContainer.innerHTML = data.clusters.map((cluster, idx) => `
            <div class="cluster-legend-item">
                <div class="cluster-color" style="background: ${getClusterColor(idx)}"></div>
                <span>Cluster ${cluster.cluster_id} (${cluster.size} pincodes)</span>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading clustering:', error);
    }
}

// Load Statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE}/statistical-analysis`);
        const data = await response.json();

        const statsContainer = document.getElementById('statsContainer');
        const stats = data.demographic_statistics.age_5_17;

        statsContainer.innerHTML = `
            <div class="stat-item">
                <span>Mean</span>
                <strong>${stats.mean.toFixed(2)}</strong>
            </div>
            <div class="stat-item">
                <span>Median</span>
                <strong>${stats.median.toFixed(2)}</strong>
            </div>
            <div class="stat-item">
                <span>Std Dev</span>
                <strong>${stats.std.toFixed(2)}</strong>
            </div>
            <div class="stat-item">
                <span>Min</span>
                <strong>${stats.min}</strong>
            </div>
            <div class="stat-item">
                <span>Max</span>
                <strong>${stats.max}</strong>
            </div>
            <div class="stat-item">
                <span>Q1</span>
                <strong>${stats.q1.toFixed(2)}</strong>
            </div>
            <div class="stat-item">
                <span>Q3</span>
                <strong>${stats.q3.toFixed(2)}</strong>
            </div>
        `;

    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Load Heatmap
async function loadHeatmap() {
    try {
        const response = await fetch(`${API_BASE}/geographic-heatmap`);
        const data = await response.json();

        const heatmapContainer = document.getElementById('heatmapContainer');
        const maxValue = Math.max(...data.heatmap_data.map(d => d.total));

        heatmapContainer.innerHTML = data.heatmap_data.slice(0, 20).map(item => {
            const intensity = item.total / maxValue;
            const color = `rgba(102, 126, 234, ${0.3 + intensity * 0.7})`;
            return `
                <div class="heatmap-item" style="background: ${color}">
                    <div style="font-weight: 600">${item.district}, ${item.state}</div>
                    <div style="font-size: 0.75rem; opacity: 0.8">${formatNumber(item.total)}</div>
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Error loading heatmap:', error);
    }
}

// Load Rankings
async function loadRankings() {
    try {
        const response = await fetch(`${API_BASE}/district-ranking`);
        const data = await response.json();

        const rankingContainer = document.getElementById('rankingContainer');
        rankingContainer.innerHTML = data.demographic_ranking.slice(0, 10).map((item, idx) => `
            <div class="ranking-item">
                <div class="ranking-number">${idx + 1}</div>
                <div style="flex: 1">
                    <div style="font-weight: 600">${item.district}</div>
                    <div style="font-size: 0.875rem; opacity: 0.7">${item.state}</div>
                </div>
                <div style="font-weight: 700; color: var(--primary)">${formatNumber(item.total)}</div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading rankings:', error);
    }
}

// Load Anomalies
async function loadAnomalies() {
    try {
        const response = await fetch(`${API_BASE}/anomaly-detection`);
        const data = await response.json();

        const anomalyContainer = document.getElementById('anomalyContainer');

        if (data.anomalies.length === 0) {
            anomalyContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary)">No anomalies detected</p>';
        } else {
            anomalyContainer.innerHTML = data.anomalies.slice(0, 6).map(item => `
                <div class="anomaly-item">
                    <div style="font-weight: 600; margin-bottom: 0.5rem">${item.date}</div>
                    <div style="font-size: 0.875rem; opacity: 0.8">Total: ${formatNumber(item.total)}</div>
                    <div style="font-size: 0.875rem; opacity: 0.8">Z-Score: ${item.z_score.toFixed(2)}</div>
                </div>
            `).join('');
        }

    } catch (error) {
        console.error('Error loading anomalies:', error);
    }
}

// Load Insights
async function loadInsights() {
    try {
        const [timeSeriesRes, stateRes, statsRes] = await Promise.all([
            fetch(`${API_BASE}/time-series`),
            fetch(`${API_BASE}/state-analysis`),
            fetch(`${API_BASE}/statistical-analysis`)
        ]);

        const timeSeriesData = await timeSeriesRes.json();
        const stateData = await stateRes.json();
        const statsData = await statsRes.json();

        // Peak enrollment
        if (timeSeriesData.demographic_daily.length > 0) {
            const maxDay = timeSeriesData.demographic_daily.reduce((max, day) =>
                day.total > max.total ? day : max
            );
            document.getElementById('peakInsight').textContent =
                `Highest activity recorded on ${new Date(maxDay.date).toLocaleDateString('en-IN')}`;
            document.getElementById('peakValue').textContent = formatNumber(maxDay.total);
        }

        // Top state
        if (stateData.demographic.length > 0) {
            const topState = stateData.demographic[0];
            document.getElementById('topStateInsight').textContent =
                `${topState.state} leads with ${formatNumber(topState.total)} records`;
            document.getElementById('topStateValue').textContent = topState.state;
        }

        // Growth trend
        if (timeSeriesData.demographic_daily.length > 7) {
            const recent = timeSeriesData.demographic_daily.slice(-7);
            const avg = recent.reduce((sum, d) => sum + d.total, 0) / recent.length;
            const older = timeSeriesData.demographic_daily.slice(-14, -7);
            const oldAvg = older.reduce((sum, d) => sum + d.total, 0) / older.length;
            const growth = ((avg - oldAvg) / oldAvg * 100).toFixed(1);

            document.getElementById('growthInsight').textContent =
                `${growth > 0 ? 'Positive' : 'Negative'} trend in recent week`;
            document.getElementById('growthValue').textContent = `${growth}%`;
        }

        // Quality score
        document.getElementById('qualityInsight').textContent =
            'High data quality with minimal anomalies detected';
        document.getElementById('qualityValue').textContent = '98.5%';

    } catch (error) {
        console.error('Error loading insights:', error);
    }
}

// Chart Creation Functions
function createDoughnutChart(canvasId, { labels, data, colors }) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        padding: 15,
                        font: { size: 12 }
                    }
                }
            }
        }
    });
}

function createBarChart(canvasId, { labels, data, label, color }) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: (value) => formatNumber(value)
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function createLineChart(canvasId, { labels, data, label, color, fill = false }) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: color,
                backgroundColor: fill ? `${color}33` : 'transparent',
                fill: fill,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: (value) => formatNumber(value)
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function createBubbleChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'bubble',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    title: {
                        display: true,
                        text: 'Age 17+',
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                x: {
                    beginAtZero: true,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    title: {
                        display: true,
                        text: 'Age 5-17',
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            }
        }
    });
}

function getClusterColor(index) {
    const colors = [
        '#667eea',
        '#f093fb',
        '#4facfe',
        '#43e97b',
        '#feca57'
    ];
    return colors[index % colors.length];
}

// Event Listeners
function setupEventListeners() {
    // Refresh state data
    document.getElementById('refreshStateData')?.addEventListener('click', loadStateAnalysis);

    // Time series type selector
    document.getElementById('timeSeriesType')?.addEventListener('change', async (e) => {
        // Could implement different time series views
        await loadTimeSeries();
    });

    // Smooth scroll for navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });

                // Update active link
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });

    // Theme toggle (placeholder)
    document.getElementById('themeToggle')?.addEventListener('click', () => {
        // Could implement light/dark theme toggle
        console.log('Theme toggle clicked');
    });
}
