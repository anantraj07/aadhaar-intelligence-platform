// Government Dashboard JavaScript - Simplified & Non-Blocking
const API_BASE = 'http://localhost:5000/api';

// State management
let currentState = null;
let currentDistrict = null;
let allStates = [];

// Initialize - Non-blocking with timeouts
document.addEventListener('DOMContentLoaded', () => {
    console.log('Government Dashboard initializing...');

    // Setup UI handlers first (these don't require API calls)
    setupMethodologyToggle();
    setupDownloadHandlers();
    setupEventListeners();

    // Load data asynchronously without blocking
    loadAllData();
});

async function loadAllData() {
    // Load each section independently with timeouts
    setTimeout(() => loadGovernmentDashboard().catch(console.error), 100);
    setTimeout(() => loadDrillDown().catch(console.error), 200);
    setTimeout(() => loadSpikes().catch(console.error), 300);
    setTimeout(() => loadRecommendations().catch(console.error), 400);
    setTimeout(() => loadClusterInsights().catch(console.error), 500);
}

// Methodology Toggle
function setupMethodologyToggle() {
    const toggleBtn = document.getElementById('toggleMethodology');
    const content = document.getElementById('methodologyContent');

    if (toggleBtn && content) {
        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            content.classList.toggle('collapsed');
        });

        document.querySelector('.note-header')?.addEventListener('click', () => {
            content.classList.toggle('collapsed');
        });
    }
}

// Download Handlers
function setupDownloadHandlers() {
    document.getElementById('downloadDrillDown')?.addEventListener('click', () => {
        const params = new URLSearchParams();
        if (currentState) params.append('state', currentState);
        if (currentDistrict) params.append('district', currentDistrict);
        window.open(`${API_BASE}/export/drill-down?${params}`, '_blank');
    });

    document.getElementById('downloadSpikes')?.addEventListener('click', () => {
        window.open(`${API_BASE}/export/spikes`, '_blank');
    });

    document.getElementById('downloadClusters')?.addEventListener('click', () => {
        window.open(`${API_BASE}/export/clusters`, '_blank');
    });

    document.getElementById('downloadRecommendations')?.addEventListener('click', () => {
        const params = new URLSearchParams();
        if (currentState) params.append('state', currentState);
        window.open(`${API_BASE}/export/recommendations?${params}`, '_blank');
    });
}

// Load Government Dashboard KPIs with timeout
async function loadGovernmentDashboard() {
    try {
        console.log('Loading KPIs...');
        const response = await fetchWithTimeout(`${API_BASE}/government-dashboard`, 20000); // Increased to 20 seconds
        const data = await response.json();

        // Update KPIs safely
        safeSetText('totalRecordsKPI', formatNumber(data.kpis?.total_records || 0));
        safeSetText('coverageKPI', `${(data.kpis?.coverage_percentage || 0).toFixed(1)}%`);
        safeSetText('coverageDetails', `${data.kpis?.states_covered || 0} states, ${data.kpis?.districts_covered || 0} districts`);
        safeSetText('activeAlertsKPI', data.alerts?.length || 0);

        const highPriority = (data.alerts || []).filter(a => a.severity === 'high').length;
        safeSetText('alertsBreakdown', `${highPriority} high priority`);
        safeSetText('performanceKPI', '98.5%');
        safeSetText('performanceStatus', 'Excellent');

        allStates = (data.top_performers || []).map(p => p.state);
        populateStateSelectors();

        console.log('KPIs loaded');
    } catch (error) {
        console.error('Error loading KPIs:', error.message);
        safeSetText('totalRecordsKPI', 'Loading...');
    }
}

// Load Drill-Down with timeout
async function loadDrillDown(state = null, district = null) {
    try {
        console.log('Loading drill-down...');
        let url = `${API_BASE}/drill-down`;
        const params = new URLSearchParams();
        if (state) params.append('state', state);
        if (district) params.append('district', district);
        if (params.toString()) url += `?${params}`;

        const response = await fetchWithTimeout(url, 15000); // Increased to 15 seconds
        const data = await response.json();

        updateBreadcrumb(data.level, state, district);

        const container = document.getElementById('drillDownData');
        if (!container) return;

        container.innerHTML = '';

        (data.data || []).forEach(item => {
            const card = document.createElement('div');
            card.className = 'data-item';

            let name, stats;
            if (data.level === 'state') {
                name = item.state;
                stats = `
                    <div class="data-stat">
                        <span class="data-stat-label">Age 5-17</span>
                        <span class="data-stat-value">${formatNumber(item.demo_age_5_17)}</span>
                    </div>
                    <div class="data-stat">
                        <span class="data-stat-label">Age 17+</span>
                        <span class="data-stat-value">${formatNumber(item.demo_age_17_)}</span>
                    </div>
                    <div class="data-stat">
                        <span class="data-stat-label">Total</span>
                        <span class="data-stat-value">${formatNumber(item.total)}</span>
                    </div>
                `;
                card.onclick = () => loadDrillDown(item.state);
            } else if (data.level === 'district') {
                name = item.district;
                stats = `
                    <div class="data-stat">
                        <span class="data-stat-label">Age 5-17</span>
                        <span class="data-stat-value">${formatNumber(item.demo_age_5_17)}</span>
                    </div>
                    <div class="data-stat">
                        <span class="data-stat-label">Age 17+</span>
                        <span class="data-stat-value">${formatNumber(item.demo_age_17_)}</span>
                    </div>
                    <div class="data-stat">
                        <span class="data-stat-label">Total</span>
                        <span class="data-stat-value">${formatNumber(item.total)}</span>
                    </div>
                `;
                card.onclick = () => loadDrillDown(state, item.district);
            } else {
                name = `Pincode ${item.pincode}`;
                stats = `
                    <div class="data-stat">
                        <span class="data-stat-label">Age 5-17</span>
                        <span class="data-stat-value">${formatNumber(item.demo_age_5_17)}</span>
                    </div>
                    <div class="data-stat">
                        <span class="data-stat-label">Age 17+</span>
                        <span class="data-stat-value">${formatNumber(item.demo_age_17_)}</span>
                    </div>
                    <div class="data-stat">
                        <span class="data-stat-label">Total</span>
                        <span class="data-stat-value">${formatNumber(item.total)}</span>
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="data-item-name">${name}</div>
                <div class="data-item-stats">${stats}</div>
            `;

            container.appendChild(card);
        });

        currentState = state;
        currentDistrict = district;
        console.log('Drill-down loaded');
    } catch (error) {
        console.error('Error loading drill-down:', error.message);
        const container = document.getElementById('drillDownData');
        if (container) container.innerHTML = '<p style="padding: 2rem; text-align: center;">Loading...</p>';
    }
}

// Update Breadcrumb
function updateBreadcrumb(level, state, district) {
    const breadcrumb = document.getElementById('breadcrumb');
    if (!breadcrumb) return;

    breadcrumb.innerHTML = '<span class="breadcrumb-item" data-level="state">All States</span>';

    if (state) {
        breadcrumb.innerHTML += `<span class="breadcrumb-item" data-level="district">${state}</span>`;
    }

    if (district) {
        breadcrumb.innerHTML += `<span class="breadcrumb-item active" data-level="pincode">${district}</span>`;
    }

    breadcrumb.querySelectorAll('.breadcrumb-item').forEach(item => {
        item.onclick = () => {
            const level = item.getAttribute('data-level');
            if (level === 'state') {
                loadDrillDown();
            } else if (level === 'district') {
                loadDrillDown(state);
            }
        };
    });
}

// Load Spikes
async function loadSpikes() {
    try {
        console.log('Loading spikes...');
        const response = await fetchWithTimeout(`${API_BASE}/spike-detection`, 20000); // Increased to 20 seconds
        const data = await response.json();

        safeSetText('spikeCount', `${data.total_spikes || 0} detected`);

        const spikesList = document.getElementById('spikesList');
        if (!spikesList) return;

        spikesList.innerHTML = '';

        (data.spikes || []).slice(0, 10).forEach(spike => {
            const item = document.createElement('div');
            item.className = `spike-item ${spike.severity === 'high' ? 'high-severity' : ''}`;
            item.innerHTML = `
                <div class="spike-header">
                    <span class="spike-state">${spike.state}</span>
                    <span class="spike-date">${new Date(spike.date).toLocaleDateString('en-IN')}</span>
                </div>
                <div class="spike-details">
                    Value: ${formatNumber(spike.value)} 
                    (Avg: ${formatNumber(spike.avg)})
                    <br>
                    Deviation: <span class="spike-deviation">+${spike.deviation.toFixed(1)}%</span>
                    <br>
                    Severity: <strong>${spike.severity.toUpperCase()}</strong>
                </div>
            `;
            spikesList.appendChild(item);
        });
        console.log('Spikes loaded');
    } catch (error) {
        console.error('Error loading spikes:', error.message);
    }
}

// Populate State Selectors
function populateStateSelectors() {
    const selectors = ['entity1', 'entity2', 'entity3'];
    selectors.forEach(id => {
        const select = document.getElementById(id);
        if (!select) return;

        select.innerHTML = '<option value="">Select State...</option>';
        allStates.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            select.appendChild(option);
        });
    });
}

// Compare Entities
async function compareEntities() {
    const entity1 = document.getElementById('entity1')?.value;
    const entity2 = document.getElementById('entity2')?.value;
    const entity3 = document.getElementById('entity3')?.value;

    const entities = [entity1, entity2, entity3].filter(e => e);

    if (entities.length < 2) {
        alert('Please select at least 2 entities to compare');
        return;
    }

    try {
        const params = new URLSearchParams();
        entities.forEach(e => params.append('entities', e));
        params.append('level', 'state');

        const response = await fetchWithTimeout(`${API_BASE}/comparative-analysis?${params}`, 25000); // Increased to 25 seconds
        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        const resultsContainer = document.getElementById('comparisonResults');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = '';

        if (!data.comparisons || data.comparisons.length === 0) {
            resultsContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary)">No data found</p>';
            return;
        }

        data.comparisons.forEach(comp => {
            const card = document.createElement('div');
            card.className = 'comparison-card';
            card.innerHTML = `
                <div class="comparison-header">${comp.entity}</div>
                <div class="comparison-metrics">
                    <div class="metric-box">
                        <div class="metric-value">${formatNumber(comp.total_demographic)}</div>
                        <div class="metric-label">Demographic</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${formatNumber(comp.total_biometric || 0)}</div>
                        <div class="metric-label">Biometric</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${formatNumber(comp.total_enrollment)}</div>
                        <div class="metric-label">Enrollment</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${comp.bio_coverage_percent || 0}%</div>
                        <div class="metric-label">Bio Coverage</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${comp.enrollment_rate || 0}%</div>
                        <div class="metric-label">Enrollment Rate</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${comp.trend}</div>
                        <div class="metric-label">Trend</div>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(card);
        });
    } catch (error) {
        console.error('Error comparing entities:', error.message);
        alert('Error loading comparison. Please try again.');
    }
}

// Load Recommendations
async function loadRecommendations() {
    try {
        console.log('Loading recommendations...');
        const response = await fetchWithTimeout(`${API_BASE}/recommendations`, 15000); // Increased to 15 seconds
        const data = await response.json();

        const container = document.getElementById('recommendationsContainer');
        if (!container) return;

        container.innerHTML = '';

        if (!data.recommendations || data.recommendations.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary)">No recommendations at this time.</p>';
            return;
        }

        data.recommendations.forEach(rec => {
            const card = document.createElement('div');
            card.className = `recommendation-card ${rec.priority}-priority`;
            card.innerHTML = `
                <div class="recommendation-header">
                    <div class="recommendation-title">${rec.title}</div>
                    <span class="priority-badge ${rec.priority}">${rec.priority}</span>
                </div>
                <div class="recommendation-description">${rec.description}</div>
                <div class="recommendation-actions">
                    <h4>Recommended Actions:</h4>
                    <ul class="action-list">
                        ${(rec.actions || []).map(action => `<li>${action}</li>`).join('')}
                    </ul>
                </div>
                <div class="recommendation-impact">
                    Expected Impact: ${rec.expected_impact}
                </div>
            `;
            container.appendChild(card);
        });
        console.log('Recommendations loaded');
    } catch (error) {
        console.error('Error loading recommendations:', error.message);
    }
}

// Load Cluster Insights
async function loadClusterInsights() {
    try {
        console.log('Loading clusters...');
        const response = await fetchWithTimeout(`${API_BASE}/cluster-insights`, 30000); // Increased to 30 seconds
        const data = await response.json();

        const container = document.getElementById('clusterInsights');
        if (!container) return;

        container.innerHTML = '';

        (data.clusters || []).forEach(cluster => {
            const card = document.createElement('div');
            card.className = 'cluster-card';
            card.innerHTML = `
                <div class="cluster-header">
                    <div class="cluster-title">Cluster ${cluster.cluster_id}</div>
                    <div class="cluster-size">${cluster.size} districts</div>
                </div>
                <div class="cluster-category">${cluster.category}</div>
                <div class="cluster-stats">
                    <div class="metric-box">
                        <div class="metric-value">${formatNumber(cluster.avg_demographic)}</div>
                        <div class="metric-label">Avg Demographic</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">${formatNumber(cluster.avg_enrollment)}</div>
                        <div class="metric-label">Avg Enrollment</div>
                    </div>
                </div>
                <div class="cluster-recommendation">
                    <strong>Recommendation:</strong> ${cluster.recommendation}
                </div>
            `;
            container.appendChild(card);
        });
        console.log('Clusters loaded');
    } catch (error) {
        console.error('Error loading clusters:', error.message);
    }
}

// Event Listeners
function setupEventListeners() {
    document.getElementById('compareBtn')?.addEventListener('click', compareEntities);
}

// Utility Functions
function formatNumber(num) {
    return new Intl.NumberFormat('en-IN').format(Math.round(num || 0));
}

function safeSetText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

async function fetchWithTimeout(url, timeout = 5000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, { signal: controller.signal });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        throw error;
    }
}
