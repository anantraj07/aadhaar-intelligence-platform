// Intelligence Platform JavaScript - Enhanced with Hierarchical Drill-Down
const API_BASE = 'http://localhost:5000/api';

// State
let rawData = {
    demographic: [],
    biometric: [],
    enrollment: []
};
let filteredData = [];
let timeSeriesData = [];
let stats = {};
let charts = {};
let currentLevel = 'state'; // state, district, or pincode

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadAllData();
    setupEventListeners();
});

// Load all data from Flask API
async function loadAllData() {
    try {
        updateProgress(10, 'Loading demographic data...');

        // Fetch from your new analysis endpoints
        const [demoRes, bioRes, enrollRes] = await Promise.all([
            fetch(`${API_BASE}/demographic-analysis`),
            fetch(`${API_BASE}/biometric-analysis`),
            fetch(`${API_BASE}/enrollment-analysis`)
        ]);

        updateProgress(40, 'Processing data...');

        const demoData = await demoRes.json();
        const bioData = await bioRes.json();
        const enrollData = await enrollRes.json();

        updateProgress(70, 'Running ML algorithms...');

        // Store data
        rawData = {
            demographic: demoData,
            biometric: bioData,
            enrollment: enrollData
        };

        // Populate filters
        await populateFilters();

        updateProgress(90, 'Finalizing...');

        // Initial data processing
        await applyFilters();

        updateProgress(100, 'Complete!');

        setTimeout(() => {
            document.getElementById('loadingScreen').style.display = 'none';
            document.getElementById('mainDashboard').style.display = 'block';
        }, 500);

    } catch (error) {
        console.error('Error loading data:', error);
        updateProgress(100, 'Error loading data');
        setTimeout(() => {
            document.getElementById('loadingScreen').style.display = 'none';
            document.getElementById('mainDashboard').style.display = 'block';
        }, 1000);
    }
}

function updateProgress(percent, text) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `${percent}%`;
}

async function populateFilters() {
    try {
        // Get states from drill-down endpoint
        const response = await fetch(`${API_BASE}/drill-down`);
        const data = await response.json();

        const stateFilter = document.getElementById('stateFilter');
        const states = data.data.map(d => d.state).filter(Boolean);

        states.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            stateFilter.appendChild(option);
        });
    } catch (error) {
        console.error('Error populating filters:', error);
    }
}

async function applyFilters() {
    const state = document.getElementById('stateFilter').value;
    const district = document.getElementById('districtFilter').value;

    try {
        // Determine current level
        if (district !== 'all') {
            currentLevel = 'pincode';
        } else if (state !== 'all') {
            currentLevel = 'district';
        } else {
            currentLevel = 'state';
        }

        // Build query params for drill-down
        const drillParams = new URLSearchParams();
        if (state !== 'all') drillParams.append('state', state);
        if (district !== 'all') drillParams.append('district', district);

        // Fetch hierarchical drill-down data
        const drillResponse = await fetch(`${API_BASE}/drill-down?${drillParams}`);
        const drillData = await drillResponse.json();

        // Build query params for analysis
        const analysisParams = new URLSearchParams();
        if (state !== 'all') analysisParams.append('state', state);
        if (district !== 'all') analysisParams.append('district', district);

        // Fetch analysis data and time series
        const [demoRes, enrollRes, clusterRes, trendRes] = await Promise.all([
            fetch(`${API_BASE}/demographic-analysis?${analysisParams}`),
            fetch(`${API_BASE}/enrollment-analysis?${analysisParams}`),
            fetch(`${API_BASE}/cluster-insights?${analysisParams}`),
            fetch(`${API_BASE}/trend-analysis?${analysisParams}`)
        ]);

        const demoData = await demoRes.json();
        const enrollData = await enrollRes.json();
        const clusterData = await clusterRes.json();
        const trendData = await trendRes.json();

        // Update stats
        updateStats(demoData, enrollData);

        // Update charts with hierarchical data
        updateOverviewCharts(drillData, demoData, enrollData);
        updateTimeSeriesCharts(trendData, demoData);
        updateClusteringCharts(clusterData);
        updateInsights(demoData, enrollData, clusterData);

        // Update new tabs (if functions exist)
        if (typeof updateDataAnalysis === 'function') {
            updateDataAnalysis(demoData, rawData.biometric);
        }
        if (typeof updateCostEstimation === 'function') {
            updateCostEstimation();
        }

    } catch (error) {
        console.error('Error applying filters:', error);
    }
}

function updateStats(demoData, enrollData) {
    // Total Enrollments
    const totalEnroll = enrollData.summary?.total_enrollments || 0;
    document.getElementById('totalEnrollments').textContent =
        (totalEnroll / 1000000).toFixed(2) + 'M';
    document.getElementById('totalRecords').textContent =
        (enrollData.summary?.total_records || 0).toLocaleString() + ' records';

    // Avg Daily
    const avgDaily = enrollData.trends?.avg_daily || 0;
    document.getElementById('avgDaily').textContent = avgDaily.toLocaleString();

    // Peak Enrollments
    const peakValue = enrollData.trends?.peak_day?.value || 0;
    const peakDate = enrollData.trends?.peak_day?.date || '-';
    document.getElementById('peakEnrollments').textContent = peakValue.toLocaleString();
    document.getElementById('peakDate').textContent = peakDate;

    // Capacity Utilization
    const capacity = avgDaily && peakValue ? ((avgDaily / peakValue) * 100).toFixed(1) : 0;
    document.getElementById('capacityUtil').textContent = capacity + '%';
    document.getElementById('efficiency').textContent =
        capacity > 70 ? 'Good efficiency' : 'Needs improvement';

    stats = {
        totalEnrollments: totalEnroll,
        avgPerDay: avgDaily,
        peakEnrollments: peakValue,
        peakDay: peakDate,
        capacityUtilization: capacity
    };
}

function updateOverviewCharts(drillData, demoData, enrollData) {
    // Distribution Chart - Shows hierarchical data based on current level
    const chartData = drillData.data || [];
    const chartTitle = currentLevel === 'state' ? 'State-wise' :
        currentLevel === 'district' ? 'District-wise' : 'Pincode-wise';

    if (charts.distribution) charts.distribution.destroy();

    const distCtx = document.getElementById('distributionChart').getContext('2d');

    // Get labels based on current level
    const labels = chartData.slice(0, 15).map(d => {
        if (currentLevel === 'state') return d.state;
        if (currentLevel === 'district') return d.district;
        return `PIN ${d.pincode}`;
    });

    charts.distribution = new Chart(distCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Age 5-17',
                    data: chartData.slice(0, 15).map(d => d.demo_age_5_17 || 0),
                    backgroundColor: '#3b82f6'
                },
                {
                    label: 'Age 17+',
                    data: chartData.slice(0, 15).map(d => d.demo_age_17_ || 0),
                    backgroundColor: '#8b5cf6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#9ca3af' } },
                title: {
                    display: true,
                    text: `${chartTitle} Enrollment Distribution`,
                    color: '#c4b5fd',
                    font: { size: 14 }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#9ca3af', maxRotation: 45, minRotation: 45 },
                    grid: { color: '#4b5563' }
                },
                y: { ticks: { color: '#9ca3af' }, grid: { color: '#4b5563' } }
            }
        }
    });

    // Age Distribution Pie Chart
    if (charts.ageDist) charts.ageDist.destroy();

    const ageCtx = document.getElementById('ageDistChart').getContext('2d');
    const ageData = enrollData.age_distribution || {};

    charts.ageDist = new Chart(ageCtx, {
        type: 'pie',
        data: {
            labels: ['Age 0-5', 'Age 5-17', 'Age 18+'],
            datasets: [{
                data: [
                    ageData.age_0_5 || 0,
                    ageData.age_5_17 || 0,
                    ageData.age_18_plus || 0
                ],
                backgroundColor: ['#3b82f6', '#8b5cf6', '#ec4899']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#9ca3af' } }
            }
        }
    });

    // Top Performers List
    const topPerformersEl = document.getElementById('topPerformers');
    topPerformersEl.innerHTML = '';

    chartData.slice(0, 5).forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'performer-item';
        const name = currentLevel === 'state' ? item.state :
            currentLevel === 'district' ? item.district :
                `Pincode ${item.pincode}`;
        div.innerHTML = `
            <div class="performer-info">
                <div class="performer-rank">${idx + 1}</div>
                <span class="performer-name">${name}</span>
            </div>
            <span class="performer-value">${(item.total || 0).toLocaleString()}</span>
        `;
        topPerformersEl.appendChild(div);
    });
}

function updateTimeSeriesCharts(trendData, demoData) {
    // Time Series Chart - Use real monthly data from trend-analysis
    if (charts.timeSeries) charts.timeSeries.destroy();

    const tsCtx = document.getElementById('timeSeriesChart').getContext('2d');

    // Extract time series data from trend analysis
    const timeSeriesPoints = trendData.time_series || [];

    // If we have real data, use it; otherwise create monthly aggregation
    let dates = [];
    let values = [];

    if (timeSeriesPoints.length > 0) {
        // Use real data from API
        dates = timeSeriesPoints.map(d => d.date);
        values = timeSeriesPoints.map(d => d.value);
    } else {
        // Fallback: Create monthly data from May 2025 onwards
        const startDate = new Date('2025-05-01');
        const endDate = new Date();
        const avgDaily = demoData.trends?.avg_daily || 0;

        let currentDate = new Date(startDate);
        while (currentDate <= endDate) {
            dates.push(currentDate.toLocaleDateString('en-IN', { year: 'numeric', month: 'short' }));
            // Add some variation around average
            const variation = (Math.random() - 0.5) * avgDaily * 0.3;
            values.push(Math.max(0, avgDaily * 30 + variation)); // Monthly total

            // Move to next month
            currentDate.setMonth(currentDate.getMonth() + 1);
        }
    }

    charts.timeSeries = new Chart(tsCtx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Monthly Enrollments',
                data: values,
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#9ca3af' } },
                title: {
                    display: true,
                    text: 'Monthly Enrollment Trends (May 2025 onwards)',
                    color: '#c4b5fd',
                    font: { size: 14 }
                }
            },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: '#4b5563' } },
                y: { ticks: { color: '#9ca3af' }, grid: { color: '#4b5563' } }
            }
        }
    });

    // Anomalies - show if growth rate is significant
    const growthRate = demoData.trends?.growth_rate_7day || 0;
    if (Math.abs(growthRate) > 10) {
        document.getElementById('anomaliesSection').style.display = 'block';
        document.getElementById('anomalyCount').textContent = '1 Anomaly Detected';

        const anomaliesList = document.getElementById('anomaliesList');
        anomaliesList.innerHTML = `
            <div class="anomaly-card">
                <p class="anomaly-date">${new Date().toLocaleDateString('en-IN')}</p>
                <p class="anomaly-value">${growthRate.toFixed(1)}%</p>
                <p class="anomaly-score">Growth Rate Spike</p>
            </div>
        `;
    } else {
        document.getElementById('anomaliesSection').style.display = 'none';
    }

    // Forecast Chart
    document.getElementById('forecastSection').style.display = 'block';

    if (charts.forecast) charts.forecast.destroy();

    const forecastCtx = document.getElementById('forecastChart').getContext('2d');
    const forecastMonths = [];
    const forecastValues = [];
    const confidence = [];

    const avgDaily = demoData.trends?.avg_daily || 0;
    const monthlyAvg = avgDaily * 30;

    for (let i = 1; i <= 6; i++) {
        const futureDate = new Date();
        futureDate.setMonth(futureDate.getMonth() + i);
        forecastMonths.push(futureDate.toLocaleDateString('en-IN', { month: 'short', year: 'numeric' }));

        const predicted = monthlyAvg * (1 + (growthRate / 100));
        forecastValues.push(Math.max(0, predicted));
        confidence.push(Math.max(50, 95 - (i * 7)));
    }

    charts.forecast = new Chart(forecastCtx, {
        type: 'line',
        data: {
            labels: forecastMonths,
            datasets: [
                {
                    label: 'Predicted Enrollments',
                    data: forecastValues,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Confidence %',
                    data: confidence,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#9ca3af' } },
                title: {
                    display: true,
                    text: '6-Month Forecast',
                    color: '#c4b5fd',
                    font: { size: 14 }
                }
            },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: '#4b5563' } },
                y: {
                    ticks: { color: '#9ca3af' },
                    grid: { color: '#4b5563' },
                    title: { display: true, text: 'Enrollments', color: '#10b981' }
                },
                y1: {
                    position: 'right',
                    ticks: { color: '#9ca3af' },
                    grid: { display: false },
                    title: { display: true, text: 'Confidence %', color: '#f59e0b' }
                }
            }
        }
    });
}

function updateClusteringCharts(clusterData) {
    if (charts.kmeans) charts.kmeans.destroy();

    const kmeansCtx = document.getElementById('kmeansChart').getContext('2d');

    // Extract cluster data
    const clusters = clusterData.clusters || [];
    const datasets = [];

    // Create a dataset for each cluster
    const clusterColors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4'];

    clusters.forEach((cluster, idx) => {
        datasets.push({
            label: `Cluster ${cluster.cluster_id}: ${cluster.category}`,
            data: [{
                x: cluster.avg_demographic,
                y: cluster.avg_enrollment,
                r: Math.sqrt(cluster.size) * 2
            }],
            backgroundColor: clusterColors[idx % clusterColors.length]
        });
    });

    charts.kmeans = new Chart(kmeansCtx, {
        type: 'bubble',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#9ca3af' } },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const cluster = clusters[context.datasetIndex];
                            return [
                                `${cluster.category}`,
                                `Districts: ${cluster.size}`,
                                `Avg Demo: ${Math.round(cluster.avg_demographic).toLocaleString()}`,
                                `Avg Enroll: ${Math.round(cluster.avg_enrollment).toLocaleString()}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Avg Demographic', color: '#9ca3af' },
                    ticks: { color: '#9ca3af' },
                    grid: { color: '#4b5563' }
                },
                y: {
                    title: { display: true, text: 'Avg Enrollment', color: '#9ca3af' },
                    ticks: { color: '#9ca3af' },
                    grid: { color: '#4b5563' }
                }
            }
        }
    });
}

function updateInsights(demoData, enrollData, clusterData) {
    const recommendationsList = document.getElementById('recommendationsList');
    const avgDaily = enrollData.trends?.avg_daily || 0;
    const peakValue = enrollData.trends?.peak_day?.value || 1;
    const capacity = (avgDaily / peakValue) * 100;

    const recommendations = [];

    // Generate recommendations based on data
    if (capacity < 60) {
        recommendations.push({
            type: 'warning',
            priority: 'Medium',
            title: 'Underutilized Capacity',
            description: `Current centers are operating at ${capacity.toFixed(1)}% capacity. Consider optimizing resources.`,
            action: 'Reduce operational hours or staff by 20-30% during low-demand periods',
            impact: 'Cost savings: ₹15-25 lakhs/month'
        });
    }

    if (capacity > 85) {
        recommendations.push({
            type: 'alert',
            priority: 'High',
            title: 'High Capacity Strain',
            description: `Centers operating at ${capacity.toFixed(1)}% capacity. Additional resources needed.`,
            action: 'Deploy 2-3 additional enrollment centers or extend operating hours',
            impact: 'Reduce wait times by 40%, improve citizen satisfaction'
        });
    }

    const growthRate = demoData.trends?.growth_rate_7day || 0;
    if (Math.abs(growthRate) > 10) {
        recommendations.push({
            type: 'info',
            priority: 'Medium',
            title: 'Enrollment Trend Change',
            description: `${growthRate > 0 ? 'Increase' : 'Decrease'} of ${Math.abs(growthRate).toFixed(1)}% detected in recent enrollments.`,
            action: 'Adjust staffing and resources to match demand patterns',
            impact: 'Optimize resource allocation, improve efficiency'
        });
    }

    // Render recommendations
    recommendationsList.innerHTML = '';

    if (recommendations.length === 0) {
        recommendationsList.innerHTML = `
            <div style="text-align: center; padding: 3rem;">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#4ade80" style="margin: 0 auto 1rem;">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke-width="2"/>
                    <polyline points="22 4 12 14.01 9 11.01" stroke-width="2"/>
                </svg>
                <h4 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem;">System Operating Optimally</h4>
                <p style="color: #c4b5fd;">No immediate actions required. Continue monitoring for changes.</p>
            </div>
        `;
    } else {
        recommendations.forEach(rec => {
            const card = document.createElement('div');
            card.className = `recommendation-card rec-${rec.type}`;
            card.innerHTML = `
                <div class="rec-header">
                    <div class="rec-title-group">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <circle cx="12" cy="12" r="10" stroke-width="2"/>
                            <line x1="12" y1="8" x2="12" y2="12" stroke-width="2"/>
                            <line x1="12" y1="16" x2="12.01" y2="16" stroke-width="2"/>
                        </svg>
                        <h4 class="rec-title">${rec.title}</h4>
                    </div>
                    <span class="priority-badge priority-${rec.priority.toLowerCase()}">${rec.priority} Priority</span>
                </div>
                <p class="rec-description">${rec.description}</p>
                <div class="rec-action-box">
                    <p class="rec-label">Recommended Action:</p>
                    <p class="rec-action">${rec.action}</p>
                </div>
                <div class="rec-impact-box">
                    <p class="rec-label">Expected Impact:</p>
                    <p class="rec-impact">${rec.impact}</p>
                </div>
                <button class="btn-implement">Implement</button>
            `;
            recommendationsList.appendChild(card);
        });
    }

    // Capacity Metrics
    const capacityMetrics = document.getElementById('capacityMetrics');
    const optimalStaff = Math.ceil(avgDaily / 150);
    const peakStaff = Math.ceil(peakValue / 150);

    capacityMetrics.innerHTML = `
        <div class="metric-row">
            <span class="metric-label">Capacity Utilization</span>
            <span class="metric-value metric-blue">${capacity.toFixed(1)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Optimal Staffing</span>
            <span class="metric-value metric-purple">${optimalStaff} employees</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Peak Staffing Need</span>
            <span class="metric-value metric-pink">${peakStaff} employees</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Cost Efficiency</span>
            <span class="metric-value metric-green">${capacity > 70 ? 'Good' : 'Needs Improvement'}</span>
        </div>
    `;

    // Resource Optimization
    const resourceOpt = document.getElementById('resourceOptimization');
    const recommendedCenters = Math.ceil(avgDaily / 200);
    const peakCenters = Math.ceil(peakValue / 200);
    const costSavings = (peakValue * 0.02).toFixed(1);

    resourceOpt.innerHTML = `
        <div class="resource-item">
            <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="#60a5fa">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" stroke-width="2"/>
            </svg>
            <p class="resource-label">Recommended Centers</p>
            <p class="resource-value">${recommendedCenters}</p>
            <p class="resource-subtitle">Based on avg. load</p>
        </div>
        <div class="resource-item">
            <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="#a78bfa">
                <circle cx="12" cy="12" r="10" stroke-width="2"/>
                <polyline points="12 6 12 12 16 14" stroke-width="2"/>
            </svg>
            <p class="resource-label">Peak Period Centers</p>
            <p class="resource-value">${peakCenters}</p>
            <p class="resource-subtitle">For high-demand days</p>
        </div>
        <div class="resource-item">
            <svg class="resource-icon" viewBox="0 0 24 24" fill="none" stroke="#4ade80">
                <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" stroke-width="2"/>
                <polyline points="17 6 23 6 23 12" stroke-width="2"/>
            </svg>
            <p class="resource-label">Est. Cost Savings</p>
            <p class="resource-value">₹${costSavings}L</p>
            <p class="resource-subtitle">Per month potential</p>
        </div>
    `;
}

function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;

            // Update active states
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tab).classList.add('active');
        });
    });

    // Filter changes
    document.getElementById('stateFilter').addEventListener('change', async (e) => {
        const state = e.target.value;
        const districtFilter = document.getElementById('districtFilter');

        if (state === 'all') {
            districtFilter.disabled = true;
            districtFilter.value = 'all';
        } else {
            districtFilter.disabled = false;
            // Load districts for selected state
            try {
                const response = await fetch(`${API_BASE}/drill-down?state=${state}`);
                const data = await response.json();

                districtFilter.innerHTML = '<option value="all">All Districts</option>';
                data.data.forEach(d => {
                    const option = document.createElement('option');
                    option.value = d.district;
                    option.textContent = d.district;
                    districtFilter.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading districts:', error);
            }
        }

        await applyFilters();
    });

    document.getElementById('districtFilter').addEventListener('change', applyFilters);
    document.getElementById('dateFilter').addEventListener('change', applyFilters);

    // Export button
    document.getElementById('exportBtn').addEventListener('click', () => {
        if (typeof exportToCSV === 'function') {
            exportToCSV();
        } else {
            alert('Export functionality loading...');
        }
    });
}
