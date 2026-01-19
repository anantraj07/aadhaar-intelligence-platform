
// Export current data to CSV
function exportToCSV() {
    const state = document.getElementById('stateFilter').value;
    const district = document.getElementById('districtFilter').value;

    // Build CSV content
    let csvContent = "data:text/csv;charset=utf-8,";

    // Headers
    csvContent += "Level,Name,Total_Enrollments,Avg_Daily,Peak_Enrollments,Capacity_Util,Demographic_5_17,Demographic_17+,Biometric_5_17,Biometric_17+,Estimated_Monthly_Cost\n";

    // Data row
    const level = district !== 'all' ? 'Pincode' : state !== 'all' ? 'District' : 'State';
    const name = district !== 'all' ? district : state !== 'all' ? state : 'All India';
    const totalEnroll = stats.totalEnrollments || 0;
    const avgDaily = stats.avgPerDay || 0;
    const peak = stats.peakEnrollments || 0;
    const capacity = stats.capacityUtilization || 0;

    // Get demographic and biometric data
    const demoData = rawData.demographic;
    const bioData = rawData.biometric;

    const demo517 = demoData.summary?.total_age_5_17 || 0;
    const demo17plus = demoData.summary?.total_age_17_plus || 0;
    const bio517 = bioData.summary?.total_age_5_17 || 0;
    const bio17plus = bioData.summary?.total_age_17_plus || 0;

    // Calculate cost
    const monthlyCost = calculateMonthlyCost(avgDaily);

    csvContent += `${level},${name},${totalEnroll},${avgDaily},${peak},${capacity}%,${demo517},${demo17plus},${bio517},${bio17plus},₹${monthlyCost.toLocaleString()}\n`;

    // Create download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `aadhaar_intelligence_report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Calculate monthly cost based on enrollment volume
function calculateMonthlyCost(avgDaily) {
    const staffRequired = Math.ceil(avgDaily / 150); // 150 enrollments per staff per day
    const staffCost = staffRequired * 25000; // ₹25,000 per staff per month
    const infrastructure = 50000; // Rent
    const utilities = 15000;
    const internet = 5000;
    const maintenance = 10000;
    const supplies = 8000;

    return staffCost + infrastructure + utilities + internet + maintenance + supplies;
}

// Update data analysis tab
function updateDataAnalysis(demoData, bioData) {
    // Demographic Summary
    const demoSummary = document.getElementById('demographicSummary');
    const demo517 = demoData.summary?.total_age_5_17 || 0;
    const demo17plus = demoData.summary?.total_age_17_plus || 0;
    const demoTotal = demo517 + demo17plus;

    demoSummary.innerHTML = `
        <div class="metric-row">
            <span class="metric-label">Total Demographic Updates</span>
            <span class="metric-value metric-blue">${demoTotal.toLocaleString()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Age 5-17</span>
            <span class="metric-value metric-purple">${demo517.toLocaleString()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Age 17+</span>
            <span class="metric-value metric-pink">${demo17plus.toLocaleString()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Coverage Rate</span>
            <span class="metric-value metric-green">${((demoTotal / stats.totalEnrollments) * 100).toFixed(1)}%</span>
        </div>
    `;

    // Biometric Summary
    const bioSummary = document.getElementById('biometricSummary');
    const bio517 = bioData.summary?.total_age_5_17 || 0;
    const bio17plus = bioData.summary?.total_age_17_plus || 0;
    const bioTotal = bio517 + bio17plus;

    bioSummary.innerHTML = `
        <div class="metric-row">
            <span class="metric-label">Total Biometric Updates</span>
            <span class="metric-value metric-blue">${bioTotal.toLocaleString()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Age 5-17</span>
            <span class="metric-value metric-purple">${bio517.toLocaleString()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Age 17+</span>
            <span class="metric-value metric-pink">${bio17plus.toLocaleString()}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Coverage Rate</span>
            <span class="metric-value metric-green">${((bioTotal / demoTotal) * 100).toFixed(1)}%</span>
        </div>
    `;

    // Coverage Comparison Chart
    if (charts.coverage) charts.coverage.destroy();

    const coverageCtx = document.getElementById('coverageComparisonChart').getContext('2d');
    charts.coverage = new Chart(coverageCtx, {
        type: 'bar',
        data: {
            labels: ['Age 5-17', 'Age 17+'],
            datasets: [
                {
                    label: 'Demographic Updates',
                    data: [demo517, demo17plus],
                    backgroundColor: '#3b82f6'
                },
                {
                    label: 'Biometric Updates',
                    data: [bio517, bio17plus],
                    backgroundColor: '#8b5cf6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#9ca3af' } }
            },
            scales: {
                x: { ticks: { color: '#9ca3af' }, grid: { color: '#4b5563' } },
                y: { ticks: { color: '#9ca3af' }, grid: { color: '#4b5563' } }
            }
        }
    });
}

// Update cost estimation
function updateCostEstimation() {
    const avgDaily = stats.avgPerDay || 0;
    const peakDaily = stats.peakEnrollments || 0;

    // Get current filter level
    const state = document.getElementById('stateFilter')?.value || 'all';
    const district = document.getElementById('districtFilter')?.value || 'all';

    // Determine region name and level
    let regionName = 'All India';
    let regionLevel = 'National';
    if (district !== 'all') {
        regionName = district;
        regionLevel = 'Pincode';
    } else if (state !== 'all') {
        regionName = state;
        regionLevel = 'District';
    }

    // Calculate costs
    const staffRequired = Math.ceil(avgDaily / 150);
    const peakStaffRequired = Math.ceil(peakDaily / 150);
    const staffCost = staffRequired * 25000;
    const peakStaffCost = peakStaffRequired * 25000;

    const infrastructure = 50000;
    const utilities = 15000;
    const internet = 5000;
    const maintenance = 10000;
    const supplies = 8000;

    const fixedCosts = infrastructure + utilities + internet + maintenance + supplies;
    const monthlyTotal = staffCost + fixedCosts;
    const annualTotal = monthlyTotal * 12;
    const costPerEnrollment = avgDaily > 0 ? (monthlyTotal / (avgDaily * 30)).toFixed(2) : 0;

    const costEstDiv = document.getElementById('costEstimation');
    costEstDiv.innerHTML = `
        <div class="cost-breakdown">
            <div style="background: rgba(37, 99, 235, 0.15); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; border: 1px solid rgba(37, 99, 235, 0.3);">
                <h4 style="color: #60a5fa; margin-bottom: 0.5rem; font-size: 1.25rem;">📍 ${regionName}</h4>
                <p style="color: #c4b5fd; font-size: 0.875rem;">Cost Analysis Level: <span style="color: #a78bfa; font-weight: 600;">${regionLevel}</span></p>
                <p style="color: #e9d5ff; font-size: 0.875rem; margin-top: 0.5rem;">Based on ${avgDaily.toLocaleString()} avg daily enrollments</p>
            </div>
            
            <h4 style="color: #c4b5fd; margin-bottom: 1.5rem; font-size: 1.125rem;">Monthly Cost Breakdown</h4>
            
            <div class="cost-section">
                <h5 style="color: #a78bfa; margin-bottom: 1rem;">👥 Staff Costs</h5>
                <div class="cost-item">
                    <span>Operators Required (Avg)</span>
                    <span class="cost-value">${staffRequired} staff</span>
                </div>
                <div class="cost-item">
                    <span>Operators Required (Peak)</span>
                    <span class="cost-value">${peakStaffRequired} staff</span>
                </div>
                <div class="cost-item">
                    <span>Monthly Staff Cost (@ ₹25,000/staff)</span>
                    <span class="cost-value">₹${staffCost.toLocaleString()}</span>
                </div>
            </div>
            
            <div class="cost-section">
                <h5 style="color: #a78bfa; margin-bottom: 1rem;">🏢 Infrastructure & Operational</h5>
                <div class="cost-item">
                    <span>Rent</span>
                    <span class="cost-value">₹${infrastructure.toLocaleString()}</span>
                </div>
                <div class="cost-item">
                    <span>Utilities</span>
                    <span class="cost-value">₹${utilities.toLocaleString()}</span>
                </div>
                <div class="cost-item">
                    <span>Internet & Connectivity</span>
                    <span class="cost-value">₹${internet.toLocaleString()}</span>
                </div>
                <div class="cost-item">
                    <span>Maintenance</span>
                    <span class="cost-value">₹${maintenance.toLocaleString()}</span>
                </div>
                <div class="cost-item">
                    <span>Supplies</span>
                    <span class="cost-value">₹${supplies.toLocaleString()}</span>
                </div>
            </div>
            
            <div class="cost-section cost-total">
                <div class="cost-item">
                    <span style="font-weight: 700; font-size: 1.125rem;">💰 Total Monthly Cost</span>
                    <span class="cost-value" style="font-size: 1.5rem; color: #60a5fa;">₹${monthlyTotal.toLocaleString()}</span>
                </div>
                <div class="cost-item">
                    <span style="font-weight: 700;">📅 Total Annual Cost</span>
                    <span class="cost-value" style="color: #a78bfa;">₹${annualTotal.toLocaleString()}</span>
                </div>
                <div class="cost-item">
                    <span style="font-weight: 700;">📊 Cost Per Enrollment</span>
                    <span class="cost-value" style="color: #10b981;">₹${costPerEnrollment}</span>
                </div>
            </div>
            
            <div class="cost-recommendations" style="margin-top: 1.5rem; padding: 1rem; background: rgba(139, 92, 246, 0.1); border-radius: 0.5rem; border: 1px solid rgba(139, 92, 246, 0.3);">
                <h5 style="color: #a78bfa; margin-bottom: 0.75rem;">💡 Cost Optimization Suggestions for ${regionName}</h5>
                <ul style="list-style: none; padding: 0; color: #e9d5ff;">
                    ${costPerEnrollment > 200 ? '<li>• ⚠️ Cost per enrollment is high (>₹200). Consider consolidating centers or increasing throughput.</li>' : ''}
                    ${staffCost / monthlyTotal > 0.6 ? '<li>• ⚠️ Staff costs are >60% of total budget. Consider automation or process optimization.</li>' : ''}
                    ${peakStaffRequired > staffRequired * 1.5 ? '<li>• 📈 Peak demand is significantly higher. Consider flexible staffing or extended hours during peak periods.</li>' : '<li>• ✅ Staff utilization is well-balanced between average and peak periods.</li>'}
                    ${costPerEnrollment < 150 ? '<li>• ✅ Cost per enrollment is optimal (<₹150). Current operations are cost-effective.</li>' : ''}
                    ${avgDaily < 1000 ? '<li>• 💡 Low enrollment volume. Consider consolidating with nearby centers to reduce fixed costs.</li>' : ''}
                    ${avgDaily > 10000 ? '<li>• 🚀 High enrollment volume. Consider adding more centers to reduce wait times and improve service quality.</li>' : ''}
                </ul>
            </div>
            
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem; border: 1px solid rgba(16, 185, 129, 0.3);">
                <h5 style="color: #4ade80; margin-bottom: 0.75rem;">📈 Efficiency Metrics</h5>
                <div class="cost-item">
                    <span>Staff Productivity</span>
                    <span class="cost-value" style="color: #4ade80;">${avgDaily > 0 ? (avgDaily / staffRequired).toFixed(0) : 0} enrollments/staff/day</span>
                </div>
                <div class="cost-item">
                    <span>Capacity Utilization</span>
                    <span class="cost-value" style="color: #4ade80;">${stats.capacityUtilization || 0}%</span>
                </div>
                <div class="cost-item">
                    <span>Monthly Revenue Potential (@ ₹100/enrollment)</span>
                    <span class="cost-value" style="color: #4ade80;">₹${(avgDaily * 30 * 100).toLocaleString()}</span>
                </div>
            </div>
        </div>
    `;
}
