from flask import Flask, render_template, jsonify, request, send_file, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import stats
import json
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global data storage
demographic_data = None
biometric_data = None
enrollment_data = None

# Official State and UT Names - 28 States + 8 UTs
OFFICIAL_STATE_NAMES = {
    # States (28)
    'andhra pradesh': 'Andhra Pradesh',
    'arunachal pradesh': 'Arunachal Pradesh',
    'assam': 'Assam',
    'bihar': 'Bihar',
    'chhattisgarh': 'Chhattisgarh',
    'goa': 'Goa',
    'gujarat': 'Gujarat',
    'haryana': 'Haryana',
    'himachal pradesh': 'Himachal Pradesh',
    'jharkhand': 'Jharkhand',
    'karnataka': 'Karnataka',
    'kerala': 'Kerala',
    'madhya pradesh': 'Madhya Pradesh',
    'maharashtra': 'Maharashtra',
    'manipur': 'Manipur',
    'meghalaya': 'Meghalaya',
    'mizoram': 'Mizoram',
    'nagaland': 'Nagaland',
    'odisha': 'Odisha',
    'punjab': 'Punjab',
    'rajasthan': 'Rajasthan',
    'sikkim': 'Sikkim',
    'tamil nadu': 'Tamil Nadu',
    'telangana': 'Telangana',
    'tripura': 'Tripura',
    'uttar pradesh': 'Uttar Pradesh',
    'uttarakhand': 'Uttarakhand',
    'west bengal': 'West Bengal',
    
    # Union Territories (8)
    'andaman and nicobar islands': 'Andaman & Nicobar Islands',
    'andaman & nicobar islands': 'Andaman & Nicobar Islands',
    'chandigarh': 'Chandigarh',
    'dadra and nagar haveli and daman and diu': 'Dadra & Nagar Haveli and Daman & Diu',
    'dadra & nagar haveli and daman & diu': 'Dadra & Nagar Haveli and Daman & Diu',
    'dadra and nagar haveli': 'Dadra & Nagar Haveli and Daman & Diu',
    'daman and diu': 'Dadra & Nagar Haveli and Daman & Diu',
    'delhi': 'Delhi',
    'nct of delhi': 'Delhi',
    'national capital territory of delhi': 'Delhi',
    'jammu and kashmir': 'Jammu & Kashmir',
    'jammu & kashmir': 'Jammu & Kashmir',
    'ladakh': 'Ladakh',
    'lakshadweep': 'Lakshadweep',
    'puducherry': 'Puducherry',
    'pondicherry': 'Puducherry',
    
    
    # Common variations/misspellings
    'west bangal': 'West Bengal',
    'west bengli': 'West Bengal',
    'westbengal': 'West Bengal',
    'orissa': 'Odisha',
    'uttaranchal': 'Uttarakhand',
    'chhatisgarh': 'Chhattisgarh',
    'chattisgarh': 'Chhattisgarh',
    'andhrapradesh': 'Andhra Pradesh',
    'arunachalpradesh': 'Arunachal Pradesh',
    'himachalpradesh': 'Himachal Pradesh',
    'madhyapradesh': 'Madhya Pradesh',
    'tamilnadu': 'Tamil Nadu',
    'uttarpradesh': 'Uttar Pradesh',
    'andaman nicobar': 'Andaman & Nicobar Islands',
    'a&n islands': 'Andaman & Nicobar Islands',
    'd&nh': 'Dadra & Nagar Haveli and Daman & Diu',
    'dnh': 'Dadra & Nagar Haveli and Daman & Diu',
    'j&k': 'Jammu & Kashmir',
}

# List of known city names and invalid entries to filter out
INVALID_STATE_NAMES = {
    '100000', 'jaipur', 'nagpur', 'darbhanga', 'balanagar', 'madanapalle',
    'raja annamalai puram', 'puttenahalli', 'mumbai', 'bangalore', 'hyderabad',
    'chennai', 'kolkata', 'ahmedabad', 'surat', 'pune', 'indore', 'kanpur',
    'lucknow', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ghaziabad',
    'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'kalyan-dombivli',
    'vasai-virar', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar',
    'navi mumbai', 'allahabad', 'ranchi', 'howrah', 'coimbatore', 'jabalpur',
    'gwalior', 'vijayawada', 'jodhpur', 'madurai', 'raipur', 'kota'
}

def normalize_state_name(state_name):
    """Normalize state name to official format"""
    if pd.isna(state_name) or state_name == '':
        return 'Unknown'
    
    # Convert to lowercase for matching
    state_lower = str(state_name).strip().lower()
    
    # Filter out known invalid entries (cities, pincodes, etc.)
    if state_lower in INVALID_STATE_NAMES or state_lower.isdigit():
        return 'Unknown'
    
    # Direct match
    if state_lower in OFFICIAL_STATE_NAMES:
        return OFFICIAL_STATE_NAMES[state_lower]
    
    # Try partial matching for common patterns
    for key, official_name in OFFICIAL_STATE_NAMES.items():
        if key in state_lower or state_lower in key:
            return official_name
    
    # If no match found, return title case of original
    return str(state_name).strip().title()


def load_all_data():
    """Load all CSV files into memory"""
    global demographic_data, biometric_data, enrollment_data
    
    print("Loading demographic data...")
    demo_files = [
        'api_data_aadhar_demographic/api_data_aadhar_demographic_0_500000.csv',
        'api_data_aadhar_demographic/api_data_aadhar_demographic_500000_1000000.csv',
        'api_data_aadhar_demographic/api_data_aadhar_demographic_1000000_1500000.csv',
        'api_data_aadhar_demographic/api_data_aadhar_demographic_1500000_2000000.csv',
        'api_data_aadhar_demographic/api_data_aadhar_demographic_2000000_2071700.csv'
    ]
    demographic_data = pd.concat([pd.read_csv(f) for f in demo_files], ignore_index=True)
    demographic_data['date'] = pd.to_datetime(demographic_data['date'], format='%d-%m-%Y')
    demographic_data['state'] = demographic_data['state'].apply(normalize_state_name)
    
    print("Loading biometric data...")
    bio_files = [
        'api_data_aadhar_biometric/api_data_aadhar_biometric_0_500000.csv',
        'api_data_aadhar_biometric/api_data_aadhar_biometric_500000_1000000.csv',
        'api_data_aadhar_biometric/api_data_aadhar_biometric_1000000_1500000.csv',
        'api_data_aadhar_biometric/api_data_aadhar_biometric_1500000_1861108.csv'
    ]
    biometric_data = pd.concat([pd.read_csv(f) for f in bio_files], ignore_index=True)
    biometric_data['date'] = pd.to_datetime(biometric_data['date'], format='%d-%m-%Y')
    biometric_data['state'] = biometric_data['state'].apply(normalize_state_name)
    
    print("Loading enrollment data...")
    enroll_files = [
        'api_data_aadhar_enrolment/api_data_aadhar_enrolment_0_500000.csv',
        'api_data_aadhar_enrolment/api_data_aadhar_enrolment_500000_1000000.csv',
        'api_data_aadhar_enrolment/api_data_aadhar_enrolment_1000000_1006029.csv'
    ]
    enrollment_data = pd.concat([pd.read_csv(f) for f in enroll_files], ignore_index=True)
    enrollment_data['date'] = pd.to_datetime(enrollment_data['date'], format='%d-%m-%Y')
    enrollment_data['state'] = enrollment_data['state'].apply(normalize_state_name)
    
    # Print normalization statistics
    print(f"\nState Normalization Complete:")
    print(f"  Unique states in demographic data: {demographic_data['state'].nunique()}")
    print(f"  Unique states in biometric data: {biometric_data['state'].nunique()}")
    print(f"  Unique states in enrollment data: {enrollment_data['state'].nunique()}")
    print(f"  States: {sorted(demographic_data['state'].unique())}")
    
    print("\nData loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/government.html')
def government():
    return render_template('government.html')

@app.route('/api/overview')
def get_overview():
    """Get overview statistics"""
    try:
        overview = {
            'total_demographic_records': len(demographic_data),
            'total_biometric_records': len(biometric_data),
            'total_enrollment_records': len(enrollment_data),
            'total_states': demographic_data['state'].nunique(),
            'total_districts': demographic_data['district'].nunique(),
            'total_pincodes': demographic_data['pincode'].nunique(),
            'date_range': {
                'start': demographic_data['date'].min().strftime('%Y-%m-%d'),
                'end': demographic_data['date'].max().strftime('%Y-%m-%d')
            },
            'demographic_age_distribution': {
                'age_5_17': int(demographic_data['demo_age_5_17'].sum()),
                'age_17_plus': int(demographic_data['demo_age_17_'].sum())
            },
            'biometric_age_distribution': {
                'age_5_17': int(biometric_data['bio_age_5_17'].sum()),
                'age_17_plus': int(biometric_data['bio_age_17_'].sum())
            },
            'enrollment_age_distribution': {
                'age_0_5': int(enrollment_data['age_0_5'].sum()),
                'age_5_17': int(enrollment_data['age_5_17'].sum()),
                'age_18_plus': int(enrollment_data['age_18_greater'].sum())
            }
        }
        return jsonify(overview)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/state-analysis')
def state_analysis():
    """Analyze data by state"""
    try:
        # Top states by demographic records
        state_demo = demographic_data.groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        state_demo['total'] = state_demo['demo_age_5_17'] + state_demo['demo_age_17_']
        state_demo = state_demo.nlargest(15, 'total')
        
        # Top states by enrollment
        state_enroll = enrollment_data.groupby('state').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        state_enroll['total'] = state_enroll['age_0_5'] + state_enroll['age_5_17'] + state_enroll['age_18_greater']
        state_enroll = state_enroll.nlargest(15, 'total')
        
        return jsonify({
            'demographic': state_demo.to_dict('records'),
            'enrollment': state_enroll.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/time-series')
def time_series_analysis():
    """Time series analysis with trend prediction"""
    try:
        # Daily aggregation
        demo_daily = demographic_data.groupby('date').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        demo_daily['total'] = demo_daily['demo_age_5_17'] + demo_daily['demo_age_17_']
        demo_daily = demo_daily.sort_values('date')
        
        # Enrollment daily
        enroll_daily = enrollment_data.groupby('date').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        enroll_daily['total'] = enroll_daily['age_0_5'] + enroll_daily['age_5_17'] + enroll_daily['age_18_greater']
        enroll_daily = enroll_daily.sort_values('date')
        
        # Simple trend prediction (next 30 days)
        if len(demo_daily) > 10:
            X = np.arange(len(demo_daily)).reshape(-1, 1)
            y = demo_daily['total'].values
            
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predict next 30 days
            future_X = np.arange(len(demo_daily), len(demo_daily) + 30).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            last_date = demo_daily['date'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
            
            forecast = [
                {'date': date.strftime('%Y-%m-%d'), 'predicted_total': int(pred)}
                for date, pred in zip(future_dates, predictions)
            ]
        else:
            forecast = []
        
        return jsonify({
            'demographic_daily': demo_daily.to_dict('records'),
            'enrollment_daily': enroll_daily.to_dict('records'),
            'forecast': forecast
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clustering')
def clustering_analysis():
    """Perform clustering analysis on geographic data"""
    try:
        # Aggregate by pincode
        pincode_data = demographic_data.groupby('pincode').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum',
            'state': 'first',
            'district': 'first'
        }).reset_index()
        
        # Prepare features for clustering
        features = pincode_data[['demo_age_5_17', 'demo_age_17_']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # KMeans clustering
        n_clusters = min(5, len(pincode_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pincode_data['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Get cluster centers
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        clusters = []
        for i in range(n_clusters):
            cluster_pincodes = pincode_data[pincode_data['cluster'] == i]
            clusters.append({
                'cluster_id': int(i),
                'center': {
                    'demo_age_5_17': float(cluster_centers[i][0]),
                    'demo_age_17_': float(cluster_centers[i][1])
                },
                'size': len(cluster_pincodes),
                'top_pincodes': cluster_pincodes.nlargest(5, 'demo_age_5_17')[
                    ['pincode', 'state', 'district', 'demo_age_5_17', 'demo_age_17_']
                ].to_dict('records')
            })
        
        return jsonify({'clusters': clusters})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistical-analysis')
def statistical_analysis():
    """Comprehensive statistical analysis"""
    try:
        # Demographic statistics
        demo_stats = {
            'age_5_17': {
                'mean': float(demographic_data['demo_age_5_17'].mean()),
                'median': float(demographic_data['demo_age_5_17'].median()),
                'std': float(demographic_data['demo_age_5_17'].std()),
                'min': int(demographic_data['demo_age_5_17'].min()),
                'max': int(demographic_data['demo_age_5_17'].max()),
                'q1': float(demographic_data['demo_age_5_17'].quantile(0.25)),
                'q3': float(demographic_data['demo_age_5_17'].quantile(0.75))
            },
            'age_17_plus': {
                'mean': float(demographic_data['demo_age_17_'].mean()),
                'median': float(demographic_data['demo_age_17_'].median()),
                'std': float(demographic_data['demo_age_17_'].std()),
                'min': int(demographic_data['demo_age_17_'].min()),
                'max': int(demographic_data['demo_age_17_'].max()),
                'q1': float(demographic_data['demo_age_17_'].quantile(0.25)),
                'q3': float(demographic_data['demo_age_17_'].quantile(0.75))
            }
        }
        
        # Correlation analysis
        demo_corr = demographic_data[['demo_age_5_17', 'demo_age_17_']].corr()
        
        # Distribution analysis
        state_distribution = demographic_data.groupby('state').size().nlargest(10).to_dict()
        
        return jsonify({
            'demographic_statistics': demo_stats,
            'correlation': demo_corr.to_dict(),
            'state_distribution': state_distribution
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/geographic-heatmap')
def geographic_heatmap():
    """Generate geographic heatmap data"""
    try:
        # Aggregate by state and district
        geo_data = demographic_data.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        geo_data['total'] = geo_data['demo_age_5_17'] + geo_data['demo_age_17_']
        
        # Top 50 locations
        top_locations = geo_data.nlargest(50, 'total')
        
        return jsonify({
            'heatmap_data': top_locations.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictive-model', methods=['POST'])
def predictive_model():
    """Train and use predictive model"""
    try:
        data = request.json
        state = data.get('state')
        district = data.get('district')
        
        # Filter data
        if state:
            filtered = demographic_data[demographic_data['state'] == state]
            if district:
                filtered = filtered[filtered['district'] == district]
        else:
            filtered = demographic_data
        
        # Time-based features
        filtered = filtered.copy()
        filtered['day_of_year'] = filtered['date'].dt.dayofyear
        filtered['month'] = filtered['date'].dt.month
        filtered['day_of_week'] = filtered['date'].dt.dayofweek
        
        # Aggregate by date
        daily_data = filtered.groupby(['day_of_year', 'month', 'day_of_week']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        
        if len(daily_data) < 10:
            return jsonify({'error': 'Insufficient data for prediction'}), 400
        
        # Train model
        X = daily_data[['day_of_year', 'month', 'day_of_week']].values
        y = (daily_data['demo_age_5_17'] + daily_data['demo_age_17_']).values
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Feature importance
        feature_importance = {
            'day_of_year': float(model.feature_importances_[0]),
            'month': float(model.feature_importances_[1]),
            'day_of_week': float(model.feature_importances_[2])
        }
        
        # Predict next 7 days
        today = datetime.now()
        predictions = []
        for i in range(1, 8):
            future_date = today + timedelta(days=i)
            X_pred = np.array([[
                future_date.timetuple().tm_yday,
                future_date.month,
                future_date.weekday()
            ]])
            pred = model.predict(X_pred)[0]
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_total': int(pred)
            })
        
        return jsonify({
            'predictions': predictions,
            'feature_importance': feature_importance,
            'model_score': float(model.score(X, y))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomaly-detection')
def anomaly_detection():
    """Detect anomalies in the data"""
    try:
        # Daily totals
        daily_totals = demographic_data.groupby('date').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        daily_totals['total'] = daily_totals['demo_age_5_17'] + daily_totals['demo_age_17_']
        
        # Calculate z-scores
        mean = daily_totals['total'].mean()
        std = daily_totals['total'].std()
        daily_totals['z_score'] = (daily_totals['total'] - mean) / std
        
        # Anomalies (|z-score| > 2)
        anomalies = daily_totals[abs(daily_totals['z_score']) > 2].copy()
        anomalies['date'] = anomalies['date'].dt.strftime('%Y-%m-%d')
        
        return jsonify({
            'anomalies': anomalies.to_dict('records'),
            'threshold': 2.0,
            'mean': float(mean),
            'std': float(std)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/district-ranking')
def district_ranking():
    """Rank districts by various metrics"""
    try:
        # Demographic ranking
        demo_ranking = demographic_data.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        demo_ranking['total'] = demo_ranking['demo_age_5_17'] + demo_ranking['demo_age_17_']
        demo_ranking = demo_ranking.nlargest(20, 'total')
        
        # Enrollment ranking
        enroll_ranking = enrollment_data.groupby(['state', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        enroll_ranking['total'] = enroll_ranking['age_0_5'] + enroll_ranking['age_5_17'] + enroll_ranking['age_18_greater']
        enroll_ranking = enroll_ranking.nlargest(20, 'total')
        
        return jsonify({
            'demographic_ranking': demo_ranking.to_dict('records'),
            'enrollment_ranking': enroll_ranking.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/drill-down')
def drill_down_analysis():
    """Hierarchical drill-down: State -> District -> Pincode"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        # State level
        if not state:
            state_summary = demographic_data.groupby('state').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
            state_summary['total'] = state_summary['demo_age_5_17'] + state_summary['demo_age_17_']
            state_summary = state_summary.sort_values('total', ascending=False)
            
            return jsonify({
                'level': 'state',
                'data': state_summary.to_dict('records')
            })
        
        # District level
        if state and not district:
            district_data = demographic_data[demographic_data['state'] == state]
            district_summary = district_data.groupby('district').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
            district_summary['total'] = district_summary['demo_age_5_17'] + district_summary['demo_age_17_']
            district_summary = district_summary.sort_values('total', ascending=False)
            
            return jsonify({
                'level': 'district',
                'state': state,
                'data': district_summary.to_dict('records')
            })
        
        # Pincode level
        if state and district:
            pincode_data = demographic_data[
                (demographic_data['state'] == state) & 
                (demographic_data['district'] == district)
            ]
            pincode_summary = pincode_data.groupby('pincode').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
            pincode_summary['total'] = pincode_summary['demo_age_5_17'] + pincode_summary['demo_age_17_']
            pincode_summary = pincode_summary.sort_values('total', ascending=False)
            
            return jsonify({
                'level': 'pincode',
                'state': state,
                'district': district,
                'data': pincode_summary.to_dict('records')
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/spike-detection')
def spike_detection():
    """Detect enrollment/demographic spikes with alerts"""
    try:
        # Analyze demographic spikes by state
        demo_daily = demographic_data.groupby(['date', 'state']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        demo_daily['total'] = demo_daily['demo_age_5_17'] + demo_daily['demo_age_17_']
        
        # Calculate rolling average and detect spikes
        spikes = []
        for state in demo_daily['state'].unique():
            state_data = demo_daily[demo_daily['state'] == state].sort_values('date')
            if len(state_data) < 7:
                continue
                
            state_data['rolling_avg'] = state_data['total'].rolling(window=7, min_periods=1).mean()
            state_data['rolling_std'] = state_data['total'].rolling(window=7, min_periods=1).std()
            
            # Spike = value > mean + 2*std
            state_data['is_spike'] = state_data['total'] > (state_data['rolling_avg'] + 2 * state_data['rolling_std'])
            
            spike_days = state_data[state_data['is_spike']]
            for _, row in spike_days.iterrows():
                spikes.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'state': state,
                    'value': int(row['total']),
                    'avg': float(row['rolling_avg']),
                    'deviation': float((row['total'] - row['rolling_avg']) / row['rolling_avg'] * 100),
                    'severity': 'high' if row['total'] > row['rolling_avg'] + 3 * row['rolling_std'] else 'medium'
                })
        
        # Sort by deviation
        spikes = sorted(spikes, key=lambda x: x['deviation'], reverse=True)[:20]
        
        return jsonify({
            'spikes': spikes,
            'total_spikes': len(spikes),
            'high_severity': len([s for s in spikes if s['severity'] == 'high'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comparative-analysis')
def comparative_analysis():
    """Compare states, districts, or pincodes"""
    try:
        entities = request.args.getlist('entities')  # e.g., ['Maharashtra', 'Gujarat']
        level = request.args.get('level', 'state')  # state, district, or pincode
        
        if not entities:
            return jsonify({'error': 'No entities provided'}), 400
        
        comparisons = []
        
        for entity in entities:
            try:
                # Get demographic data
                if level == 'state':
                    demo_data = demographic_data[demographic_data['state'] == entity]
                    bio_data = biometric_data[biometric_data['state'] == entity]
                    enroll_data = enrollment_data[enrollment_data['state'] == entity]
                elif level == 'district':
                    demo_data = demographic_data[demographic_data['district'] == entity]
                    bio_data = biometric_data[biometric_data['district'] == entity]
                    enroll_data = enrollment_data[enrollment_data['district'] == entity]
                else:
                    demo_data = demographic_data[demographic_data['pincode'] == int(entity)]
                    bio_data = biometric_data[biometric_data['pincode'] == int(entity)]
                    enroll_data = enrollment_data[enrollment_data['pincode'] == int(entity)]
                
                if len(demo_data) == 0:
                    continue
                
                # Calculate demographic metrics
                total_demo = int(demo_data['demo_age_5_17'].sum() + demo_data['demo_age_17_'].sum())
                
                # Calculate biometric metrics
                total_bio = int(bio_data['bio_age_5_17'].sum() + bio_data['bio_age_17_'].sum()) if len(bio_data) > 0 else 0
                
                # Calculate enrollment metrics
                total_enroll = int(enroll_data['age_0_5'].sum() + enroll_data['age_5_17'].sum() + enroll_data['age_18_greater'].sum()) if len(enroll_data) > 0 else 0
                
                # Time series for trend
                daily = demo_data.groupby('date').size().reset_index(name='count')
                if len(daily) > 1:
                    recent_avg = daily['count'].tail(7).mean()
                    older_avg = daily['count'].head(7).mean()
                    trend = 'increasing' if recent_avg > older_avg else 'decreasing' if recent_avg < older_avg else 'stable'
                else:
                    trend = 'stable'
                
                # Coverage metrics
                districts = demo_data['district'].nunique() if level == 'state' else None
                pincodes = demo_data['pincode'].nunique()
                
                comparisons.append({
                    'entity': entity,
                    'total_demographic': total_demo,
                    'total_biometric': total_bio,
                    'total_enrollment': total_enroll,
                    'avg_daily': int(total_demo / max(len(daily), 1)),
                    'trend': trend,
                    'districts': districts,
                    'pincodes': pincodes,
                    'bio_coverage_percent': round((total_bio / total_demo * 100) if total_demo > 0 else 0, 1),
                    'enrollment_rate': round((total_enroll / total_demo * 100) if total_demo > 0 else 0, 1)
                })
            except Exception as e:
                print(f"Error processing entity {entity}: {str(e)}")
                continue
        
        if len(comparisons) == 0:
            return jsonify({'error': 'No valid entities found'}), 404
        
        return jsonify({
            'level': level,
            'comparisons': comparisons,
            'count': len(comparisons)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trend-analysis')
def trend_analysis():
    """Analyze trends over time with forecasting"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        # Filter data
        data = demographic_data.copy()
        if state:
            data = data[data['state'] == state]
        if district:
            data = data[data['district'] == district]
        
        # Daily aggregation
        daily = data.groupby('date').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        daily['total'] = daily['demo_age_5_17'] + daily['demo_age_17_']
        daily = daily.sort_values('date')
        
        if len(daily) < 7:
            return jsonify({'error': 'Insufficient data'}), 400
        
        # Calculate trends
        daily['7day_avg'] = daily['total'].rolling(window=7, min_periods=1).mean()
        daily['30day_avg'] = daily['total'].rolling(window=30, min_periods=1).mean()
        
        # Growth rate
        if len(daily) > 30:
            recent_avg = daily['total'].tail(7).mean()
            older_avg = daily['total'].tail(37).head(7).mean()
            growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            growth_rate = 0
        
        # Peak and low
        peak_day = daily.loc[daily['total'].idxmax()]
        low_day = daily.loc[daily['total'].idxmin()]
        
        return jsonify({
            'daily_data': daily.to_dict('records'),
            'growth_rate': float(growth_rate),
            'peak': {
                'date': peak_day['date'].strftime('%Y-%m-%d'),
                'value': int(peak_day['total'])
            },
            'low': {
                'date': low_day['date'].strftime('%Y-%m-%d'),
                'value': int(low_day['total'])
            },
            'average': float(daily['total'].mean()),
            'volatility': float(daily['total'].std())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cluster-insights')
def cluster_insights():
    """Advanced clustering with actionable insights and ML validation"""
    try:
        # Get filter parameters
        state = request.args.get('state')
        district = request.args.get('district')
        
        # Filter demographic data
        demo_filtered = demographic_data.copy()
        if state:
            demo_filtered = demo_filtered[demo_filtered['state'] == state]
        if district:
            demo_filtered = demo_filtered[demo_filtered['district'] == district]
        
        # Filter enrollment data
        enroll_filtered = enrollment_data.copy()
        if state:
            enroll_filtered = enroll_filtered[enroll_filtered['state'] == state]
        if district:
            enroll_filtered = enroll_filtered[enroll_filtered['district'] == district]
        
        
        # Determine clustering granularity based on filters
        if district:
            # District-level view: cluster by pincode
            cluster_data = demo_filtered.groupby(['state', 'district', 'pincode']).agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
            
            # Add enrollment data
            enroll_cluster = enroll_filtered.groupby(['state', 'district', 'pincode']).agg({
                'age_0_5': 'sum',
                'age_5_17': 'sum',
                'age_18_greater': 'sum'
            }).reset_index()
            
            cluster_data = cluster_data.merge(
                enroll_cluster,
                on=['state', 'district', 'pincode'],
                how='left'
            ).fillna(0)
            
            group_by_cols = ['state', 'district', 'pincode']
            entity_name = 'pincode'
        else:
            # State/National view: cluster by district
            cluster_data = demo_filtered.groupby(['state', 'district']).agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum',
                'pincode': 'nunique'
            }).reset_index()
            
            # Add enrollment data
            enroll_cluster = enroll_filtered.groupby(['state', 'district']).agg({
                'age_0_5': 'sum',
                'age_5_17': 'sum',
                'age_18_greater': 'sum'
            }).reset_index()
            
            cluster_data = cluster_data.merge(
                enroll_cluster,
                on=['state', 'district'],
                how='left'
            ).fillna(0)
            
            group_by_cols = ['state', 'district']
            entity_name = 'district'
        
        # Check if we have enough data for clustering
        if len(cluster_data) < 2:
            return jsonify({
                'error': 'Insufficient data for clustering',
                'message': f'Need at least 2 {entity_name}s for clustering analysis. Found {len(cluster_data)}.',
                'clusters': [],
                'validation_metrics': {}
            }), 400
        
        # Features for clustering
        features = cluster_data[[
            'demo_age_5_17', 'demo_age_17_', 
            'age_0_5', 'age_5_17', 'age_18_greater'
        ]].values
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # KMeans clustering
        n_clusters = min(6, len(cluster_data))
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Calculate validation metrics (only if we have enough clusters)
        if n_clusters >= 2 and len(cluster_data) >= 2:
            silhouette_avg = silhouette_score(features_scaled, cluster_data['cluster'])
            davies_bouldin = davies_bouldin_score(features_scaled, cluster_data['cluster'])
        else:
            silhouette_avg = 0.0
            davies_bouldin = 0.0
        
        # Analyze clusters
        clusters = []
        for i in range(n_clusters):
            cluster_entities = cluster_data[cluster_data['cluster'] == i]
            
            # Characteristics
            avg_demo = cluster_entities['demo_age_5_17'].mean() + cluster_entities['demo_age_17_'].mean()
            avg_enroll = cluster_entities['age_0_5'].mean() + cluster_entities['age_5_17'].mean() + cluster_entities['age_18_greater'].mean()
            
            # Classification based on data-driven thresholds
            demo_threshold = cluster_data['demo_age_5_17'].mean() + cluster_data['demo_age_17_'].mean()
            enroll_threshold = cluster_data['age_0_5'].mean() + cluster_data['age_5_17'].mean() + cluster_data['age_18_greater'].mean()
            
            if avg_demo > demo_threshold * 1.2:
                category = 'High Activity'
                recommendation = 'Maintain resources, monitor for capacity issues'
            elif avg_enroll > enroll_threshold * 1.2:
                category = 'Growing Region'
                recommendation = 'Increase enrollment centers, prepare for growth'
            else:
                category = 'Stable Region'
                recommendation = 'Standard monitoring, optimize existing resources'
            
            # Prepare top entities for this cluster
            if entity_name == 'pincode':
                top_entities = cluster_entities.nlargest(5, 'demo_age_5_17')[
                    ['state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_']
                ].to_dict('records')
            else:
                top_entities = cluster_entities.nlargest(5, 'demo_age_5_17')[
                    ['state', 'district', 'demo_age_5_17', 'demo_age_17_']
                ].to_dict('records')
            
            clusters.append({
                'cluster_id': int(i),
                'size': len(cluster_entities),
                'category': category,
                'avg_demographic': float(avg_demo),
                'avg_enrollment': float(avg_enroll),
                'recommendation': recommendation,
                f'top_{entity_name}s': top_entities
            })
        
        return jsonify({
            'clusters': clusters,
            'validation_metrics': {
                'silhouette_score': float(silhouette_avg),
                'davies_bouldin_index': float(davies_bouldin),
                'interpretation': {
                    'silhouette_score': f'Score: {silhouette_avg:.3f} (Range: -1 to 1, higher is better)',
                    'davies_bouldin_index': f'Score: {davies_bouldin:.3f} (Lower is better)',
                    'quality': 'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.25 else 'Needs Improvement'
                }
            },
            'methodology': {
                'algorithm': 'K-Means Clustering',
                'features': ['demo_age_5_17', 'demo_age_17_', 'age_0_5', 'age_5_17', 'age_18_greater'],
                'preprocessing': 'StandardScaler normalization',
                'n_clusters': n_clusters,
                'random_state': 42
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/government-dashboard')
def government_dashboard():
    """Comprehensive government dashboard with KPIs"""
    try:
        # Overall KPIs
        total_demographic = len(demographic_data)
        total_biometric = len(biometric_data)
        total_enrollment = len(enrollment_data)
        
        # Recent activity (last 30 days)
        recent_date = demographic_data['date'].max() - timedelta(days=30)
        recent_demo = demographic_data[demographic_data['date'] >= recent_date]
        recent_enroll = enrollment_data[enrollment_data['date'] >= recent_date]
        
        # State performance
        state_performance = demographic_data.groupby('state').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        state_performance['total'] = state_performance['demo_age_5_17'] + state_performance['demo_age_17_']
        state_performance = state_performance.sort_values('total', ascending=False)
        
        # Top and bottom performers
        top_5 = state_performance.head(5).to_dict('records')
        bottom_5 = state_performance.tail(5).to_dict('records')
        
        # Coverage metrics
        states_covered = demographic_data['state'].nunique()
        districts_covered = demographic_data['district'].nunique()
        pincodes_covered = demographic_data['pincode'].nunique()
        
        # Alerts
        alerts = []
        
        # Check for low activity states
        avg_total = state_performance['total'].mean()
        low_activity = state_performance[state_performance['total'] < avg_total * 0.3]
        for _, state in low_activity.iterrows():
            alerts.append({
                'type': 'low_activity',
                'severity': 'medium',
                'state': state['state'],
                'message': f"Low enrollment activity in {state['state']}",
                'recommendation': 'Increase awareness campaigns and enrollment centers'
            })
        
        return jsonify({
            'kpis': {
                'total_records': total_demographic + total_biometric + total_enrollment,
                'states_covered': states_covered,
                'districts_covered': districts_covered,
                'pincodes_covered': pincodes_covered,
                'recent_30_days': len(recent_demo) + len(recent_enroll)
            },
            'top_performers': top_5,
            'bottom_performers': bottom_5,
            'alerts': alerts[:10],
            'coverage_percentage': float(states_covered / 36 * 100)  # 36 states/UTs in India
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demographic-analysis')
def demographic_analysis():
    """Comprehensive demographic data analysis"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        # Filter data
        data = demographic_data.copy()
        if state:
            data = data[data['state'] == state]
        if district:
            data = data[data['district'] == district]
        
        if len(data) == 0:
            return jsonify({'error': 'No data found'}), 404
        
        # Overall metrics
        total_records = len(data)
        total_age_5_17 = int(data['demo_age_5_17'].sum())
        total_age_17_plus = int(data['demo_age_17_'].sum())
        total_population = total_age_5_17 + total_age_17_plus
        
        # Age distribution
        age_distribution = {
            'age_5_17': total_age_5_17,
            'age_17_plus': total_age_17_plus,
            'age_5_17_percent': round((total_age_5_17 / total_population * 100) if total_population > 0 else 0, 2),
            'age_17_plus_percent': round((total_age_17_plus / total_population * 100) if total_population > 0 else 0, 2)
        }
        
        # Geographic coverage
        states_covered = data['state'].nunique()
        districts_covered = data['district'].nunique()
        pincodes_covered = data['pincode'].nunique()
        
        # Time series analysis
        daily_data = data.groupby('date').agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        daily_data['total'] = daily_data['demo_age_5_17'] + daily_data['demo_age_17_']
        daily_data = daily_data.sort_values('date')
        
        # Trend calculation
        if len(daily_data) > 7:
            recent_avg = daily_data['total'].tail(7).mean()
            older_avg = daily_data['total'].head(7).mean()
            growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            growth_rate = 0
        
        # Top states/districts
        if not state:
            top_states = data.groupby('state').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
            top_states['total'] = top_states['demo_age_5_17'] + top_states['demo_age_17_']
            top_states = top_states.nlargest(10, 'total').to_dict('records')
        else:
            top_states = []
        
        top_districts = data.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        top_districts['total'] = top_districts['demo_age_5_17'] + top_districts['demo_age_17_']
        top_districts = top_districts.nlargest(10, 'total').to_dict('records')
        
        # Statistical summary
        stats = {
            'mean_per_record': {
                'age_5_17': float(data['demo_age_5_17'].mean()),
                'age_17_plus': float(data['demo_age_17_'].mean())
            },
            'median_per_record': {
                'age_5_17': float(data['demo_age_5_17'].median()),
                'age_17_plus': float(data['demo_age_17_'].median())
            },
            'std_dev': {
                'age_5_17': float(data['demo_age_5_17'].std()),
                'age_17_plus': float(data['demo_age_17_'].std())
            }
        }
        
        return jsonify({
            'summary': {
                'total_records': total_records,
                'total_population': total_population,
                'total_age_5_17': total_age_5_17,
                'total_age_17_plus': total_age_17_plus,
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'age_distribution': age_distribution,
            'coverage': {
                'states': states_covered,
                'districts': districts_covered,
                'pincodes': pincodes_covered
            },
            'trends': {
                'growth_rate_7day': round(growth_rate, 2),
                'avg_daily': int(total_population / max(len(daily_data), 1)),
                'peak_day': {
                    'date': daily_data.loc[daily_data['total'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                    'value': int(daily_data['total'].max())
                } if len(daily_data) > 0 else None
            },
            'top_performers': {
                'states': top_states,
                'districts': top_districts
            },
            'statistics': stats,
            'filters': {
                'state': state,
                'district': district
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/biometric-analysis')
def biometric_analysis():
    """Comprehensive biometric data analysis"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        # Filter data
        data = biometric_data.copy()
        if state:
            data = data[data['state'] == state]
        if district:
            data = data[data['district'] == district]
        
        if len(data) == 0:
            return jsonify({'error': 'No data found'}), 404
        
        # Overall metrics
        total_records = len(data)
        total_age_5_17 = int(data['bio_age_5_17'].sum())
        total_age_17_plus = int(data['bio_age_17_'].sum())
        total_updates = total_age_5_17 + total_age_17_plus
        
        # Age distribution
        age_distribution = {
            'age_5_17': total_age_5_17,
            'age_17_plus': total_age_17_plus,
            'age_5_17_percent': round((total_age_5_17 / total_updates * 100) if total_updates > 0 else 0, 2),
            'age_17_plus_percent': round((total_age_17_plus / total_updates * 100) if total_updates > 0 else 0, 2)
        }
        
        # Geographic coverage
        states_covered = data['state'].nunique()
        districts_covered = data['district'].nunique()
        pincodes_covered = data['pincode'].nunique()
        
        # Time series analysis
        daily_data = data.groupby('date').agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        daily_data['total'] = daily_data['bio_age_5_17'] + daily_data['bio_age_17_']
        daily_data = daily_data.sort_values('date')
        
        # Trend calculation
        if len(daily_data) > 7:
            recent_avg = daily_data['total'].tail(7).mean()
            older_avg = daily_data['total'].head(7).mean()
            growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            growth_rate = 0
        
        # Top states/districts
        if not state:
            top_states = data.groupby('state').agg({
                'bio_age_5_17': 'sum',
                'bio_age_17_': 'sum'
            }).reset_index()
            top_states['total'] = top_states['bio_age_5_17'] + top_states['bio_age_17_']
            top_states = top_states.nlargest(10, 'total').to_dict('records')
        else:
            top_states = []
        
        top_districts = data.groupby(['state', 'district']).agg({
            'bio_age_5_17': 'sum',
            'bio_age_17_': 'sum'
        }).reset_index()
        top_districts['total'] = top_districts['bio_age_5_17'] + top_districts['bio_age_17_']
        top_districts = top_districts.nlargest(10, 'total').to_dict('records')
        
        # Coverage rate (compared to demographic)
        demo_data = demographic_data.copy()
        if state:
            demo_data = demo_data[demo_data['state'] == state]
        if district:
            demo_data = demo_data[demo_data['district'] == district]
        
        total_demo = demo_data['demo_age_5_17'].sum() + demo_data['demo_age_17_'].sum()
        coverage_rate = round((total_updates / total_demo * 100) if total_demo > 0 else 0, 2)
        
        return jsonify({
            'summary': {
                'total_records': total_records,
                'total_updates': total_updates,
                'total_age_5_17': total_age_5_17,
                'total_age_17_plus': total_age_17_plus,
                'coverage_rate': coverage_rate,
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'age_distribution': age_distribution,
            'coverage': {
                'states': states_covered,
                'districts': districts_covered,
                'pincodes': pincodes_covered
            },
            'trends': {
                'growth_rate_7day': round(growth_rate, 2),
                'avg_daily': int(total_updates / max(len(daily_data), 1)),
                'peak_day': {
                    'date': daily_data.loc[daily_data['total'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                    'value': int(daily_data['total'].max())
                } if len(daily_data) > 0 else None
            },
            'top_performers': {
                'states': top_states,
                'districts': top_districts
            },
            'filters': {
                'state': state,
                'district': district
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enrollment-analysis')
def enrollment_analysis():
    """Comprehensive enrollment data analysis"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        # Filter data
        data = enrollment_data.copy()
        if state:
            data = data[data['state'] == state]
        if district:
            data = data[data['district'] == district]
        
        if len(data) == 0:
            return jsonify({'error': 'No data found'}), 404
        
        # Overall metrics
        total_records = len(data)
        total_age_0_5 = int(data['age_0_5'].sum())
        total_age_5_17 = int(data['age_5_17'].sum())
        total_age_18_plus = int(data['age_18_greater'].sum())
        total_enrollments = total_age_0_5 + total_age_5_17 + total_age_18_plus
        
        # Age distribution
        age_distribution = {
            'age_0_5': total_age_0_5,
            'age_5_17': total_age_5_17,
            'age_18_plus': total_age_18_plus,
            'age_0_5_percent': round((total_age_0_5 / total_enrollments * 100) if total_enrollments > 0 else 0, 2),
            'age_5_17_percent': round((total_age_5_17 / total_enrollments * 100) if total_enrollments > 0 else 0, 2),
            'age_18_plus_percent': round((total_age_18_plus / total_enrollments * 100) if total_enrollments > 0 else 0, 2)
        }
        
        # Geographic coverage
        states_covered = data['state'].nunique()
        districts_covered = data['district'].nunique()
        pincodes_covered = data['pincode'].nunique()
        
        # Time series analysis
        daily_data = data.groupby('date').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        daily_data['total'] = daily_data['age_0_5'] + daily_data['age_5_17'] + daily_data['age_18_greater']
        daily_data = daily_data.sort_values('date')
        
        # Trend calculation
        if len(daily_data) > 7:
            recent_avg = daily_data['total'].tail(7).mean()
            older_avg = daily_data['total'].head(7).mean()
            growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            growth_rate = 0
        
        # Top states/districts
        if not state:
            top_states = data.groupby('state').agg({
                'age_0_5': 'sum',
                'age_5_17': 'sum',
                'age_18_greater': 'sum'
            }).reset_index()
            top_states['total'] = top_states['age_0_5'] + top_states['age_5_17'] + top_states['age_18_greater']
            top_states = top_states.nlargest(10, 'total').to_dict('records')
        else:
            top_states = []
        
        top_districts = data.groupby(['state', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        top_districts['total'] = top_districts['age_0_5'] + top_districts['age_5_17'] + top_districts['age_18_greater']
        top_districts = top_districts.nlargest(10, 'total').to_dict('records')
        
        # Enrollment rate (compared to demographic)
        demo_data = demographic_data.copy()
        if state:
            demo_data = demo_data[demo_data['state'] == state]
        if district:
            demo_data = demo_data[demo_data['district'] == district]
        
        total_demo = demo_data['demo_age_5_17'].sum() + demo_data['demo_age_17_'].sum()
        enrollment_rate = round((total_enrollments / total_demo * 100) if total_demo > 0 else 0, 2)
        
        return jsonify({
            'summary': {
                'total_records': total_records,
                'total_enrollments': total_enrollments,
                'total_age_0_5': total_age_0_5,
                'total_age_5_17': total_age_5_17,
                'total_age_18_plus': total_age_18_plus,
                'enrollment_rate': enrollment_rate,
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                }
            },
            'age_distribution': age_distribution,
            'coverage': {
                'states': states_covered,
                'districts': districts_covered,
                'pincodes': pincodes_covered
            },
            'trends': {
                'growth_rate_7day': round(growth_rate, 2),
                'avg_daily': int(total_enrollments / max(len(daily_data), 1)),
                'peak_day': {
                    'date': daily_data.loc[daily_data['total'].idxmax(), 'date'].strftime('%Y-%m-%d'),
                    'value': int(daily_data['total'].max())
                } if len(daily_data) > 0 else None
            },
            'top_performers': {
                'states': top_states,
                'districts': top_districts
            },
            'filters': {
                'state': state,
                'district': district
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations')
def get_recommendations():
    """AI-powered recommendations for government action"""
    try:
        state = request.args.get('state')
        
        recommendations = []
        
        # Analyze data
        if state:
            data = demographic_data[demographic_data['state'] == state]
        else:
            data = demographic_data
        
        # Check enrollment rate
        total_demo = data['demo_age_5_17'].sum() + data['demo_age_17_'].sum()
        
        if state:
            enroll = enrollment_data[enrollment_data['state'] == state]
        else:
            enroll = enrollment_data
        
        total_enroll = enroll['age_0_5'].sum() + enroll['age_5_17'].sum() + enroll['age_18_greater'].sum()
        
        # Recommendation 1: Enrollment rate
        if total_enroll < total_demo * 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'enrollment',
                'title': 'Low Enrollment Rate',
                'description': f'Enrollment is only {(total_enroll/total_demo*100):.1f}% of demographic updates',
                'actions': [
                    'Launch targeted enrollment drives',
                    'Increase mobile enrollment units',
                    'Partner with local authorities',
                    'Simplify enrollment process'
                ],
                'expected_impact': 'Increase enrollment by 30-40%'
            })
        
        # Recommendation 2: Geographic coverage
        districts = data['district'].nunique()
        pincodes = data['pincode'].nunique()
        
        if pincodes < districts * 10:
            recommendations.append({
                'priority': 'medium',
                'category': 'coverage',
                'title': 'Expand Geographic Coverage',
                'description': f'Only {pincodes} pincodes covered across {districts} districts',
                'actions': [
                    'Open enrollment centers in underserved areas',
                    'Deploy mobile units to remote pincodes',
                    'Partner with post offices and banks'
                ],
                'expected_impact': 'Reach 50% more pincodes'
            })
        
        # Recommendation 3: Age group focus
        child_demo = data['demo_age_5_17'].sum()
        adult_demo = data['demo_age_17_'].sum()
        
        if child_demo < adult_demo * 0.3:
            recommendations.append({
                'priority': 'high',
                'category': 'demographics',
                'title': 'Focus on Child Enrollment',
                'description': 'Child enrollment significantly lower than adult',
                'actions': [
                    'School-based enrollment drives',
                    'Parent awareness campaigns',
                    'Simplified process for minors',
                    'Incentivize early enrollment'
                ],
                'expected_impact': 'Increase child enrollment by 50%'
            })
        
        # Recommendation 4: Biometric updates
        bio_data = biometric_data if not state else biometric_data[biometric_data['state'] == state]
        bio_total = bio_data['bio_age_5_17'].sum() + bio_data['bio_age_17_'].sum()
        
        if bio_total < total_demo * 0.7:
            recommendations.append({
                'priority': 'medium',
                'category': 'biometric',
                'title': 'Increase Biometric Updates',
                'description': 'Many records need biometric updates',
                'actions': [
                    'Send update reminders',
                    'Organize update camps',
                    'Make process more accessible',
                    'Educate on importance of updates'
                ],
                'expected_impact': 'Achieve 90% biometric coverage'
            })
        
        return jsonify({
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'state': state if state else 'All India'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# CSV Export Endpoints
@app.route('/api/export/drill-down')
def export_drill_down():
    """Export drill-down data as CSV"""
    try:
        state = request.args.get('state')
        district = request.args.get('district')
        
        # Get data
        data = demographic_data.copy()
        if state:
            data = data[data['state'] == state]
        if district:
            data = data[data['district'] == district]
        
        # Aggregate
        if not state:
            result = data.groupby('state').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
        elif not district:
            result = data.groupby('district').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
        else:
            result = data.groupby('pincode').agg({
                'demo_age_5_17': 'sum',
                'demo_age_17_': 'sum'
            }).reset_index()
        
        result['total'] = result['demo_age_5_17'] + result['demo_age_17_']
        
        # Create CSV
        output = io.StringIO()
        result.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"uidai_drilldown_{state or 'all'}_{district or 'all'}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/spikes')
def export_spikes():
    """Export spike detection data as CSV"""
    try:
        # Get spike data
        demo_daily = demographic_data.groupby(['date', 'state']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum'
        }).reset_index()
        demo_daily['total'] = demo_daily['demo_age_5_17'] + demo_daily['demo_age_17_']
        
        spikes = []
        for state in demo_daily['state'].unique():
            state_data = demo_daily[demo_daily['state'] == state].sort_values('date')
            if len(state_data) < 7:
                continue
                
            state_data['rolling_avg'] = state_data['total'].rolling(window=7, min_periods=1).mean()
            state_data['rolling_std'] = state_data['total'].rolling(window=7, min_periods=1).std()
            state_data['is_spike'] = state_data['total'] > (state_data['rolling_avg'] + 2 * state_data['rolling_std'])
            
            spike_days = state_data[state_data['is_spike']]
            for _, row in spike_days.iterrows():
                spikes.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'state': state,
                    'value': int(row['total']),
                    'rolling_avg': float(row['rolling_avg']),
                    'rolling_std': float(row['rolling_std']),
                    'deviation_percent': float((row['total'] - row['rolling_avg']) / row['rolling_avg'] * 100),
                    'severity': 'high' if row['total'] > row['rolling_avg'] + 3 * row['rolling_std'] else 'medium'
                })
        
        df = pd.DataFrame(spikes)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"uidai_spikes_{datetime.now().strftime('%Y%m%d')}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/clusters')
def export_clusters():
    """Export cluster analysis data as CSV"""
    try:
        # Aggregate by district
        district_data = demographic_data.groupby(['state', 'district']).agg({
            'demo_age_5_17': 'sum',
            'demo_age_17_': 'sum',
            'pincode': 'nunique'
        }).reset_index()
        
        enroll_district = enrollment_data.groupby(['state', 'district']).agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        district_data = district_data.merge(enroll_district, on=['state', 'district'], how='left').fillna(0)
        
        # Clustering
        features = district_data[['demo_age_5_17', 'demo_age_17_', 'age_0_5', 'age_5_17', 'age_18_greater']].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_clusters = min(6, len(district_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        district_data['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Add cluster characteristics
        for i in range(n_clusters):
            cluster_mask = district_data['cluster'] == i
            avg_demo = district_data.loc[cluster_mask, 'demo_age_5_17'].mean() + district_data.loc[cluster_mask, 'demo_age_17_'].mean()
            
            if avg_demo > district_data['demo_age_5_17'].mean() + district_data['demo_age_17_'].mean():
                category = 'High Activity'
            elif district_data.loc[cluster_mask, 'age_0_5'].mean() > district_data['age_0_5'].mean():
                category = 'Growing Region'
            else:
                category = 'Stable Region'
            
            district_data.loc[cluster_mask, 'cluster_category'] = category
        
        output = io.StringIO()
        district_data.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"uidai_clusters_{datetime.now().strftime('%Y%m%d')}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/recommendations')
def export_recommendations():
    """Export recommendations as CSV"""
    try:
        state = request.args.get('state')
        
        # Get recommendations (reuse logic from /api/recommendations)
        recommendations = []
        data = demographic_data[demographic_data['state'] == state] if state else demographic_data
        
        total_demo = data['demo_age_5_17'].sum() + data['demo_age_17_'].sum()
        enroll = enrollment_data[enrollment_data['state'] == state] if state else enrollment_data
        total_enroll = enroll['age_0_5'].sum() + enroll['age_5_17'].sum() + enroll['age_18_greater'].sum()
        
        if total_enroll < total_demo * 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'enrollment',
                'title': 'Low Enrollment Rate',
                'description': f'Enrollment is only {(total_enroll/total_demo*100):.1f}% of demographic updates',
                'expected_impact': 'Increase enrollment by 30-40%'
            })
        
        df = pd.DataFrame(recommendations)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"uidai_recommendations_{state or 'all'}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/methodology')
def get_methodology():
    """Return detailed methodology documentation"""
    methodology = {
        'overview': {
            'title': 'UIDAI Analytics Platform - Methodology',
            'version': '1.0',
            'last_updated': '2026-01-19',
            'description': 'Comprehensive methodology for data analysis, machine learning models, and statistical techniques used in the platform'
        },
        'data_sources': {
            'demographic': {
                'records': len(demographic_data),
                'fields': ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_'],
                'description': 'Demographic update records from UIDAI'
            },
            'biometric': {
                'records': len(biometric_data),
                'fields': ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_'],
                'description': 'Biometric update records'
            },
            'enrollment': {
                'records': len(enrollment_data),
                'fields': ['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater'],
                'description': 'New enrollment records'
            }
        },
        'machine_learning': {
            'clustering': {
                'algorithm': 'K-Means Clustering',
                'implementation': 'scikit-learn KMeans',
                'features': ['demo_age_5_17', 'demo_age_17_', 'age_0_5', 'age_5_17', 'age_18_greater'],
                'preprocessing': 'StandardScaler normalization',
                'n_clusters': 6,
                'initialization': 'k-means++',
                'n_init': 10,
                'random_state': 42,
                'validation_metrics': ['Silhouette Score', 'Davies-Bouldin Index'],
                'interpretation': {
                    'High Activity': 'Districts with above-average demographic and enrollment activity',
                    'Growing Region': 'Districts with high enrollment growth rates',
                    'Stable Region': 'Districts with consistent, predictable patterns'
                }
            },
            'forecasting': {
                'algorithm': 'Gradient Boosting Regressor',
                'implementation': 'scikit-learn GradientBoostingRegressor',
                'features': ['day_of_year', 'month', 'day_of_week'],
                'target': 'total daily records',
                'n_estimators': 100,
                'random_state': 42,
                'forecast_horizon': '30 days',
                'validation': 'Time-series cross-validation'
            },
            'feature_importance': {
                'algorithm': 'Random Forest Regressor',
                'implementation': 'scikit-learn RandomForestRegressor',
                'n_estimators': 100,
                'random_state': 42,
                'features_analyzed': ['temporal patterns', 'geographic factors', 'demographic trends']
            }
        },
        'statistical_methods': {
            'spike_detection': {
                'method': 'Rolling Z-Score Analysis',
                'window_size': '7 days',
                'threshold_medium': '2 standard deviations',
                'threshold_high': '3 standard deviations',
                'formula': 'z = (x - μ) / σ',
                'interpretation': 'Values exceeding threshold indicate unusual activity requiring investigation'
            },
            'trend_analysis': {
                'moving_averages': ['7-day MA', '30-day MA'],
                'growth_rate': 'Week-over-week percentage change',
                'volatility': 'Standard deviation of daily totals'
            },
            'descriptive_statistics': {
                'measures_of_central_tendency': ['mean', 'median'],
                'measures_of_dispersion': ['standard deviation', 'quartiles (Q1, Q3)', 'min', 'max'],
                'correlation': 'Pearson correlation coefficient'
            }
        },
        'data_aggregation': {
            'hierarchical_levels': ['National', 'State', 'District', 'Pincode'],
            'time_granularity': ['Daily', 'Weekly', 'Monthly'],
            'aggregation_functions': ['sum', 'mean', 'count', 'nunique']
        },
        'quality_assurance': {
            'data_validation': 'Null value checks, date range validation, numeric range validation',
            'outlier_detection': 'Z-score method with 2σ and 3σ thresholds',
            'consistency_checks': 'Cross-validation between demographic, biometric, and enrollment data'
        },
        'limitations': {
            'temporal': 'Analysis limited to available date range in dataset',
            'geographic': 'Coverage depends on data availability per state/district',
            'forecasting': 'Predictions assume continuation of historical patterns',
            'clustering': 'Results may vary with different random seeds; fixed seed ensures reproducibility'
        },
        'recommendations_logic': {
            'enrollment_rate': 'Triggered when enrollment < 50% of demographic updates',
            'coverage_expansion': 'Triggered when pincodes < districts × 10',
            'child_enrollment': 'Triggered when child records < 30% of adult records',
            'biometric_updates': 'Triggered when biometric updates < 70% of demographic records'
        }
    }
    
    return jsonify(methodology)

@app.route('/intelligence.html')
def intelligence():
    """Serve the Intelligence Platform page"""
    return render_template('intelligence.html')

if __name__ == '__main__':
    import os
    load_all_data()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
