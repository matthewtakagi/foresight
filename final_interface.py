import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

try:
    plt.switch_backend('Agg')
    
    model = joblib.load('digital_oracle_model.pkl')
    le = joblib.load('le.pkl')
    model_columns = joblib.load('model_columns.pkl')
    df_raw = pd.read_csv("Health_Risk_Dataset.csv")
    
    # Use a high-risk patient as default baseline for demonstration
    high_risk_patients = df_raw[df_raw['Risk_Level'] == 'High']
    if high_risk_patients.empty:
        baseline_patient = df_raw.iloc[0] # Fallback to first row
    else:
        # Choose a patient who is clearly high risk
        try:
            baseline_patient = df_raw[df_raw['Patient_ID'] == 'P0860'].iloc[0]
        except:
             baseline_patient = high_risk_patients.iloc[0]


except FileNotFoundError:
    st.error("Error: Model or data files not found. Ensure 'digital_oracle_model.pkl', 'le.pkl', 'model_columns.pkl', and 'Health_Risk_Dataset.csv' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during file loading: {e}")
    st.stop()


# --- Clinical Thresholds and Constants ---

# Absolute physiological bounds (clipping)
VITAL_BOUNDS = {
    'Respiratory_Rate': {'low': 4, 'high': 60},
    'Oxygen_Saturation': {'low': 40, 'high': 100},
    'Systolic_BP': {'low': 50, 'high': 220},
    'Heart_Rate': {'low': 30, 'high': 220},
    'Temperature': {'low': 34.0, 'high': 42.0}
}

# NEWS2-based thresholds for visual bar coloring
THRESHOLDS = {
    'Respiratory_Rate': [4, 8, 11, 12, 20, 24, 25, 60], 
    'Oxygen_Saturation': [40, 91, 93, 96, 100], # [Abs Min, 3 pts, 2 pts, 0 pts start, Abs Max]
    'Systolic_BP': [50, 90, 100, 111, 219, 220], # [Abs Min, 3 pts L, 2 pts L, 0 pts start, 0 pts end, 3 pts R, Abs Max]
    'Heart_Rate': [30, 40, 50, 90, 110, 130, 220], # [Abs Min, 3 pts L, 1 pt L, 0 pts end, 1 pt R, 2 pts R, 3 pts R, Abs Max]
    'Temperature': [34.0, 35.0, 36.0, 38.0, 39.1, 42.0], # [Abs Min, 3 pts L, 1 pt L, 0 pts end, 1 pt R, 3 pts R, Abs Max]
}


# Simplified Deterioration Factors (Mu/Sigma for deterioration magnitude)
DETERIORATION_FACTORS = {
    'Respiratory_Rate': {'mu': 1.5, 'sigma': 0.5}, # Tends to increase
    'Oxygen_Saturation': {'mu': 1.0, 'sigma': 0.3}, # Tends to decrease
    'Systolic_BP': {'mu': 5.0, 'sigma': 3.0}, # Magnitude of change
    'Heart_Rate': {'mu': 8.0, 'sigma': 4.0}, # Tends to increase
    'Temperature': {'mu': 0.2, 'sigma': 0.1}, # Tends to increase
}


# --- Monte Carlo Simulation Helpers ---

def normal_random(mu, sigma):
    """Generates a random number from a normal distribution N(mu, sigma)."""
    return np.random.normal(loc=mu, scale=sigma)

def sample_deterioration_single(base_row):
    """Simulates a single step of physiological deterioration based on factors."""
    simulated = base_row.copy()
    
    for vital, factors in DETERIORATION_FACTORS.items():
        mu = factors['mu']
        sigma = factors['sigma']
        orig = simulated[vital]
        
        # 1. Sample deterioration magnitude (ensure magnitude is positive)
        sampled_deterioration = max(0.01, normal_random(mu, sigma))
        
        new_val = orig

        # 2. Determine direction of change based on physiological expectation
        if vital == "Oxygen_Saturation":
            new_val = orig - sampled_deterioration
        elif vital in ["Respiratory_Rate", "Heart_Rate", "Temperature"]:
            new_val = orig + sampled_deterioration
        elif vital == "Systolic_BP":
            low = 100 # Low end of safe range for SBP
            high = 140 # High end of safe range for SBP
            if orig < low:
                new_val = orig - sampled_deterioration # Worsening hypotension
            elif orig > high:
                new_val = orig + sampled_deterioration * 0.5 # Worsening hypertension (slower rate)
            else:
                new_val = orig - sampled_deterioration * 0.2 # Slight dip towards hypotension (more common in sepsis)
        
        # 3. Clip to absolute physiological bounds
        lo = VITAL_BOUNDS[vital]['low']
        hi = VITAL_BOUNDS[vital]['high']
        simulated[vital] = np.clip(new_val, lo, hi)
        
    # Consciousness (AVPU): Simulate possible worsening to a non-'A' state if currently 'A'
    if simulated['Consciousness'] == 'A' and np.random.rand() < 0.1: # 10% chance of worsening LoC
        simulated['Consciousness'] = np.random.choice(['V', 'P', 'U'])

    return simulated

def monte_carlo_future_prob(base_input, n_sims=300):
    """
    Runs the Monte Carlo simulation and returns mean, P95, and the array of High Risk probabilities.
    Returns: mean_prob, p95, n_sims, probs_array
    """
    prob_high_array = []
    
    # Ensure vitals are copied and converted to the correct numpy types for simulation
    base_vitals_sim = base_input.copy()
    for k, v in base_vitals_sim.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            base_vitals_sim[k] = float(v)
    
    for _ in range(n_sims):
        # 1. Sample the deteriorated state
        sim_row_dict = sample_deterioration_single(base_vitals_sim)
        
        # 2. Get prediction for the simulated state
        sim_risk_df = get_risk_prediction(sim_row_dict)
        
        # High Risk is the target index
        high_risk_prob = sim_risk_df.loc[sim_risk_df['Risk Level'] == 'High', 'Probability_raw'].iloc[0]
        prob_high_array.append(high_risk_prob)

    # Convert to numpy array for efficient percentile calculation
    probs = np.array(prob_high_array)
    mean_prob = np.mean(probs)
    
    # Calculate 95th percentile (P95) for worst-case future risk
    p95 = np.percentile(probs, 95)
    
    return mean_prob, p95, n_sims, probs # Now returns the full array

# --- Helper 1: NEWS2 Score Calculation ---
def calculate_news2(vitals):
    """Calculates the National Early Warning Score (NEWS2) based on input vitals."""
    score = 0
    rr = vitals['Respiratory_Rate']
    o2_sat = vitals['Oxygen_Saturation']
    sbp = vitals['Systolic_BP']
    hr = vitals['Heart_Rate']
    temp = vitals['Temperature']
    conscious = vitals['Consciousness']
    on_o2 = vitals['On_Oxygen']

    # 1. Respiratory Rate (RR)
    if rr <= 8 or rr >= 25: score += 3
    elif rr >= 21: score += 2
    elif rr >= 9 and rr <= 11: score += 1

    # 2. Oxygen Saturation (SpO2) - Using default Scale 2
    if o2_sat <= 91: score += 3
    elif o2_sat <= 93: score += 2
    elif o2_sat <= 95: score += 1
    # >95 is 0

    # 3. Systolic BP (SBP)
    if sbp <= 90 or sbp >= 220: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    # 111-219 is 0

    # 4. Heart Rate (HR)
    if hr <= 40 or hr >= 131: score += 3
    elif hr >= 111 and hr <= 130: score += 2
    elif hr <= 50 or (hr >= 91 and hr <= 110): score += 1
    # 51-90 is 0

    # 5. Temperature (Temp)
    if temp <= 35.0 or temp >= 39.1: score += 3
    elif temp <= 36.0 or temp >= 38.1: score += 1
    # 36.1-38.0 is 0

    # 6. Consciousness (AVPU)
    if conscious in ['V', 'P', 'U']: score += 3
    # A is 0

    # 7. Supplemental Oxygen (On_Oxygen)
    if on_o2 == 1: score += 2
    # Not on Oxygen is 0

    return score

# --- Helper 2: Model Prediction and Feature Alignment ---
def get_risk_prediction(input_data):
    """
    Predicts risk level probability for a single set of vitals.
    returns: DataFrame with columns ['Risk Level','Probability_raw','Probability']
    """
    patient_df = pd.DataFrame([input_data])

    # One-hot Consciousness columns
    consciousness_cols = [c for c in model_columns if c.startswith('Consciousness_')]
    for col in consciousness_cols:
        patient_df[col] = 0

    conscious_state = patient_df['Consciousness'].iloc[0] if 'Consciousness' in patient_df.columns else None
    if conscious_state is not None:
        target_col = f'Consciousness_{conscious_state}'
        if target_col in model_columns:
            patient_df[target_col] = 1

    patient_df = patient_df.drop('Consciousness', axis=1, errors='ignore')
    
    # Ensure correct column order and fill missing OHE columns with 0
    patient_data_aligned = patient_df.reindex(columns=model_columns, fill_value=0)

    # Convert to appropriate types before prediction if necessary (e.g., ensure float)
    for col in patient_data_aligned.columns:
        if patient_data_aligned[col].dtype == 'object':
            # This handles the case where O2_Scale might be read as object if all other values are integers
            patient_data_aligned[col] = pd.to_numeric(patient_data_aligned[col], errors='coerce').fillna(0)
        else:
            patient_data_aligned[col] = patient_data_aligned[col].astype(float)


    probs = model.predict_proba(patient_data_aligned)[0]  # array
    probs_pct = probs * 100.0

    risk_df = pd.DataFrame({
        'Risk Level': le.classes_,
        'Probability_raw': probs_pct
    })
    risk_df['Probability'] = risk_df['Probability_raw'].round(1).astype(str) + '%'
    return risk_df

# --- Helper 3: Mock SHAP/Feature Attribution (Influence on Dominant Class) ---
def get_feature_attributions(input_vitals):
    """
    Perturb each feature and measure change in the probability of the predicted class.
    This mimics a SHAP-like force calculation for transparency.
    Returns DataFrame: ['Feature', 'Influence_Raw', 'Influence_Direction']
    """
    base_df = get_risk_prediction(input_vitals)
    # Find the predicted class (highest probability index)
    predicted_class = base_df.loc[base_df['Probability_raw'].idxmax(), 'Risk Level']
    base_prob = float(base_df.loc[base_df['Risk Level'] == predicted_class, 'Probability_raw'].iloc[0])

    influences = {}

    for k, v in input_vitals.items():
        if k in ['O2_Scale', 'Patient_ID', 'Risk_Level', 'Risk_Level_Encoded']: continue

        # Handle numeric features (RR, O2_Sat, SBP, HR, Temp)
        if isinstance(v, (int, float, np.integer, np.floating)):
            # Define perturbation delta (meaningful change)
            delta = 0.5 if abs(v) < 5 else max(1.0, abs(v) * 0.05)

            # Test positive perturbation
            plus = input_vitals.copy()
            plus[k] = v + delta
            p_plus = float(get_risk_prediction(plus).loc[lambda df: df['Risk Level'] == predicted_class, 'Probability_raw'].iloc[0])
            
            # Test negative perturbation
            minus = input_vitals.copy()
            minus[k] = max( (v - delta), 0 ) 
            p_minus = float(get_risk_prediction(minus).loc[lambda df: df['Risk Level'] == predicted_class, 'Probability_raw'].iloc[0])
            
            # Find the perturbation that has the largest positive impact (or smallest negative) on the PREDICTED class
            change_plus = p_plus - base_prob
            change_minus = p_minus - base_prob

            best_change = change_plus if abs(change_plus) >= abs(change_minus) else change_minus
            
            influences[k] = best_change

        # Handle categorical feature (Consciousness)
        elif k == 'Consciousness':
            states = ['A', 'V', 'P', 'U']
            best_change = 0
            # Test flipping to the 'most normal' state 'A'
            if v != 'A':
                test = input_vitals.copy()
                test[k] = 'A'
                p_test = float(get_risk_prediction(test).loc[lambda df: df['Risk Level'] == predicted_class, 'Probability_raw'].iloc[0])
                # The influence of being in a non-'A' state is the *negative* of the change when fixing it to 'A'
                influences[k] = -(p_test - base_prob) 
            
        
        # Handle binary feature (On_Oxygen)
        elif k == 'On_Oxygen':
            if v == 1: # Patient is on Oxygen, which often implies higher baseline risk
                test = input_vitals.copy()
                test[k] = 0 # Test turning off O2
                p_test = float(get_risk_prediction(test).loc[lambda df: df['Risk Level'] == predicted_class, 'Probability_raw'].iloc[0])
                # The influence of being ON oxygen is the *negative* of the change when turning it off
                influences[k] = -(p_test - base_prob) 
            else:
                 influences[k] = 0 
            
    influence_df = pd.DataFrame(influences.items(), columns=['Feature', 'Influence_Raw'])
    influence_df['Influence_Direction'] = influence_df['Influence_Raw'].apply(lambda x: 'Positive (Risk Increasing)' if x > 0 else 'Negative (Risk Decreasing)' if x < 0 else 'Neutral')
    influence_df['Influence_Abs'] = influence_df['Influence_Raw'].abs()
    
    # Sort by absolute influence and keep only non-zero influences
    influence_df = influence_df[influence_df['Influence_Raw'].abs() > 0.01].sort_values(by='Influence_Abs', ascending=False)
    
    return influence_df.head(5), predicted_class, base_prob # Return top 5 influential features

# --- Helper 4: Vitals Table Styling Core Logic (used for pointer color in the bar) ---
def get_value_style(vital, value): 
    """Returns a tuple of (background_color, text_color, severity_label) based on clinical thresholds (NEWS2/Standard)."""
    
    # List of vitals that are expected to be numeric
    numeric_vitals = ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate', 'Temperature']

    # Handle numeric vitals
    if vital in numeric_vitals:
        # Safely convert the input value to float
        try:
            value = float(value)
        except ValueError:
            return ('#FFFFFF', '#000000', 'Unknown') # Default white background if conversion fails

        # CRITICAL (Red Flag)
        if (vital == 'Respiratory_Rate' and (value <= 8 or value >= 25)) or \
           (vital == 'Oxygen_Saturation' and value <= 91) or \
           (vital == 'Systolic_BP' and (value <= 90 or value >= 220)) or \
           (vital == 'Heart_Rate' and (value <= 40 or value >= 131)) or \
           (vital == 'Temperature' and (value <= 35.0 or value >= 39.1)):
            return ('#f7a8a8', '#000000', '⚠️ CRITICAL')
        
        # CONCERNING (Yellow Flag)
        elif (vital == 'Respiratory_Rate' and (value <= 11 or value >= 21)) or \
             (vital == 'Oxygen_Saturation' and value <= 95) or \
             (vital == 'Systolic_BP' and (value <= 100 or value >= 210)) or \
             (vital == 'Heart_Rate' and (value <= 50 or value >= 111)) or \
             (vital == 'Temperature' and (value <= 36.0 or value >= 38.1)):
            return ('#fae39d', '#000000', '⚠ Concerning')
        
        # NORMAL
        else:
            return ('#e8f5e9', '#000000', '✓ Normal')
    
    # Handle categorical/binary vitals
    
    # Consciousness (V, P, U is concerning)
    if vital == 'Consciousness':
        if str(value) in ['V', 'P', 'U']:
            return ('#f7a8a8', '#000000', '⚠️ CRITICAL')
        else:
            return ('#e8f5e9', '#000000', '✓ Alert')
    
    # On_Oxygen (1 is concerning)
    if vital == 'On_Oxygen':
        if str(value) == '1':
            return ('#fae39d', '#000000', '⚠ On O2')
        else:
            return ('#e8f5e9', '#000000', '✓ Room Air')

    # Default case
    return ('#FFFFFF', '#000000', '')

# --- Helper 5: Create Color-Coded Vital Display ---
def create_vital_display(vital, current_value):
    """
    Generates a color-coded display for the vital sign.
    Returns formatted HTML string with value and status badge.
    """
    bg_color, text_color, severity_label = get_value_style(vital, current_value)
    
    # Format the value for display
    if vital in ['Respiratory_Rate', 'Oxygen_Saturation', 'Systolic_BP', 'Heart_Rate']:
        display_value = f"{int(current_value)}"
    elif vital == 'Temperature':
        display_value = f"{current_value:.1f}°C"
    elif vital == 'Consciousness':
        display_value = f"{current_value}"
    elif vital == 'On_Oxygen':
        display_value = "Yes" if current_value == 1 else "No"
    elif vital == 'O2_Scale':
        display_value = f"{current_value:.1f} L/min"
    else:
        display_value = str(current_value)
    
    # Create the HTML with colored badge
    html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; margin: 4px 0; 
                background-color: {bg_color}; border-radius: 6px; border-left: 4px solid {text_color};">
        <span style="font-size: 18px; font-weight: bold; color: #1e1e1e;">{display_value}</span>
        <span style="font-size: 12px; color: #424242; font-weight: 600;">{severity_label}</span>
    </div>
    """
    
    return html

# --- Helper 6: AI-Driven Recommendation Mock (Simulated Gemini Output) ---
def get_recommendations(vitals, news2_score, risk_level):
    """
    Simulates a call to a structured LLM API to get context-aware recommendations.
    Returns a list of structured dictionaries.
    """
    recommendations = []
    
    if news2_score >= 7 or risk_level == 'High':
        # Critical patient pathway
        recommendations.append({
            'title': 'Senior Clinician Review', 
            'intervention': 'Immediate request for a review by a senior physician/critical care team.', 
            'target': 'Overall Condition',
            'severity': 'Critical'
        })
        
        if vitals['Oxygen_Saturation'] <= 91:
            recommendations.append({
                'title': 'Urgent Oxygen Escalation', 
                'intervention': 'Increase O2 therapy to non-rebreather mask (15 L/min) or consider non-invasive ventilation (NIV). Monitor SpO2 continuously.', 
                'target': 'Oxygen_Saturation',
                'severity': 'Critical'
            })
        elif vitals['Oxygen_Saturation'] <= 95:
             recommendations.append({
                'title': 'Adjust Oxygen Flow', 
                'intervention': 'Increase O2 flow by 1-2 L/min and reassess SpO2 in 5 minutes. Confirm correct device fit.', 
                'target': 'Oxygen_Saturation',
                'severity': 'Urgent'
            })
            
        if vitals['Systolic_BP'] <= 90 or vitals['Heart_Rate'] >= 131:
            recommendations.append({
                'title': 'Fluid Resuscitation Protocol', 
                'intervention': 'Administer 500ml of IV crystalloid fluid challenge STAT. Repeat assessment of BP and HR after 30 minutes.', 
                'target': 'Systolic_BP / Heart_Rate',
                'severity': 'Critical'
            })
            
        if vitals['Temperature'] >= 39.1:
            recommendations.append({
                'title': 'Manage Hyperthermia/Infection', 
                'intervention': 'Administer weight-based Paracetamol (Acetaminophen) dose. Initiate septic workup (cultures, lactate).', 
                'target': 'Temperature',
                'severity': 'Urgent'
            })
        
        if vitals['Consciousness'] != 'A':
            recommendations.append({
                'title': 'Neurovascular Check',
                'intervention': 'Perform a blood glucose check and reassess level of consciousness (GCS). Protect airway if necessary.',
                'target': 'Consciousness',
                'severity': 'Critical'
            })

    elif news2_score >= 3 or risk_level == 'Medium':
        # Medium risk pathway
        recommendations.append({
            'title': 'Increase Monitoring Frequency', 
            'intervention': 'Escalate vital sign monitoring to every 1-2 hours.', 
            'target': 'NEWS2 Score',
            'severity': 'Standard'
        })
        
        if vitals['Respiratory_Rate'] >= 21:
             recommendations.append({
                'title': 'Respiratory Assessment', 
                'intervention': 'Sit patient up and assess for accessory muscle use. Document respiratory effort.', 
                'target': 'Respiratory_Rate',
                'severity': 'Urgent'
            })
            
    else:
        # Low/Normal risk pathway
        recommendations.append({
            'title': 'Routine Monitoring', 
            'intervention': 'Maintain routine monitoring schedule (e.g., every 4 hours). Focus on comfort and preventative care.', 
            'target': 'Overall Stability',
            'severity': 'Standard'
        })
        
    return pd.DataFrame(recommendations)

# --- NEW: Helper 7: Create NEWS2 Gauge Chart ---
def create_news2_gauge(score):
    """Creates a visual gauge chart for NEWS2 score."""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Color zones based on NEWS2 scoring
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
    zones = [0, 3, 5, 7, 20]
    labels = ['Low (0-2)', 'Low-Medium (3-4)', 'Medium (5-6)', 'High (7+)']
    
    # Draw colored zones
    for i in range(len(zones)-1):
        width = zones[i+1] - zones[i]
        ax.barh(0, width, left=zones[i], 
                height=0.4, color=colors[i], alpha=0.8, 
                edgecolor='white', linewidth=2)
        
        # Add zone labels
        mid_point = zones[i] + width/2
        if width > 1:  # Only add label if zone is wide enough
            ax.text(mid_point, 0, labels[i], 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
    
    # Score indicator (triangle marker)
    ax.scatter([score], [0], s=800, c='black', marker='v', zorder=5, edgecolors='white', linewidths=2)
    
    # Score text below marker
    ax.text(score, -0.25, f'Score: {score}', 
           ha='center', fontsize=14, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))
    
    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title('NEWS2 Score Assessment', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# --- NEW: Helper 8: Load Random Patient ---
def load_random_patient():
    """Loads a random patient from the dataset and returns their vitals."""
    random_patient = df_raw.sample(n=1).iloc[0]
    
    return {
        'Respiratory_Rate': int(random_patient['Respiratory_Rate']),
        'Oxygen_Saturation': int(random_patient['Oxygen_Saturation']),
        'Systolic_BP': int(random_patient['Systolic_BP']),
        'Heart_Rate': int(random_patient['Heart_Rate']),
        'Temperature': float(random_patient['Temperature']),
        'Consciousness': random_patient['Consciousness'],
        'On_Oxygen': int(random_patient['On_Oxygen']),
        'O2_Scale': float(random_patient.get('O2_Scale', 1.0)),
        'Patient_ID': random_patient.get('Patient_ID', 'Unknown'),
        'Actual_Risk': random_patient.get('Risk_Level', 'Unknown')
    }

# --- UI design ---
st.set_page_config(page_title="Clinical Intervention Simulator (Digital Oracle)", layout="wide")

st.markdown("""
    <style>
    /* Tailwind classes are simulated for styling, but custom CSS is added for clarity */
    .stApp {
        background-color: #f4f7f9;
        font-family: 'Inter', sans-serif;
    }
    .header-style {
        color: #0d47a1;
        border-bottom: 3px solid #0d47a1;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .stTable {
        border-radius: 8px;
        overflow: hidden;
    }
    .stPlot {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background-color: white;
    }
    .recommendation-box {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .critical { background-color: #ffe0e0; border-left: 5px solid #ff5252; }
    .urgent { background-color: #fff8e1; border-left: 5px solid #ffc107; }
    .standard { background-color: #e8f5e9; border-left: 5px solid #4caf50; }
    .rec-title { font-weight: bold; color: #1e88e5; margin-bottom: 5px; }
    .intervention-preset {
        padding: 10px;
        margin: 5px 0;
        border-radius: 6px;
        border-left: 4px solid #1e88e5;
        background-color: #e3f2fd;
        cursor: pointer;
        transition: all 0.3s;
    }
    .intervention-preset:hover {
        background-color: #bbdefb;
        transform: translateX(5px);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="header-style">Foresight: Clinical Intervention Simulator</h1>', unsafe_allow_html=True)
st.markdown("Use the controls below to simulate clinical interventions and quantify the resulting change in a patient's predicted health risk.")

# Initialize Session State for Baseline and History
if 'baseline_vitals' not in st.session_state:
    st.session_state['baseline_vitals'] = {
        'Respiratory_Rate': int(baseline_patient['Respiratory_Rate']),
        'Oxygen_Saturation': int(baseline_patient['Oxygen_Saturation']),
        'Systolic_BP': int(baseline_patient['Systolic_BP']),
        'Heart_Rate': int(baseline_patient['Heart_Rate']),
        'Temperature': float(baseline_patient['Temperature']),
        'Consciousness': baseline_patient['Consciousness'],
        'On_Oxygen': int(baseline_patient['On_Oxygen']),
        'O2_Scale': float(baseline_patient.get('O2_Scale', 1.0))
    }
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'current_patient_id' not in st.session_state:
    st.session_state['current_patient_id'] = baseline_patient.get('Patient_ID', 'P0860')

if 'current_actual_risk' not in st.session_state:
    st.session_state['current_actual_risk'] = baseline_patient.get('Risk_Level', 'High')

# --- NEW: Patient Management in Sidebar ---
st.sidebar.header("Patient Management")

# Display current patient info
st.sidebar.info(f"**Current Patient:** {st.session_state['current_patient_id']}\n\n**Actual Risk Level:** {st.session_state['current_actual_risk']}")

col_patient1, col_patient2 = st.sidebar.columns(2)

with col_patient1:
    if st.button("Random Patient", use_container_width=True):
        random_patient_data = load_random_patient()
        
        # Update session state with new patient
        st.session_state['baseline_vitals'] = {
            'Respiratory_Rate': random_patient_data['Respiratory_Rate'],
            'Oxygen_Saturation': random_patient_data['Oxygen_Saturation'],
            'Systolic_BP': random_patient_data['Systolic_BP'],
            'Heart_Rate': random_patient_data['Heart_Rate'],
            'Temperature': random_patient_data['Temperature'],
            'Consciousness': random_patient_data['Consciousness'],
            'On_Oxygen': random_patient_data['On_Oxygen'],
            'O2_Scale': random_patient_data['O2_Scale']
        }
        
        st.session_state['current_patient_id'] = random_patient_data['Patient_ID']
        st.session_state['current_actual_risk'] = random_patient_data['Actual_Risk']
        
        # Clear history for new patient
        st.session_state['history'] = []
        
        st.sidebar.success(f"Loaded: {random_patient_data['Patient_ID']}")
        st.rerun()

with col_patient2:
    if st.button("Reset Session", use_container_width=True):
        # Clear history but keep current patient
        st.session_state['history'] = []
        st.sidebar.success("Session history cleared!")
        st.rerun()

st.sidebar.markdown("---")

# --- NEW: FEATURE 4 - Intervention Library ---
st.sidebar.header("Quick Interventions")
st.sidebar.markdown("Apply standard clinical interventions with one click:")

# Define intervention presets with their effects
intervention_presets = {
    "Oxygen Therapy (+2L)": {
        'description': 'Increase oxygen flow by 2 L/min',
        'function': lambda v: {
            **v, 
            'O2_Scale': min(v['O2_Scale'] + 2, 15.0), 
            'On_Oxygen': 1,
            'Oxygen_Saturation': min(v['Oxygen_Saturation'] + 2, 100)
        }
    },
    "IV Fluid Bolus (500ml)": {
        'description': 'Administer fluid challenge for hypotension/tachycardia',
        'function': lambda v: {
            **v, 
            'Systolic_BP': min(v['Systolic_BP'] + 10, 220), 
            'Heart_Rate': max(v['Heart_Rate'] - 8, 30)
        }
    },
    "Antipyretic Given": {
        'description': 'Administer paracetamol for fever reduction',
        'function': lambda v: {
            **v, 
            'Temperature': max(v['Temperature'] - 0.8, 34.0)
        }
    },
    "Position Change (Sitting Up)": {
        'description': 'Reposition patient to improve oxygenation',
        'function': lambda v: {
            **v, 
            'Oxygen_Saturation': min(v['Oxygen_Saturation'] + 3, 100),
            'Respiratory_Rate': max(v['Respiratory_Rate'] - 2, 4)
        }
    },
    "Bronchodilator Given": {
        'description': 'Administer inhaled bronchodilator',
        'function': lambda v: {
            **v, 
            'Respiratory_Rate': max(v['Respiratory_Rate'] - 3, 4),
            'Oxygen_Saturation': min(v['Oxygen_Saturation'] + 2, 100),
            'Heart_Rate': min(v['Heart_Rate'] + 5, 220)  # Common side effect
        }
    }
}

# Display intervention buttons
selected_intervention = None
for intervention_name, intervention_data in intervention_presets.items():
    if st.sidebar.button(intervention_name, use_container_width=True):
        selected_intervention = intervention_name

if selected_intervention:
    st.sidebar.success(f"✅ Applied: {selected_intervention}")
    st.sidebar.info(intervention_presets[selected_intervention]['description'])

st.sidebar.markdown("---")

# --- Vitals Inputs ---
st.sidebar.header("Current Vitals & Parameters")

vitals = {}

# If an intervention was selected, apply it to the baseline
if selected_intervention:
    base_vitals = intervention_presets[selected_intervention]['function'](st.session_state['baseline_vitals'])
else:
    base_vitals = st.session_state['baseline_vitals']

# Inputs are cast to match model's expected types (int/float)
vitals['Respiratory_Rate'] = st.sidebar.number_input(
    "1. Respiratory Rate (breaths/min)", 
    min_value=5, max_value=60, 
    value=int(base_vitals['Respiratory_Rate']), 
    step=1, format="%d"
)
vitals['Oxygen_Saturation'] = st.sidebar.number_input(
    "2. Oxygen Saturation (%)", 
    min_value=50, max_value=100, 
    value=int(base_vitals['Oxygen_Saturation']), 
    step=1, format="%d"
)
vitals['Systolic_BP'] = st.sidebar.number_input(
    "3. Systolic BP (mmHg)", 
    min_value=40, max_value=220, 
    value=int(base_vitals['Systolic_BP']), 
    step=1, format="%d"
)
vitals['Heart_Rate'] = st.sidebar.number_input(
    "4. Heart Rate (bpm)", 
    min_value=20, max_value=220, 
    value=int(base_vitals['Heart_Rate']), 
    step=1, format="%d"
)
vitals['Temperature'] = st.sidebar.number_input(
    "5. Temperature (°C)", 
    min_value=30.0, max_value=45.0, 
    value=float(base_vitals['Temperature']), 
    step=0.1, format="%.1f"
)
vitals['Consciousness'] = st.sidebar.selectbox(
    "6. Consciousness (AVPU)", 
    ['A', 'V', 'P', 'U'], 
    index=['A','V','P','U'].index(base_vitals['Consciousness'])
)
vitals['On_Oxygen'] = int(st.sidebar.radio(
    "7. Supplemental Oxygen?", 
    [0, 1], 
    index=1 if base_vitals['On_Oxygen'] == 1 else 0, 
    format_func=lambda x: 'Yes' if x == 1 else 'No'
))
vitals['O2_Scale'] = st.sidebar.number_input(
    "8. O2 Scale (L/min)", 
    min_value=0.0, max_value=15.0, 
    value=float(base_vitals.get('O2_Scale', 1.0)), 
    step=0.1, format="%.1f"
)

if st.sidebar.button("Set Current Vitals as New Baseline"):
    # Update baseline and log the event if history is not empty
    if st.session_state['history']:
        # Log the current risk profile *before* setting new baseline
        current_risk_df = get_risk_prediction(vitals.copy())
        
        log_entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'vitals': vitals.copy(),
            'risk_profile': current_risk_df.set_index('Risk Level')['Probability_raw'].to_dict(),
            'news2': calculate_news2(vitals.copy())
        }
        st.session_state['history'].append(log_entry)
        
    st.session_state['baseline_vitals'] = vitals.copy()
    st.sidebar.success("New Baseline Set and Current State Logged.")

# --- NEW: FEATURE 1 - Data Export ---
st.sidebar.markdown("---")
st.sidebar.header("Data Management")

if st.session_state['history']:
    # Prepare export data
    export_records = []
    for entry in st.session_state['history']:
        record = {
            'Patient_ID': st.session_state['current_patient_id'],
            'Actual_Risk_Level': st.session_state['current_actual_risk'],
            'Timestamp': entry['time'],
            'NEWS2': entry['news2'],
            **entry['vitals'],
            **{f'Risk_{k}': v for k, v in entry['risk_profile'].items()}
        }
        export_records.append(record)
    
    export_df = pd.DataFrame(export_records)
    csv = export_df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="Download Session Data (CSV)",
        data=csv,
        file_name=f"clinical_session_{st.session_state['current_patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.sidebar.info(f"Session contains {len(st.session_state['history'])} recorded states")
else:
    st.sidebar.info("No session data to export yet. Run simulations to generate data.")

# --- Display Current State ---
initial_risk_df = get_risk_prediction(vitals.copy())
predicted_class = initial_risk_df.loc[initial_risk_df['Probability_raw'].idxmax(), 'Risk Level']
news2_score = calculate_news2(vitals.copy())

# NEW LAYOUT: Two columns - vitals on left, stacked risk & attribution on right
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Current Vitals Status")
    st.markdown("*(Color-coded by NEWS2 severity: Green=Normal, Yellow=Concerning, Red=Critical)*")
    
    # Order for display
    vitals_display_order = [
        ('Respiratory_Rate', 'Respiratory Rate'),
        ('Oxygen_Saturation', 'Oxygen Saturation'),
        ('Systolic_BP', 'Systolic BP'),
        ('Heart_Rate', 'Heart Rate'),
        ('Temperature', 'Temperature'),
        ('Consciousness', 'Consciousness'),
        ('On_Oxygen', 'Supplemental O2'),
        ('O2_Scale', 'O2 Flow Rate')
    ]

    for vital_key, vital_label in vitals_display_order:
        vital_value = vitals[vital_key]
        
        # Display label
        st.markdown(f"**{vital_label}:**")
        
        # Display color-coded value
        display_html = create_vital_display(vital_key, vital_value)
        st.markdown(display_html, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- NEWS2 Gauge Chart ---
    st.markdown("### NEWS2 Score Assessment")
    gauge_fig = create_news2_gauge(news2_score)
    st.pyplot(gauge_fig)
    plt.close(gauge_fig)  # Clean up
    
    # Add interpretation
    if news2_score >= 7:
        st.error("**HIGH RISK** - Urgent clinical response required")
    elif news2_score >= 5:
        st.warning("**MEDIUM RISK** - Increase monitoring frequency")
    elif news2_score >= 3:
        st.info("ℹ️ **LOW-MEDIUM RISK** - Continue monitoring")
    else:
        st.success("**LOW RISK** - Routine monitoring")


with col2:
    # Section 2: Predicted Risk
    st.subheader("Predicted Risk")
    st.markdown(f"**Model Prediction:** <span style='font-size: 24px; color: #1e88e5; font-weight: bold;'>{predicted_class}</span>", unsafe_allow_html=True)

    # Bar chart of probabilities
    chart = alt.Chart(initial_risk_df).mark_bar().encode(
        x=alt.X('Probability_raw:Q', title='Probability (%)'),
        y=alt.Y('Risk Level:N', sort='-x', title='Risk Level'),
        color=alt.Color('Risk Level:N', scale=alt.Scale(
            domain=['Normal', 'Low', 'Medium', 'High'],
            range=['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
        )),
        tooltip=['Risk Level', alt.Tooltip('Probability_raw:Q', format='.1f')]
    ).properties(height=250, title='Risk Profile Distribution')
    
    st.altair_chart(chart, use_container_width=True)
    st.table(initial_risk_df.set_index('Risk Level')[['Probability']])
    
    st.markdown("---")
    
    # Section 3: Feature Attribution (stacked below)
    st.subheader("Feature Attribution")
    
    try:
        influence_df, dominant_class, base_prob = get_feature_attributions(vitals.copy())
        
        if not influence_df.empty:
            st.markdown(f"Model prediction of **{dominant_class} ({base_prob:.1f}%)** is influenced by:")
            
            # Create a simple bar chart to mock the SHAP forces plot
            influence_df['Display_Name'] = influence_df['Feature'].str.replace('_', ' ')
            
            # Use Altair to plot forces
            base = alt.Chart(influence_df).encode(
                y=alt.Y('Display_Name:N', title=None, sort=alt.EncodingSortField(field='Influence_Abs', order='descending'))
            )
            
            # Color points based on direction
            points = base.mark_circle(size=100).encode(
                x=alt.X('Influence_Raw:Q', title='Influence on Predicted Risk Probability (pp)'),
                color=alt.Color('Influence_Direction', scale=alt.Scale(
                    domain=['Positive (Risk Increasing)', 'Negative (Risk Decreasing)', 'Neutral'],
                    range=['#F44336', '#4CAF50', '#9E9E9E']
                )),
                tooltip=['Display_Name', alt.Tooltip('Influence_Raw:Q', format='.2f')]
            )

            # Add zero line
            line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[2,2]).encode(x='y')
            
            st.altair_chart((line + points).properties(height=250), use_container_width=True)
            
        else:
            st.info("Feature attribution is too close to zero to display.")
    except Exception as e:
        st.error(f"Could not compute feature attribution: {e}")

st.markdown("---")
st.header("Intervention Simulation & Trajectory")

# --- Simulation Button: Compare Baseline vs Current ---
col_sim, col_hist = st.columns([1, 2])
with col_sim:
    st.markdown("Click **Simulate** to compare the stored **Baseline Vitals** to the **Current Vitals** (as set in the sidebar).")
    if st.button("Simulate Intervention and Check New Risk Profile", use_container_width=True):
        st.subheader("Intervention Impact")
    
        baseline_v = st.session_state['baseline_vitals']
        baseline_df = get_risk_prediction(baseline_v)
        simulated_df = get_risk_prediction(vitals)
    
        # Log the current intervention state
        current_risk_df = simulated_df.copy()
        log_entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'vitals': vitals.copy(),
            'risk_profile': current_risk_df.set_index('Risk Level')['Probability_raw'].to_dict(),
            'news2': calculate_news2(vitals.copy())
        }
        st.session_state['history'].append(log_entry)

        # Merge for comparison table
        baseline_df = baseline_df.rename(columns={'Probability': 'Baseline Risk (%)', 'Probability_raw': 'Baseline_raw'})
        simulated_df = simulated_df.rename(columns={'Probability': 'Current Risk (%)', 'Probability_raw': 'Current_raw'})
    
        comparison_df = baseline_df.merge(simulated_df, on='Risk Level')
        comparison_df['Change_raw'] = comparison_df['Current_raw'] - comparison_df['Baseline_raw']
        comparison_df['Change (%)'] = comparison_df['Change_raw'].round(1).astype(str) + '%'
    
        # Display comparison table
        st.table(comparison_df[['Risk Level', 'Baseline Risk (%)', 'Current Risk (%)', 'Change (%)']].set_index('Risk Level'))
    
        # Success/Warning message
        risk_reduction = comparison_df[(comparison_df['Risk Level'].isin(['High', 'Medium'])) & (comparison_df['Change_raw'] < 0)]
        if not risk_reduction.empty and risk_reduction['Change_raw'].min() < -5.0:
            st.success(f"Significant Risk Reduction! High Risk probability reduced by {abs(risk_reduction[risk_reduction['Risk Level'] == 'High']['Change_raw'].min()):.1f} percentage points.")
        else:
            st.warning("Risk profile shift was minor or towards higher risk. Try another intervention.")
            
        # --- FUTURE FRAGILITY ANALYSIS ---
        st.markdown("---")
        with st.expander("**Future Fragility Analysis (Monte Carlo)**", expanded=True):
            with st.spinner("Running Monte Carlo Simulation (300 Scenarios)..."):
                # Retrieve the full array of simulated probabilities
                mean_prob, p95, n_sims, probs_array = monte_carlo_future_prob(vitals.copy(), n_sims=300)
            
            st.subheader(f"Simulated Risk Distribution over {n_sims} Scenarios")

            # --- Altair Visualization of Probability Distribution ---
            fragility_df = pd.DataFrame({'High Risk Probability (%)': probs_array})
            
            # Create the Histogram
            base_chart = alt.Chart(fragility_df).encode(
                x=alt.X('High Risk Probability (%):Q', bin=alt.Bin(maxbins=30), title='Simulated High Risk Probability (%)'),
                y=alt.Y('count()', title='Frequency of Scenarios'),
            ).properties(height=250)
            
            # Add Histogram Bars (Blue)
            histogram = base_chart.mark_bar(opacity=0.7, color='#1e88e5').encode(
                tooltip=[alt.Tooltip('High Risk Probability (%):Q', bin=True, title='Prob Range'), 'count()']
            )

            # Data for reference lines
            ref_lines_df = pd.DataFrame({
                'Value': [mean_prob, p95],
                'Label': ['Mean', 'P95 Worst-Case'],
                'Color': ['orange', '#F44336']
            })
            
            # Add Reference Lines (Mean and P95)
            ref_lines = alt.Chart(ref_lines_df).mark_rule(size=2).encode(
                x='Value',
                color=alt.Color('Label:N', scale=alt.Scale(domain=['Mean', 'P95 Worst-Case'], range=['orange', '#F44336']), legend=alt.Legend(title="Reference")),
                tooltip=[alt.Tooltip('Value', title='Risk Prob (%)', format='.1f'), 'Label']
            )

            st.altair_chart(histogram + ref_lines, use_container_width=True)
            # --- End Altair Visualization ---
            
            # Display P95 result as the key metric
            p95_color = "#F44336" if p95 >= 50 else ("#FFC107" if p95 >= 25 else "#4CAF50")
            
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 8px; border: 2px solid {p95_color}; background-color: #ffffff; text-align: center;">
                <p style="font-size: 16px; color: #666;">P95 Worst-Case High Risk Fragility Score</p>
                <p style="font-size: 32px; font-weight: bold; color: {p95_color}; margin: 0;">{p95:.1f}%</p>
                <p style="font-size: 12px; color: #666; margin-top: 5px;">(Mean Simulated High Risk: {mean_prob:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <small>The **P95 score** is the probability threshold exceeded in only the worst 5% of deterioration scenarios. It measures the patient's **fragility** to expected clinical stress.</small>
            """, unsafe_allow_html=True)
            
            if p95 >= 50:
                st.error("**Clinical Alert:** The patient is **highly fragile**. Minor clinical deterioration is very likely to push them into a high-risk state.")
            elif p95 >= 25:
                st.warning("**Caution:** The patient shows **moderate fragility**. Increased surveillance is warranted as deterioration risk is significant.")
            else:
                st.success("The patient is currently stable, with **low predicted fragility** even under expected clinical stress.")


with col_hist:
    st.subheader("Patient Risk Trajectory")
    
    if st.session_state['history']:
        # Prepare data for history plot
        history_records = []
        for i, entry in enumerate(st.session_state['history']):
            # Convert dict risk profile to rows for Altair melt
            for risk_level, prob in entry['risk_profile'].items():
                history_records.append({
                    'Intervention_Step': i + 1,
                    'Timestamp': entry['time'],
                    'Risk_Level': risk_level,
                    'Probability': prob,
                    'NEWS2': entry['news2']
                })
        
        history_df = pd.DataFrame(history_records)
        
        # Set a consistent color scale for all risk levels
        color_scale = alt.Scale(
            domain=['Normal', 'Low', 'Medium', 'High'],
            range=['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
        )
        
        if not history_df.empty:
            # Chart displaying all four risk level probabilities
            chart_all_risks = alt.Chart(history_df).mark_line(point=True).encode(
                x=alt.X('Intervention_Step:O', title='Intervention Step', axis=alt.Axis(tickMinStep=1)),
                y=alt.Y('Probability:Q', title='Probability (%)', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('Risk_Level:N', scale=color_scale, title="Risk Level"),
                tooltip=['Timestamp', 'Probability', 'Risk_Level', 'NEWS2']
            ).properties(title='Full Risk Profile Trajectory Over Interventions')

            st.altair_chart(chart_all_risks, use_container_width=True)
            
        # Display history table with NEWS2
        history_table = history_df[history_df['Risk_Level'] == predicted_class].drop(columns=['Risk_Level']).rename(columns={'Probability': f'{predicted_class} Prob (%)'})
        history_table = history_table[['Timestamp', 'Intervention_Step', f'{predicted_class} Prob (%)', 'NEWS2']]
        st.dataframe(history_table.tail(5), use_container_width=True, hide_index=True)

    else:
        st.info("Run the simulation a few times to start tracking the patient's risk trajectory and see if your interventions are effective!")

# --- Trend Analysis ---
if len(st.session_state['history']) >= 3:
    st.markdown("---")
    st.header("5. Statistical Trend Analysis")
    
    # Extract high risk probabilities over time
    high_risk_trend = [
        entry['risk_profile'].get('High', 0) 
        for entry in st.session_state['history']
    ]
    
    # Calculate trend using linear regression
    x = np.arange(len(high_risk_trend))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, high_risk_trend)
    
    col_trend1, col_trend2, col_trend3, col_trend4 = st.columns(4)
    
    with col_trend1:
        trend_direction = "Improving" if slope < 0 else "Worsening"
        trend_color = "normal" if slope < 0 else "inverse"
        st.metric(
            "Trend Direction", 
            trend_direction,
            f"{slope:.2f}% per step",
            delta_color=trend_color
        )
    
    with col_trend2:
        correlation_strength = "Strong" if abs(r_value) > 0.7 else ("Moderate" if abs(r_value) > 0.4 else "Weak")
        st.metric(
            "Trend Strength", 
            correlation_strength,
            f"r = {r_value:.3f}"
        )
    
    with col_trend3:
        volatility = np.std(high_risk_trend)
        stability_status = "Stable" if volatility < 5 else ("Moderate" if volatility < 10 else "Volatile")
        st.metric(
            "Risk Stability",
            stability_status,
            f"σ = {volatility:.2f}%"
        )
    
    with col_trend4:
        current_high_risk = high_risk_trend[-1]
        previous_high_risk = high_risk_trend[-2] if len(high_risk_trend) > 1 else high_risk_trend[-1]
        delta_risk = current_high_risk - previous_high_risk
        st.metric(
            "Latest Change",
            f"{current_high_risk:.1f}%",
            f"{delta_risk:+.1f}%"
        )
    
    # Visualization of trend with regression line
    st.markdown("### High Risk Probability Trend")
    
    trend_df = pd.DataFrame({
        'Step': x,
        'High_Risk_Probability': high_risk_trend,
        'Trend_Line': slope * x + intercept
    })
    
    # Create scatter plot
    scatter = alt.Chart(trend_df).mark_circle(size=100, color='#1e88e5').encode(
        x=alt.X('Step:Q', title='Intervention Step'),
        y=alt.Y('High_Risk_Probability:Q', title='High Risk Probability (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['Step', alt.Tooltip('High_Risk_Probability:Q', format='.1f')]
    )
    
    # Create trend line
    line = alt.Chart(trend_df).mark_line(color='#F44336', strokeDash=[5,5]).encode(
        x='Step:Q',
        y='Trend_Line:Q'
    )
    
    st.altair_chart((scatter + line).properties(height=300), use_container_width=True)
    
    # Statistical interpretation
    st.markdown("#### Interpretation")
    
    if abs(r_value) > 0.7 and slope < -2:
        st.success(f"**Strong positive trend**: Interventions show consistent risk reduction (slope: {slope:.2f}%/step, r²: {r_value**2:.3f})")
    elif abs(r_value) > 0.7 and slope > 2:
        st.error(f"**Strong negative trend**: Patient condition is consistently deteriorating (slope: {slope:.2f}%/step, r²: {r_value**2:.3f}). Review intervention strategy.")
    elif volatility > 10:
        st.warning(f"**High volatility detected**: Risk fluctuates significantly (σ = {volatility:.2f}%). Consider more frequent monitoring.")
    else:
        st.info(f"**Moderate trends**: Continue current monitoring strategy. Trend slope: {slope:.2f}%/step")

st.markdown("---")
st.header("Smart Recommendations")
st.markdown(f"Based on the **NEWS2 Score of {news2_score}** and a **{predicted_class} Risk** prediction, here are prioritized intervention suggestions:")

# --- Display Recommendations ---
recommendations_df = get_recommendations(vitals.copy(), news2_score, predicted_class)

if not recommendations_df.empty:
    for _, rec in recommendations_df.iterrows():
        # Determine CSS class for visual severity
        severity_class = rec['severity'].lower().replace(' ', '') 
        
        st.markdown(
            f"""
            <div class="recommendation-box {severity_class}">
                <p class="rec-title">{rec['title']} (Target: {rec['target']})</p>
                <p>{rec['intervention']}</p>
            </div>
            """, unsafe_allow_html=True
        )
else:
    st.info("No immediate clinical interventions are explicitly recommended based on the current thresholds.")

# --- Final Check: Initial Log ---
if not st.session_state['history']:
    # Log the initial state automatically on first load for a starting point
    initial_log_entry = {
        'time': datetime.now().strftime("%H:%M:%S") + " (Initial)",
        'vitals': vitals.copy(),
        'risk_profile': initial_risk_df.set_index('Risk Level')['Probability_raw'].to_dict(),
        'news2': calculate_news2(vitals.copy())
    }
    st.session_state['history'].append(initial_log_entry)