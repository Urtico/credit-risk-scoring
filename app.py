from flask import Flask, render_template_string, request, jsonify
import pickle
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

try:
    model = pickle.load(open('models/model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
except:
    model = None
    scaler = None

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Credit Risk</title>
    <style>
        body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; }
        .container { background: white; padding: 40px; border-radius: 10px; max-width: 500px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); }
        h1 { color: #333; text-align: center; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 5px; }
        button { width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 5px; font-weight: bold; cursor: pointer; margin-top: 10px; }
        button:hover { transform: translateY(-2px); }
        .result { margin-top: 30px; padding: 20px; border-radius: 5px; display: none; }
        .result.show { display: block; }
        .success { background: #d4edda; border: 2px solid #28a745; color: #155724; }
        .danger { background: #f8d7da; border: 2px solid #dc3545; color: #721c24; }
        .error { background: #ffe5e5; border: 2px solid #ff6b6b; color: #c92a2a; padding: 15px; border-radius: 5px; margin-top: 20px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Credit Risk Predictor</h1>
        
        <form id="form">
            <div class="form-group">
                <label>Age</label>
                <input type="number" id="age" required>
            </div>
            <div class="form-group">
                <label>Annual Income</label>
                <input type="number" id="income" required>
            </div>
            <div class="form-group">
                <label>Loan Amount</label>
                <input type="number" id="loan" required>
            </div>
            <div class="form-group">
                <label>Credit History Years</label>
                <input type="number" id="credit" required>
            </div>
            <div class="form-group">
                <label>Employment Years</label>
                <input type="number" id="employment" required>
            </div>
            <button type="submit">🔮 Predict Risk</button>
        </form>
        
        <div class="error" id="error"></div>
        <div class="result" id="result"></div>
    </div>

    <script>
        document.getElementById('form').addEventListener('submit', async (e) => {
            e.preventDefault();
            document.getElementById('error').style.display = 'none';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        Age: parseFloat(document.getElementById('age').value),
                        Annual_Income: parseFloat(document.getElementById('income').value),
                        Loan_Amount: parseFloat(document.getElementById('loan').value),
                        Credit_History_Years: parseFloat(document.getElementById('credit').value),
                        Employment_Years: parseFloat(document.getElementById('employment').value)
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    const rec = data.recommendation.toLowerCase();
                    const isApprove = rec.includes('approve');
                    
                    document.getElementById('result').innerHTML = `
                        <h2>📊 Result</h2>
                        <p><strong>Probability:</strong> ${(data.default_probability * 100).toFixed(2)}%</p>
                        <p><strong>Risk:</strong> ${data.risk_level}</p>
                        <p style="font-size: 24px; font-weight: bold; margin-top: 20px;">${data.recommendation}</p>
                    `;
                    document.getElementById('result').className = isApprove ? 'result show success' : 'result show danger';
                } else {
                    throw new Error(data.error);
                }
            } catch(e) {
                document.getElementById('error').innerHTML = '❌ Error: ' + e.message;
                document.getElementById('error').style.display = 'block';
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    input_array = np.array([[
        float(data['Age']),
        float(data['Annual_Income']),
        float(data['Loan_Amount']),
        float(data['Credit_History_Years']),
        float(data['Employment_Years'])
    ]])
    
    prob = float(model.predict_proba(scaler.transform(input_array))[0][1])
    
    if prob < 0.15:
        risk = "🟢 VERY_LOW"
        rec = "✅ APPROVE"
    elif prob < 0.35:
        risk = "🟡 LOW"
        rec = "✅ APPROVE"
    elif prob < 0.55:
        risk = "🟠 MEDIUM"
        rec = "⚠️ REVIEW"
    elif prob < 0.75:
        risk = "🔴 HIGH"
        rec = "❌ REJECT"
    else:
        risk = "⚫ VERY_HIGH"
        rec = "❌ REJECT"
    
    return jsonify({
        'status': 'success',
        'default_probability': prob,
        'risk_level': risk,
        'recommendation': rec,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    print("\n🏦 API STARTING...\n")
    app.run(debug=True, port=5000, host='127.0.0.1')