import requests
import json
import time

def test_api():
    """Test the FastAPI endpoints"""
    BASE_URL = "http://localhost:8000"
    
    print("🧪 Testing Patient Readmission Prediction API...")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("   ✅ Health check PASSED")
            print(f"   Response: {response.json()}")
        else:
            print("   ❌ Health check FAILED")
            return False
    except Exception as e:
        print(f"   ❌ Health check ERROR: {e}")
        return False
    
    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        if response.status_code == 200:
            print("   ✅ Model info PASSED")
            print(f"   Loaded models: {list(response.json().keys())}")
        else:
            print("   ❌ Model info FAILED")
            return False
    except Exception as e:
        print(f"   ❌ Model info ERROR: {e}")
        return False
    
    # Test 3: Prediction
    print("\n3. Testing prediction endpoint...")
    try:
        sample_patient = {
            "patient_id": "TEST_001",
            "patient_data": {
                "age": 72,
                "blood_pressure": 145,
                "cholesterol": 235,
                "previous_admissions": 3,
                "comorbidity_count": 2,
                "medication_count": 6,
                "lab_result_1": 1.8,
                "lab_result_2": 0.9
            },
            "model_type": "readmission"
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=sample_patient)
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Prediction PASSED")
            print(f"   Patient ID: {result['patient_id']}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Probability: {result['probability']:.3f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"   ❌ Prediction FAILED: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Prediction ERROR: {e}")
        return False
    
    print("\n🎉 All API tests passed successfully!")
    return True

if __name__ == "__main__":
    print("Note: Make sure the API is running on http://localhost:8000")
    print("Start the API with: python deployment/fastapi_app.py")
    print()
    
    # Wait a bit for API to be ready
    time.sleep(2)
    test_api()
