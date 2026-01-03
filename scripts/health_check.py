import requests
import time
import sys
import os

# Replace these with your actual Google Cloud Run URLs
# You can find these by running 'gcloud run services list'
BACKEND_URL = os.getenv("BACKEND_URL", "https://asteroid-backend-617598390128.us-central1.run.app")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://asteroid-frontend-617598390128.us-central1.run.app")

def check_service(name, url):
    """Verifies basic connectivity to a service endpoint."""
    if "xxxxxx" in url:
        print(f"Checking {name}... ❌ CONFIG ERROR: Update URL in script.")
        return False

    print(f"Checking {name} at {url}...", end=" ", flush=True)
    try:
        # Using a 15s timeout to account for Cloud Run 'cold starts'
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            print("✅ ONLINE")
            return True
        else:
            print(f"⚠️ STATUS {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ OFFLINE ({type(e).__name__})")
        return False

def check_end_to_end():
    """Validates the full inference pipeline and JSON response structure."""
    print("Testing End-to-End Prediction Loop...", end=" ", flush=True)
    
    if "xxxxxx" in BACKEND_URL:
        print("❌ SKIPPED: Backend URL not set.")
        return False

    # Sample payload for testing
    payload = {
        "est_diameter_min": 0.15,
        "relative_velocity": 45000,
        "miss_distance": 1000000,
        "absolute_magnitude": 22.0
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            # Safety check for the 'probability' key to avoid schema errors
            prob = data.get('probability')
            is_haz = data.get('is_hazardous')
            
            if prob is not None:
                print(f"✅ SUCCESS (Result: {is_haz}, Probability: {prob})")
                return True
            else:
                print(f"❌ SCHEMA ERROR: 'probability' key missing in response: {data}")
                return False
        else:
            print(f"❌ BACKEND ERROR (Status: {response.status_code})")
            print(f"Detail: {response.text}")
            return False
    except Exception as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False

if __name__ == "__main__":
    print("--- Planetary Defense System: Cloud Health Dashboard ---\n")
    
    # 1. Check Connectivity
    # Backend health is usually at /health; Frontend is at root
    b_ok = check_service("Backend API", f"{BACKEND_URL}/health")
    f_ok = check_service("Frontend UI ", FRONTEND_URL)
    
    print("\n" + "="*60)
    
    # 2. Check Logic Loop
    if b_ok:
        e2e_ok = check_end_to_end()
    else:
        print("Note: End-to-End test skipped - Backend is unreachable.")
        e2e_ok = False
        
    print("="*60)
    
    if b_ok and f_ok and e2e_ok:
        print("\n✨ ALL SYSTEMS OPERATIONAL: Deployment is healthy.")
    else:
        print("\n🚨 ISSUES DETECTED: Review logs for 'asteroid-backend' in GCP Console.")
        sys.exit(1)