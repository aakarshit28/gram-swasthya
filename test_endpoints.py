import urllib.request
import urllib.parse
import json
import http.cookiejar

BASE_URL = "http://127.0.0.1:5000"

def test_home():
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200, "Home page failed"

def test_disease_info():
    data = json.dumps({"disease": "dengue"}).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/disease_info", data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        res = json.loads(response.read().decode('utf-8'))
        assert res["pmjay_covered"] == True

def test_flow():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    
    # Try register
    data = json.dumps({
        "name": "Test Asha",
        "phone": "9999999999",
        "password": "test",
        "role": "asha"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/register", data=data, headers={'Content-Type': 'application/json'})
    try:
        response = opener.open(req)
    except Exception:
        pass # phone might be registered
        
    # Login
    data = json.dumps({
        "phone": "9999999999",
        "password": "test"
    }).encode('utf-8')
    req = urllib.request.Request(f"{BASE_URL}/login", data=data, headers={'Content-Type': 'application/json'})
    response = opener.open(req)
    assert response.status == 200
    res = json.loads(response.read().decode('utf-8'))
    assert res["success"] == True
    
    # Check dashboard 
    req = urllib.request.Request(f"{BASE_URL}/asha_dashboard")
    response = opener.open(req)
    assert response.status == 200
    html = response.read().decode('utf-8')
    assert "ASHA Worker Dashboard" in html

try:
    test_home()
    print("test_home passed")
    test_disease_info()
    print("test_disease_info passed")
    test_flow()
    print("test_flow passed")
    print("All endpoints OK!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Test failed: {str(e)}")
