import urllib.request
import json
import time

url = "http://127.0.0.1:8080/predict"

data = {
    "Timestamp": "2022/09/01 00:20",
    "From_Bank": "10",
    "Account": "8000EBD30",
    "To_Bank": "10",
    "Account_1": "8000EBD30",
    "Amount_Received": 10.50,
    "Receiving_Currency": "US Dollar",
    "Amount_Paid": 10.50,
    "Payment_Currency": "US Dollar",
    "Payment_Format": "Reinvestment"
}

req = urllib.request.Request(url)
req.add_header('Content-Type', 'application/json; charset=utf-8')
jsondata = json.dumps(data).encode('utf-8')

print("Sending request to API...")
try:
    response = urllib.request.urlopen(req, jsondata)
    result = json.loads(response.read())
    print("API Response:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'read'):
        print(e.read().decode())
