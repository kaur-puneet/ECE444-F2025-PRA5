import requests
import time
import csv
import matplotlib.pyplot as plt

# Replace with your deployed EB URL
BASE_URL = "http://fakenewsapp-env.eba-mxanibg2.us-east-2.elasticbeanstalk.com"

# Define 4 test cases (2 fake news, 2 real news)
test_cases = [
    {"text": "This news is completely false."},                       # FAKE
    {"text": "Breaking: UFO spotted over New York City!"},            # FAKE
    {"text": "This is real news."},                                   # REAL
    {"text": "The university announced the official graduation date."} # REAL
]

# Functional test
print("=== Functional Tests ===")
for i, case in enumerate(test_cases, 1):
    try:
        response = requests.post(f"{BASE_URL}/predict", json=case, timeout=10)
        print(f"Test case {i}: Input: {case['text']} Prediction: {response.json().get('prediction')}")
    except requests.exceptions.RequestException as e:
        print(f"Test case {i}: FAILED ({e})")

# Latency/performance test
print("\n=== Latency Tests ===")
all_timestamps = []  # To save all latencies
latency_per_case = {}  # To compute average latency per test case

for i, case in enumerate(test_cases, 1):
    print(f"Running latency test for Test Case {i}...")
    timestamps = []
    for j in range(100):
        try:
            start = time.time()
            response = requests.post(f"{BASE_URL}/predict", json=case, timeout=10)
            end = time.time()
            timestamps.append(end - start)
        except requests.exceptions.RequestException as e:
            print(f"Request {j+1} failed: {e}")
            timestamps.append(None)  # Record None for failed requests
        if (j + 1) % 20 == 0:
            print(f"  Completed {j+1}/100 requests")
    
    latency_per_case[i] = [t for t in timestamps if t is not None]  # exclude failures
    all_timestamps.extend([(i, t) for t in latency_per_case[i]])    # (test_case_id, latency)

# Save one CSV with all data
csv_filename = "latency_all_cases.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_case_id", "latency_seconds"])
    for test_id, latency in all_timestamps:
        writer.writerow([test_id, latency])

print(f"Saved all latency data to {csv_filename}")

# Generate boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([latency_per_case[i] for i in sorted(latency_per_case.keys())],
            labels=[f"Case {i}" for i in sorted(latency_per_case.keys())])
plt.ylabel("Latency (seconds)")
plt.title("API Latency per Test Case")
plt.grid(True)
plt.savefig("latency_boxplot.png")
plt.show()
print("Saved boxplot as latency_boxplot.png")

# Print average latency per test case
print("\n=== Average Latency per Test Case ===")
for test_id in sorted(latency_per_case.keys()):
    avg_latency = sum(latency_per_case[test_id]) / len(latency_per_case[test_id])
    print(f"Test Case {test_id}: Average latency = {avg_latency:.4f} seconds")
