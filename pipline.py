import socket
import json
import threading
import traceback
from numpy._core.fromnumeric import mean
import pandas as pd
import os
import time

clean_lock = threading.Lock()
bad_lock = threading.Lock()
drift_lock = threading.Lock()

def needs_header(filepath):
    return (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0)

clean_header_needed = needs_header("clean_reading.csv")
bad_header_needed = needs_header("bad_reading.csv")

clean_file = open("clean_reading.csv", "a", newline="", encoding="utf-8")
bad_file = open("bad_reading.csv", "a", newline="", encoding="utf-8")

def validation(data):
    global clean_header_needed, bad_header_needed
    df = pd.DataFrame([data])
    
    reading_id = data["reading_id"]
    confidence = data["confidence_score"]
    station_id = data["station_id"]
    model_name = data["model_name"]
    predicted_label = data["predicted_label"]
    response_time = data["response_time_ms"]
    timestamp = data["timestamp"]


    valid_reading_id = (
        isinstance(reading_id, int)
        and reading_id > 0
    )

    valid_station_id = station_id in [
        "ST-01", "ST-02", "ST-03", "ST-04", "ST-05",
        "ST-06", "ST-07", "ST-08", "ST-09", "ST-10"
    ]

    valid_model_name = model_name in [
        "leak_detector",
        "pressure_drop_predictor",
        "demand_forecaster"
    ]

    valid_confidence = (
        isinstance(confidence, (int, float))
        and 0 <= confidence <= 1
    )

    valid_response_time = (
        isinstance(response_time, (int, float))
        and response_time > 0
    )

    valid_predicted_label = False

    if model_name == "leak_detector":

        valid_predicted_label = predicted_label in [
            "leak",
            "normal"
        ]

    elif model_name == "pressure_drop_predictor":

        valid_predicted_label = predicted_label in [
            "drop",
            "stable"
        ]

    elif model_name == "demand_forecaster":

        valid_predicted_label = (
            isinstance(predicted_label, int)
            and 1 <= predicted_label <= 5000
        )


    try:
        pd.to_datetime(timestamp)
        valid_timestamp = True

    except (ValueError, TypeError):
        valid_timestamp = False


    valid_record = (
        valid_reading_id
        and valid_station_id
        and valid_model_name
        and valid_predicted_label
        and valid_confidence
        and valid_response_time
        and valid_timestamp
    )

    if valid_record:
        with clean_lock:
            df.to_csv(clean_file, header=clean_header_needed, index=False)  
            clean_file.flush()
            clean_header_needed = False
    else:
        with bad_lock:
            df.to_csv(bad_file, header=bad_header_needed, index=False)  
            bad_file.flush()
            bad_header_needed = False

    return validation

def calculate():
    clean_read = pd.read_csv("clean_reading.csv")
    bad_read = pd.read_csv("bad_reading.csv")

    total = clean_read["reading_id"].count() + bad_read["reading_id"].count()
    reading_id_clean = clean_read["reading_id"].count()
    reading_id_bad = bad_read["reading_id"].count()

    leak_confidence = clean_read[clean_read["model_name"] == "leak_detector"]["confidence_score"].mean()
    pressure_confidence = clean_read[clean_read["model_name"] == "pressure_drop_predictor"]["confidence_score"].mean()
    demand_confidence = clean_read[clean_read["model_name"] == "demand_forecaster"]["confidence_score"].mean()

    leak_response_time = clean_read[clean_read["model_name"] == "leak_detector"]["response_time_ms"].mean()
    pressure_response_time = clean_read[clean_read["model_name"] == "pressure_drop_predictor"]["response_time_ms"].mean()
    demand_response_time = clean_read[clean_read["model_name"] == "demand_forecaster"]["response_time_ms"].mean()

    low_confidence_count = clean_read[clean_read["confidence_score"] < 0.5].shape[0]
    active_stations = clean_read["station_id"].nunique()

    report = pd.DataFrame([{
        "total_readings": total,
        "clean_readings": reading_id_clean,
        "bad_readings": reading_id_bad,
        "leak_mean_confidence": leak_confidence,
        "pressure_mean_confidence": pressure_confidence,
        "demand_mean_confidence": demand_confidence,
        "leak_mean_response_time": leak_response_time,
        "pressure_mean_response_time": pressure_response_time,
        "demand_mean_response_time": demand_response_time,
        "low_confidence_count": low_confidence_count,
        "active_stations": active_stations
    }])

    report_exists = os.path.exists("real_time_reports.csv")
    report.to_csv("real_time_reports.csv", mode="a", header=not report_exists, index=False)

def validation_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", 9034))

    buffer = ""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                data = json.loads(line)
                print(data)
                validation(data)
    finally:
        sock.close()
        clean_file.close()
        bad_file.close()
    return validation_loop

def calculate_loop():
    while True:
        time.sleep(20)
        try:
            calculate()
        except Exception:
            print("=== خطا در calculate ===")
            traceback.print_exc()

BASELINE_SIZE = 30
WINDOW_SIZE = 10
Z_THRESHOLD = 0.5

MODEL_NAMES = ["leak_detector", "pressure_drop_predictor", "demand_forecaster"]
METRICS = ["confidence_score", "response_time_ms"]
baseline_stats = {}

drift_header_needed = (not os.path.exists("alerts_drift_model.csv"))or (os.path.getsize("alerts_drift_model.csv") == 0)

def compute_baseline():
    global baseline_stats

    clean_read = pd.read_csv("clean_reading.csv")

    for model in MODEL_NAMES:
        model_data = clean_read[clean_read["model_name"] == model].head(BASELINE_SIZE)

        baseline_stats[model] = {}

        for metric in METRICS:
            mean = model_data[metric].mean()
            std = model_data[metric].std()

            if std == 0 or pd.isna(std):
                std = 1e-6

            baseline_stats[model][metric] = {"mean": mean, "std": std}


def detect_drift():
    global drift_header_needed

    clean_read = pd.read_csv("clean_reading.csv")

    if not baseline_stats:
        compute_baseline()
        return

    alerts = []

    for model in MODEL_NAMES:
        if model not in baseline_stats:
            continue

    model_data = clean_read[clean_read["model_name"] == model]
    current_window = model_data.tail(WINDOW_SIZE)

    for metric in METRICS:
        base_mean = baseline_stats[model][metric]["mean"]
        base_std = baseline_stats[model][metric]["std"]

    current_mean = current_window[metric].mean()
    z_score = (current_mean - base_mean) / base_std

    if abs(z_score) > Z_THRESHOLD:
            alerts.append({
                "timestamp": pd.Timestamp.now(),
                "model_name": model,
                "baseline_mean": base_mean,
                "current_window_mean": current_mean,
                "z_score": z_score,
                "message": f"Drift detected in {metric}"
            })

    if alerts:
        alert_df = pd.DataFrame(alerts)
        with drift_lock:
            alert_df.to_csv(
                "alerts_drift_model.csv",
                mode="a",
                header=drift_header_needed,
                index=False
            )
            drift_header_needed = False

def drift_detection_loop():
    while True:
        time.sleep(180)
        try:
            detect_drift()
        except Exception:
            print("=== خطا در detect_drift ===")
            traceback.print_exc()

if __name__ == "__main__":
    compute_baseline()

    t1 = threading.Thread(target=validation_loop, daemon=True)
    t2 = threading.Thread(target=calculate_loop, daemon=True)
    t3 = threading.Thread(target=drift_detection_loop, daemon=True)
    t1.start()
    t2.start(5)
    t3.start(10)

    t1.join()
    t2.join(5)
    t3.join(10)