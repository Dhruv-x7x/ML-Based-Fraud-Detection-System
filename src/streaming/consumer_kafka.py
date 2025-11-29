# src/streaming/consumer_kafka.py
import json
import argparse
import requests
from kafka import KafkaConsumer

def run_consumer(kafka_bootstrap, topic, api_url, model="lightgbm", batch_size=16):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000,
    )
    print(f"[consumer] Listening on '{topic}', model={model}")
    
    buffer = []
    for msg in consumer:
        tx = msg.value
        buffer.append({
            "Time": float(tx.get("Time", 0)),
            "Amount": float(tx.get("Amount", 0)),
            **{f"V{i}": float(tx.get(f"V{i}", 0)) for i in range(1, 29)}
        })
        
        if len(buffer) >= batch_size:
            resp = requests.post(f"{api_url}/predict_batch", json={"transactions": buffer, "model": model}, timeout=30)
            print(f"[consumer] batch={len(buffer)} status={resp.status_code}")
            buffer = []
    
    if buffer:
        requests.post(f"{api_url}/predict_batch", json={"transactions": buffer, "model": model}, timeout=30)
    
    consumer.close()
    print("[consumer] Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    run_consumer(args.kafka, args.topic, args.api, args.model, args.batch)

