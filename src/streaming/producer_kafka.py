# src/streaming/producer_kafka.py
import time
import json
import argparse
import pandas as pd
from kafka import KafkaProducer

def run_producer(csv_path, kafka_bootstrap, topic, rate=10.0, max_rows=None):
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)
    
    producer = KafkaProducer(
        bootstrap_servers=kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"[producer] Streaming {len(df)} rows to '{topic}' at {rate} tx/s")
    
    for _, row in df.iterrows():
        tx = {k: float(v) if pd.notna(v) else 0.0 for k, v in row.to_dict().items()}
        producer.send(topic, tx)
        time.sleep(1.0 / rate)
    
    producer.flush()
    producer.close()
    print("[producer] Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/creditcard.csv")
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    run_producer(args.csv, args.kafka, args.topic, args.rate, args.max_rows)

