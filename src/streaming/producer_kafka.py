# src/streaming/producer_kafka.py
import time
import json
import argparse
import pandas as pd
from kafka import KafkaProducer


def kafka_producer_loop(
    csv_path, kafka_bootstrap, topic, rate=5.0, max_rows=None, repeat=False
):
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(int(max_rows))
    producer = KafkaProducer(
        bootstrap_servers=kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print(f"[producer] Starting: {len(df)} rows -> topic '{topic}' at {rate} tx/s")
    try:
        while True:
            for _, row in df.iterrows():
                tx = row.to_dict()
                # ensure floats are normal python floats
                tx = {k: (float(v) if pd.notna(v) else 0.0) for k, v in tx.items()}
                producer.send(topic, tx)
                time.sleep(1.0 / rate)
            producer.flush()
            if not repeat:
                break
    except KeyboardInterrupt:
        print("[producer] Interrupted by user")
    finally:
        producer.flush()
        producer.close()
        print("[producer] Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/creditcard.csv")
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument(
        "--rate", type=float, default=5.0, help="transactions per second"
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--repeat", action="store_true", help="loop dataset indefinitely"
    )
    args = parser.parse_args()
    kafka_producer_loop(
        args.csv, args.kafka, args.topic, args.rate, args.max_rows, args.repeat
    )
