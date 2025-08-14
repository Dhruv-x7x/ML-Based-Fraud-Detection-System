# src/streaming/consumer_kafka.py
import time
import json
import argparse
import requests
from kafka import KafkaConsumer


def kafka_consumer_loop(
    kafka_bootstrap, topic, api_url, batch_size=16, consumer_timeout_ms=1000
):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_bootstrap,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=consumer_timeout_ms,
    )
    print(f"[consumer] Subscribed to {topic}, posting to API {api_url}/predict_batch")
    buffer = []
    try:
        for msg in consumer:
            buffer.append(
                {
                    "Time": float(msg.value.get("Time", 0.0)),
                    "Amount": float(msg.value.get("Amount", 0.0)),
                    **{
                        f"V{i}": float(msg.value.get(f"V{i}", 0.0))
                        for i in range(1, 29)
                    },
                }
            )
            if len(buffer) >= batch_size:
                resp = requests.post(
                    f"{api_url}/predict_batch",
                    json={
                        "transactions": [
                            {
                                "Time": b["Time"],
                                "Amount": b["Amount"],
                                **{k: b[k] for k in b if k.startswith("V")},
                            }
                            for b in buffer
                        ]
                    },
                    timeout=30,
                )
                print(
                    f"[consumer] posted batch size={len(buffer)} status={resp.status_code}"
                )
                buffer = []
        # flush remaining
        if buffer:
            resp = requests.post(
                f"{api_url}/predict_batch",
                json={
                    "transactions": [
                        {
                            "Time": b["Time"],
                            "Amount": b["Amount"],
                            **{k: b[k] for k in b if k.startswith("V")},
                        }
                        for b in buffer
                    ]
                },
                timeout=30,
            )
            print(
                f"[consumer] posted final batch size={len(buffer)} status={resp.status_code}"
            )
    except KeyboardInterrupt:
        print("[consumer] Interrupted by user")
    finally:
        consumer.close()
        print("[consumer] Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kafka", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    kafka_consumer_loop(args.kafka, args.topic, args.api, args.batch)
