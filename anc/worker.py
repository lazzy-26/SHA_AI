import argparse
import pickle
import socket
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE,
    DEFAULT_PORT,
    DEFAULT_WORKERS,
    EPOCHS,
    LEARNING_RATE,
)
from dataset import get_worker_loader
from model import create_model


class SHAANCWorker:
    def __init__(
        self,
        server_ip,
        port,
        worker_id,
        num_workers,
        epochs,
        batch_size,
        learning_rate,
    ):
        self.server_ip = server_ip
        self.port = port

        self.worker_id = worker_id
        self.num_workers = num_workers

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = create_model()

        self.criterion = nn.L1Loss()

        self.socket = None

    def connect(
        self,
        max_retries=30,
        delay=2.0,
    ):
        for attempt in range(
            1,
            max_retries + 1,
        ):
            try:
                self.socket = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )

                self.socket.settimeout(
                    300
                )

                self.socket.connect(
                    (
                        self.server_ip,
                        self.port,
                    )
                )

                print(
                    f"[Worker {self.worker_id}] "
                    f"Connected to "
                    f"{self.server_ip}:"
                    f"{self.port}"
                )

                return

            except OSError:
                print(
                    f"[Worker {self.worker_id}] "
                    f"Waiting for server "
                    f"({attempt}/"
                    f"{max_retries})..."
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Worker {self.worker_id} "
            "could not connect."
        )

    def send_bytes(self, data):
        self.socket.sendall(
            len(data).to_bytes(
                8,
                "big",
            )
        )

        self.socket.sendall(data)

    def recv_bytes(self):
        header = self._recv_exact(8)

        if not header:
            return b""

        size = int.from_bytes(
            header,
            "big",
        )

        return self._recv_exact(size)

    def _recv_exact(self, size):
        buffer = bytearray()

        while len(buffer) < size:
            chunk = self.socket.recv(
                min(
                    65536,
                    size - len(buffer),
                )
            )

            if not chunk:
                raise ConnectionError(
                    "Server connection closed."
                )

            buffer.extend(chunk)

        return bytes(buffer)

    def receive_weights(self):
        payload = self.recv_bytes()

        weights = pickle.loads(
            payload
        )

        self.model.load_state_dict(
            weights
        )

    def train_local_epoch(self):
        loader = get_worker_loader(
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.model.train()

        total_loss = 0.0
        batches = 0

        for noisy, clean in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            enhanced = self.model(
                noisy
            )

            loss = self.criterion(
                enhanced,
                clean,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
            )

            batches += 1

        avg_loss = (
            total_loss
            / max(batches, 1)
        )

        return avg_loss

    def send_weights(
        self,
        loss,
    ):
        payload = {
            "worker_id":
                self.worker_id,

            "loss":
                loss,

            "weights":
                {
                    key:
                        value.cpu()
                    for key, value
                    in self.model
                    .state_dict()
                    .items()
                },
        }

        self.send_bytes(
            pickle.dumps(
                payload,
                protocol=
                    pickle.HIGHEST_PROTOCOL,
            )
        )

    def train(self):
        self.connect()

        print(
            f"[Worker {self.worker_id}] "
            f"Starting S.H.A ANC training"
        )

        for epoch in range(
            1,
            self.epochs + 1,
        ):
            print()
            print(
                f"[Worker {self.worker_id}] "
                f"Epoch {epoch}/"
                f"{self.epochs}"
            )

            print(
                "  Receiving "
                "global model..."
            )

            self.receive_weights()

            print(
                "  Training local "
                "audio shard..."
            )

            start = time.time()

            loss = (
                self.train_local_epoch()
            )

            elapsed = (
                time.time()
                - start
            )

            print(
                f"  Local loss: "
                f"{loss:.6f}"
            )

            print(
                f"  Time: "
                f"{elapsed:.2f}s"
            )

            print(
                "  Sending updated "
                "model to server..."
            )

            self.send_weights(
                loss
            )

        print(
            f"[Worker {self.worker_id}] "
            "Training finished."
        )

        self.socket.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "S.H.A ANC distributed worker"
        )
    )

    parser.add_argument(
        "--server",
        required=True,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )

    parser.add_argument(
        "--id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )

    args = parser.parse_args()

    worker = SHAANCWorker(
        server_ip=args.server,
        port=args.port,
        worker_id=args.id,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    worker.train()


if __name__ == "__main__":
    main()