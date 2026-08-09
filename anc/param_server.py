import argparse
import pickle
import socket
import threading
import time

import torch

from config import (
    DEFAULT_PORT,
    DEFAULT_WORKERS,
    EPOCHS,
    GLOBAL_MODEL_PATH,
)
from model import create_model


class SHAANCParameterServer:
    def __init__(
        self,
        port,
        num_workers,
        epochs,
    ):
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs

        self.model = create_model()

        self.workers = {}

        self.updates = {}

        self.lock = threading.Lock()

    @staticmethod
    def send_bytes(
        sock,
        data,
    ):
        sock.sendall(
            len(data).to_bytes(
                8,
                "big",
            )
        )

        sock.sendall(data)

    @staticmethod
    def recv_exact(
        sock,
        size,
    ):
        buffer = bytearray()

        while len(buffer) < size:
            chunk = sock.recv(
                min(
                    65536,
                    size - len(buffer),
                )
            )

            if not chunk:
                raise ConnectionError(
                    "Worker disconnected."
                )

            buffer.extend(chunk)

        return bytes(buffer)

    @classmethod
    def recv_bytes(
        cls,
        sock,
    ):
        header = cls.recv_exact(
            sock,
            8,
        )

        size = int.from_bytes(
            header,
            "big",
        )

        return cls.recv_exact(
            sock,
            size,
        )

    def handle_worker(
        self,
        worker_id,
        sock,
    ):
        try:
            while True:
                payload = self.recv_bytes(
                    sock
                )

                update = pickle.loads(
                    payload
                )

                with self.lock:
                    self.updates[
                        worker_id
                    ] = update

        except Exception as error:
            print(
                f"[Server] Worker "
                f"{worker_id} stopped: "
                f"{error}"
            )

    def broadcast_model(self):
        state = {
            key:
                value.cpu()
            for key, value
            in self.model
            .state_dict()
            .items()
        }

        payload = pickle.dumps(
            state,
            protocol=
                pickle.HIGHEST_PROTOCOL,
        )

        for worker_id, sock in (
            self.workers.items()
        ):
            try:
                self.send_bytes(
                    sock,
                    payload,
                )

            except Exception as error:
                print(
                    f"[Server] Broadcast "
                    f"failed for worker "
                    f"{worker_id}: "
                    f"{error}"
                )

    def wait_for_updates(
        self,
        timeout=600,
    ):
        self.updates.clear()

        deadline = (
            time.time()
            + timeout
        )

        while True:
            with self.lock:
                count = len(
                    self.updates
                )

            if count >= self.num_workers:
                break

            if time.time() > deadline:
                raise TimeoutError(
                    "Timed out waiting "
                    "for worker updates."
                )

            time.sleep(0.1)

        return [
            self.updates[
                worker_id
            ]
            for worker_id
            in sorted(
                self.updates
            )
        ]

    def average_models(
        self,
        updates,
    ):
        worker_states = [
            update["weights"]
            for update in updates
        ]

        averaged = {}

        keys = worker_states[
            0
        ].keys()

        for key in keys:
            stacked = torch.stack(
                [
                    state[key]
                    .float()
                    for state
                    in worker_states
                ],
                dim=0,
            )

            averaged[key] = (
                stacked.mean(
                    dim=0
                )
            )

        self.model.load_state_dict(
            averaged
        )

    def save_checkpoint(
        self,
        epoch,
        avg_loss,
    ):
        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    self.model
                    .state_dict(),

                "average_worker_loss":
                    avg_loss,

                "num_workers":
                    self.num_workers,
            },
            GLOBAL_MODEL_PATH,
        )

    def start(self):
        print("=" * 70)
        print(
            "S.H.A ANC PARAMETER SERVER"
        )
        print("=" * 70)

        print(
            f"Port       : "
            f"{self.port}"
        )

        print(
            f"Workers    : "
            f"{self.num_workers}"
        )

        print(
            f"Epochs     : "
            f"{self.epochs}"
        )

        print("=" * 70)

        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server.bind(
            (
                "0.0.0.0",
                self.port,
            )
        )

        server.listen(
            self.num_workers
        )

        print(
            f"Waiting for "
            f"{self.num_workers} "
            f"workers..."
        )

        for worker_id in range(
            1,
            self.num_workers + 1,
        ):
            connection, address = (
                server.accept()
            )

            connection.settimeout(
                900
            )

            self.workers[
                worker_id
            ] = connection

            print(
                f"Worker "
                f"{worker_id} "
                f"connected from "
                f"{address}"
            )

            thread = threading.Thread(
                target=
                    self.handle_worker,
                args=(
                    worker_id,
                    connection,
                ),
                daemon=True,
            )

            thread.start()

        print()
        print(
            "ALL WORKERS CONNECTED"
        )

        best_loss = float("inf")

        for epoch in range(
            1,
            self.epochs + 1,
        ):
            print()
            print("=" * 70)

            print(
                f"EPOCH "
                f"{epoch}/"
                f"{self.epochs}"
            )

            print("=" * 70)

            print(
                "Broadcasting "
                "global ANC model..."
            )

            self.broadcast_model()

            print(
                "Waiting for "
                "worker updates..."
            )

            updates = (
                self.wait_for_updates()
            )

            losses = [
                float(
                    update["loss"]
                )
                for update
                in updates
            ]

            avg_loss = (
                sum(losses)
                / len(losses)
            )

            for update in updates:
                print(
                    f"Worker "
                    f"{update['worker_id']} "
                    f"loss = "
                    f"{update['loss']:.6f}"
                )

            print(
                f"Average loss = "
                f"{avg_loss:.6f}"
            )

            print(
                "Averaging worker "
                "models..."
            )

            self.average_models(
                updates
            )

            self.save_checkpoint(
                epoch,
                avg_loss,
            )

            if avg_loss < best_loss:
                best_loss = avg_loss

                print(
                    "New best global "
                    "model."
                )

        print()
        print("=" * 70)
        print(
            "S.H.A ANC DISTRIBUTED "
            "TRAINING COMPLETE"
        )
        print("=" * 70)

        print(
            f"Final model: "
            f"{GLOBAL_MODEL_PATH}"
        )

        for sock in (
            self.workers.values()
        ):
            sock.close()

        server.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "S.H.A ANC parameter server"
        )
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
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

    args = parser.parse_args()

    server = SHAANCParameterServer(
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
    )

    server.start()


if __name__ == "__main__":
    main()