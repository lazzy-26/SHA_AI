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
    MIN_IMPROVEMENT,
    PATIENCE,
)

from dataset import get_local_loaders
from model import create_model


# ============================================================
# S.H.A ANC FEDERATED WORKER
# ============================================================

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
        min_improvement=MIN_IMPROVEMENT,
        patience=PATIENCE,
    ):

        self.server_ip = server_ip
        self.port = port

        self.worker_id = worker_id
        self.num_workers = num_workers

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.min_improvement = (
            float(min_improvement)
        )

        self.patience = max(
            int(patience),
            1,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = create_model()

        self.criterion = nn.L1Loss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        self.socket = None

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        self.train_loader = None
        self.validation_loader = None

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        self.best_validation_loss = float(
            "inf"
        )

        self.previous_validation_loss = None

        self.epochs_without_improvement = 0

        self.converged = False

        self.convergence_epoch = None

        # ----------------------------------------------------
        # Last training result
        # ----------------------------------------------------

        self.last_loss = float(
            "inf"
        )

        self.last_validation_loss = float(
            "inf"
        )

    # ========================================================
    # NETWORK
    # ========================================================

    @staticmethod
    def send_bytes(
        sock,
        data,
    ):

        header = len(data).to_bytes(
            8,
            "big",
        )

        sock.sendall(
            header
        )

        sock.sendall(
            data
        )

    def recv_exact(
        self,
        size,
    ):

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

            buffer.extend(
                chunk
            )

        return bytes(
            buffer
        )

    def recv_bytes(self):

        header = self.recv_exact(
            8
        )

        size = int.from_bytes(
            header,
            "big",
        )

        if size <= 0:

            raise ConnectionError(
                "Invalid payload size."
            )

        return self.recv_exact(
            size
        )

    # ========================================================
    # CONNECT
    # ========================================================

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
                    1800
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

                if self.socket:

                    try:
                        self.socket.close()
                    except OSError:
                        pass

                    self.socket = None

                time.sleep(
                    delay
                )

        raise RuntimeError(
            f"Worker {self.worker_id} "
            "could not connect to server."
        )

    # ========================================================
    # DATASET
    # ========================================================

    def prepare_dataset(self):

        if self.train_loader is not None:

            return

        (
            self.train_loader,
            self.validation_loader,
        ) = get_local_loaders(
            participant_id=(
                f"Worker {self.worker_id}"
            ),
            batch_size=self.batch_size,
        )

    # ========================================================
    # RECEIVE GLOBAL MODEL
    # ========================================================

    def receive_weights(self):

        payload = self.recv_bytes()

        weights = pickle.loads(
            payload
        )

        self.model.load_state_dict(
            weights
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train_local_epoch(self):

        self.model.train()

        total_loss = 0.0
        batches = 0

        start_time = time.time()

        for noisy, clean in self.train_loader:

            self.optimizer.zero_grad(
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

            self.optimizer.step()

            total_loss += float(
                loss.item()
            )

            batches += 1

        elapsed = (
            time.time()
            - start_time
        )

        average_loss = (
            total_loss
            / max(
                batches,
                1,
            )
        )

        return (
            average_loss,
            elapsed,
            batches,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        batches = 0

        for noisy, clean in (
            self.validation_loader
        ):

            enhanced = self.model(
                noisy
            )

            loss = self.criterion(
                enhanced,
                clean,
            )

            total_loss += float(
                loss.item()
            )

            batches += 1

        return (
            total_loss
            / max(
                batches,
                1,
            )
        )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    def check_convergence(
        self,
        validation_loss,
        epoch,
    ):

        if (
            self.previous_validation_loss
            is None
        ):

            self.previous_validation_loss = (
                validation_loss
            )

            self.best_validation_loss = (
                validation_loss
            )

            return False

        improvement = (
            self.best_validation_loss
            - validation_loss
        )

        if (
            validation_loss
            < self.best_validation_loss
        ):

            self.best_validation_loss = (
                validation_loss
            )

        if improvement >= (
            self.min_improvement
        ):

            self.epochs_without_improvement = 0

        else:

            self.epochs_without_improvement += 1

        self.previous_validation_loss = (
            validation_loss
        )

        if (
            self.epochs_without_improvement
            >= self.patience
        ):

            self.converged = True

            self.convergence_epoch = (
                epoch
            )

            return True

        return False

    # ========================================================
    # SEND UPDATE
    # ========================================================

    def send_update(
        self,
        loss,
        validation_loss,
        elapsed,
        batches,
    ):

        weights = {
            key:
                value.detach().cpu()
            for key, value
            in self.model.state_dict().items()
        }

        status = (
            "CONVERGED"
            if self.converged
            else "TRAINING"
        )

        payload = {

            "participant_type":
                "worker",

            "worker_id":
                self.worker_id,

            "participant_id":
                self.worker_id,

            "loss":
                float(loss),

            "validation_loss":
                float(validation_loss),

            "status":
                status,

            "epoch":
                self.convergence_epoch,

            "training_time":
                float(elapsed),

            "batches":
                int(batches),

            "num_samples":
                len(
                    self.train_loader.dataset
                ),

            "weights":
                weights,
        }

        self.send_bytes(
            self.socket,
            pickle.dumps(
                payload,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )

    # ========================================================
    # TRAINING
    # ========================================================

    def train(self):

        self.prepare_dataset()

        self.connect()

        print()
        print(
            "=" * 70
        )

        print(
            f"S.H.A ANC WORKER {self.worker_id}"
        )

        print(
            "=" * 70
        )

        print(
            f"Server          : "
            f"{self.server_ip}:{self.port}"
        )

        print(
            f"Worker ID       : "
            f"{self.worker_id}"
        )

        print(
            f"Remote workers  : "
            f"{self.num_workers}"
        )

        print(
            f"Maximum epochs  : "
            f"{self.epochs}"
        )

        print(
            f"Learning rate   : "
            f"{self.learning_rate}"
        )

        print(
            f"Batch size      : "
            f"{self.batch_size}"
        )

        print(
            f"Min improvement : "
            f"{self.min_improvement}"
        )

        print(
            f"Patience        : "
            f"{self.patience}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Global rounds
        # ----------------------------------------------------

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            print()
            print(
                "=" * 70
            )

            print(
                f"[Worker {self.worker_id}] "
                f"GLOBAL ROUND "
                f"{epoch}/{self.epochs}"
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # Receive global model
            # ------------------------------------------------

            print(
                "Receiving global model..."
            )

            self.receive_weights()

            # ------------------------------------------------
            # If already converged
            # ------------------------------------------------

            if self.converged:

                print(
                    "Worker has converged."
                )

                print(
                    "Skipping local training."
                )

                self.send_update(
                    loss=self.last_loss,
                    validation_loss=(
                        self.last_validation_loss
                    ),
                    elapsed=0.0,
                    batches=0,
                )

                continue

            # ------------------------------------------------
            # Local training
            # ------------------------------------------------

            print(
                "Training local LibriSpeech/MUSAN data..."
            )

            (
                loss,
                elapsed,
                batches,
            ) = self.train_local_epoch()

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            validation_loss = (
                self.validate()
            )

            self.last_loss = loss

            self.last_validation_loss = (
                validation_loss
            )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            reached_convergence = (
                self.check_convergence(
                    validation_loss,
                    epoch,
                )
            )

            print()
            print(
                f"Train loss       : "
                f"{loss:.6f}"
            )

            print(
                f"Validation loss   : "
                f"{validation_loss:.6f}"
            )

            print(
                f"Best validation  : "
                f"{self.best_validation_loss:.6f}"
            )

            print(
                f"Patience          : "
                f"{self.epochs_without_improvement}/"
                f"{self.patience}"
            )

            print(
                f"Epoch time       : "
                f"{elapsed:.2f}s"
            )

            # ------------------------------------------------
            # Convergence announcement
            # ------------------------------------------------

            if reached_convergence:

                print()
                print(
                    "-" * 70
                )

                print(
                    f"[Worker {self.worker_id}] "
                    "EARLY STOPPING"
                )

                print(
                    f"Validation loss "
                    f"stopped improving."
                )

                print(
                    f"Converged at round "
                    f"{epoch}."
                )

                print(
                    "-" * 70
                )

            # ------------------------------------------------
            # Send update
            # ------------------------------------------------

            print(
                "Sending local model update..."
            )

            self.send_update(
                loss=loss,
                validation_loss=(
                    validation_loss
                ),
                elapsed=elapsed,
                batches=batches,
            )

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            f"[Worker {self.worker_id}] "
            "TRAINING FINISHED"
        )

        print(
            "=" * 70
        )

        if self.converged:

            print(
                f"Converged at round: "
                f"{self.convergence_epoch}"
            )

            print(
                f"Best validation loss: "
                f"{self.best_validation_loss:.6f}"
            )

        else:

            print(
                "Maximum epochs reached."
            )

        print(
            "=" * 70
        )

        if self.socket:

            try:
                self.socket.close()
            except OSError:
                pass

            self.socket = None


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "S.H.A ANC federated worker"
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

    parser.add_argument(
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
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
        min_improvement=args.min_improvement,
        patience=args.patience,
    )

    worker.train()


if __name__ == "__main__":
    main()