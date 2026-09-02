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
# TCP KEEPALIVE HELPERS
# ============================================================

def enable_keepalive(sock, idle_seconds=20, interval_seconds=10, max_probes=5):
    """
    Enable TCP keepalive on a socket to prevent NAT/routers from
    silently dropping idle connections.

    Args:
        sock: The socket to configure
        idle_seconds: Seconds of idleness before sending first probe
        interval_seconds: Seconds between subsequent probes
        max_probes: Number of unacknowledged probes before declaring dead
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # Linux-specific keepalive tuning (works on most systems)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_seconds)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_seconds)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_probes)
    except (AttributeError, OSError):
        pass  # Not all systems support these options

    # Windows-specific keepalive tuning
    try:
        if hasattr(socket, "SIO_KEEPALIVE_VALS"):
            sock.ioctl(
                socket.SIO_KEEPALIVE_VALS,
                (1, idle_seconds * 1000, interval_seconds * 1000),
            )
    except (AttributeError, OSError):
        pass


# ============================================================
# S.H.A ANC FEDERATED WORKER
#
# Every worker owns its own local:
#
#     VoiceBank-DEMAND (paired clean/noisy recordings)
#
# Only model weights and metrics are transmitted.
#
# Worker protocol:
#
#     CONNECT
#       ↓
#     PREPARE LOCAL DATASET
#       ↓
#     READY
#       ↓
#     WAIT FOR GLOBAL MODEL
#       ↓
#     LOCAL TRAINING
#       ↓
#     VALIDATION
#       ↓
#     SEND UPDATE
#       ↓
#     NEXT ROUND
#
# Participant early stopping is supported.
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
        max_samples=None,  # NEW: limit dataset size for fast testing
    ):

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

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

        self.max_samples = max_samples  # NEW

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = create_model()

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.criterion = nn.L1Loss()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

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
                    "Server connection closed."
                )

            buffer.extend(
                chunk
            )

        return bytes(
            buffer
        )

    def recv_bytes(
        self,
    ):

        header = self.recv_exact(
            self.socket,
            8,
        )

        size = int.from_bytes(
            header,
            "big",
        )

        if size <= 0:

            raise ConnectionError(
                "Invalid server payload."
            )

        return self.recv_exact(
            self.socket,
            size,
        )

    # ========================================================
    # SEND MESSAGE
    # ========================================================

    def send_message(
        self,
        message,
    ):

        payload = pickle.dumps(
            message,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        self.send_bytes(
            self.socket,
            payload,
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

                # ==== ENABLE TCP KEEPALIVE RIGHT AFTER CONNECT ====
                enable_keepalive(
                    self.socket,
                    idle_seconds=20,
                    interval_seconds=10,
                    max_probes=5,
                )
                # ==================================================

                print()
                print(
                    f"[Worker {self.worker_id}] "
                    f"Connected to "
                    f"{self.server_ip}:"
                    f"{self.port}"
                )
                print(
                    f"[Worker {self.worker_id}] "
                    "TCP keepalive enabled (idle=20s, interval=10s)"
                )

                return

            except OSError as error:

                print(
                    f"[Worker {self.worker_id}] "
                    f"Waiting for server "
                    f"({attempt}/"
                    f"{max_retries})..."
                )

                if self.socket is not None:

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
    # PREPARE LOCAL DATASET
    # ========================================================

    def prepare_dataset(self):

        print()
        print(
            "=" * 70
        )

        print(
            f"[Worker {self.worker_id}] "
            "PREPARING LOCAL DATASET"
        )

        print(
            "=" * 70
        )

        print(
            f"[Worker {self.worker_id}] "
            "Using local VoiceBank-DEMAND dataset."
        )

        if self.max_samples is not None:
            print(
                f"[Worker {self.worker_id}] "
                f"LIMITING to {self.max_samples} samples "
                "(for fast testing)"
            )

        (
            self.train_loader,
            self.validation_loader,
        ) = get_local_loaders(
            participant_id=(
                f"WORKER_{self.worker_id}"
            ),
            batch_size=self.batch_size,
            max_samples=self.max_samples,  # NEW: pass through
        )

        print()
        print(
            f"[Worker {self.worker_id}] "
            "LOCAL DATASET READY"
        )

        print(
            f"[Worker {self.worker_id}] "
            "VoiceBank-DEMAND paths are local."
        )

    # ========================================================
    # READY HANDSHAKE
    # ========================================================

    def send_ready(self):

        print()
        print(
            f"[Worker {self.worker_id}] "
            "Sending READY to server..."
        )

        self.send_message(
            {
                "type":
                    "READY",

                "worker_id":
                    self.worker_id,
            }
        )

        print(
            f"[Worker {self.worker_id}] "
            "READY sent."
        )

    # ========================================================
    # RECEIVE GLOBAL MODEL
    # ========================================================

    def receive_global_model(
        self,
    ):

        print()
        print(
            f"[Worker {self.worker_id}] "
            "Receiving global model..."
        )

        payload = self.recv_bytes()

        message = pickle.loads(
            payload
        )

        message_type = message.get(
            "type"
        )

        if message_type == "TRAINING_COMPLETE":

            return False

        if message_type != "GLOBAL_MODEL":

            raise RuntimeError(
                f"Unexpected server message: "
                f"{message_type}"
            )

        weights = message.get(
            "weights"
        )

        if weights is None:

            raise RuntimeError(
                "Global model contains no weights."
            )

        self.model.load_state_dict(
            weights
        )

        print(
            f"[Worker {self.worker_id}] "
            "Global model received."
        )

        return True

    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def train_local_epoch(
        self,
    ):

        if self.train_loader is None:

            raise RuntimeError(
                "Training loader has not been prepared."
            )

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
    def validate(
        self,
    ):

        if self.validation_loader is None:

            return 0.0

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

        # ----------------------------------------------------
        # First validation measurement
        # ----------------------------------------------------

        if self.previous_validation_loss is None:

            self.previous_validation_loss = (
                validation_loss
            )

            self.best_validation_loss = (
                validation_loss
            )

            return False

        # ----------------------------------------------------
        # Improvement from previous round
        # ----------------------------------------------------

        improvement = (
            self.previous_validation_loss
            - validation_loss
        )

        # ----------------------------------------------------
        # Best validation loss
        # ----------------------------------------------------

        if validation_loss < (
            self.best_validation_loss
        ):

            self.best_validation_loss = (
                validation_loss
            )

        # ----------------------------------------------------
        # Patience
        # ----------------------------------------------------

        if improvement >= (
            self.min_improvement
        ):

            self.epochs_without_improvement = 0

        else:

            self.epochs_without_improvement += 1

        self.previous_validation_loss = (
            validation_loss
        )

        # ----------------------------------------------------
        # Converged
        # ----------------------------------------------------

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

        message = {

            "type":
                "UPDATE",

            "worker_id":
                self.worker_id,

            "epoch":
                self.convergence_epoch,

            "loss":
                float(loss),

            "validation_loss":
                float(validation_loss),

            "status":
                status,

            "training_time":
                float(elapsed),

            "batches":
                int(batches),

            "weights":
                weights,
        }

        self.send_message(
            message
        )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    def print_performance(
        self,
        epoch,
        loss,
        validation_loss,
        elapsed,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"[Worker {self.worker_id}] "
            f"ROUND {epoch} PERFORMANCE"
        )

        print(
            f"  Training loss      : "
            f"{loss:.6f}"
        )

        print(
            f"  Validation loss    : "
            f"{validation_loss:.6f}"
        )

        print(
            f"  Best validation    : "
            f"{self.best_validation_loss:.6f}"
        )

        print(
            f"  Patience           : "
            f"{self.epochs_without_improvement}/"
            f"{self.patience}"
        )

        print(
            f"  Training time      : "
            f"{elapsed:.2f}s"
        )

        print(
            f"  Status             : "
            f"{'CONVERGED' if self.converged else 'TRAINING'}"
        )

        print(
            "-" * 70
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        self.connect()

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        self.prepare_dataset()

        # ----------------------------------------------------
        # READY
        # ----------------------------------------------------

        self.send_ready()

        print()
        print(
            "=" * 70
        )

        print(
            f"S.H.A ANC WORKER "
            f"{self.worker_id}"
        )

        print(
            "=" * 70
        )

        print(
            f"Server          : "
            f"{self.server_ip}:"
            f"{self.port}"
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
            f"Total participants: "
            f"{self.num_workers + 1}"
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
            f"Min improvement: "
            f"{self.min_improvement}"
        )

        print(
            f"Patience        : "
            f"{self.patience}"
        )

        if self.max_samples is not None:
            print(
                f"Max samples     : "
                f"{self.max_samples} (TESTING MODE)"
            )

        print(
            "=" * 70
        )

        # ====================================================
        # GLOBAL FEDERATED ROUNDS
        # ====================================================

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
            # RECEIVE GLOBAL MODEL
            # ------------------------------------------------

            try:

                received = (
                    self.receive_global_model()
                )

            except Exception as error:

                print(
                    f"[Worker {self.worker_id}] "
                    f"Failed receiving model: "
                    f"{error}"
                )

                break

            if not received:

                print(
                    f"[Worker {self.worker_id}] "
                    "Server ended training."
                )

                break

            # ------------------------------------------------
            # CONVERGED WORKER
            # ------------------------------------------------

            if self.converged:

                print(
                    f"[Worker {self.worker_id}] "
                    "Participant has converged."
                )

                print(
                    f"[Worker {self.worker_id}] "
                    "Skipping local training."
                )

                validation_loss = (
                    self.validate()
                )

                self.send_update(
                    loss=(
                        self.best_validation_loss
                    ),
                    validation_loss=(
                        validation_loss
                    ),
                    elapsed=0.0,
                    batches=0,
                )

                continue

            # ------------------------------------------------
            # LOCAL TRAINING
            # ------------------------------------------------

            print()
            print(
                f"[Worker {self.worker_id}] "
                "Training local "
                "VoiceBank-DEMAND data..."
            )

            try:

                (
                    loss,
                    elapsed,
                    batches,
                ) = self.train_local_epoch()

            except Exception as error:

                print(
                    f"[Worker {self.worker_id}] "
                    f"Training failed: "
                    f"{error}"
                )

                break

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            validation_loss = (
                self.validate()
            )

            # ------------------------------------------------
            # EARLY STOPPING
            # ------------------------------------------------

            reached_convergence = (
                self.check_convergence(
                    validation_loss,
                    epoch,
                )
            )

            # ------------------------------------------------
            # PERFORMANCE
            # ------------------------------------------------

            self.print_performance(
                epoch=epoch,
                loss=loss,
                validation_loss=(
                    validation_loss
                ),
                elapsed=elapsed,
            )

            if reached_convergence:

                print()
                print(
                    "=" * 70
                )

                print(
                    f"[Worker {self.worker_id}] "
                    "EARLY STOPPING TRIGGERED"
                )

                print(
                    f"Converged at round "
                    f"{epoch}"
                )

                print(
                    f"Best validation loss: "
                    f"{self.best_validation_loss:.6f}"
                )

                print(
                    "=" * 70
                )

            # ------------------------------------------------
            # SEND UPDATE
            # ------------------------------------------------

            print()
            print(
                f"[Worker {self.worker_id}] "
                "Sending model update..."
            )

            try:

                self.send_update(
                    loss=loss,
                    validation_loss=(
                        validation_loss
                    ),
                    elapsed=elapsed,
                    batches=batches,
                )

            except Exception as error:

                print(
                    f"[Worker {self.worker_id}] "
                    f"Failed sending update: "
                    f"{error}"
                )

                break

        # ====================================================
        # FINISHED
        # ====================================================

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
                "Maximum training rounds "
                "completed."
            )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Notify server
        # ----------------------------------------------------

        try:

            self.send_message(
                {
                    "type":
                        "FINISHED",

                    "worker_id":
                        self.worker_id,
                }
            )

        except Exception:

            pass

        # ----------------------------------------------------
        # Close
        # ----------------------------------------------------

        if self.socket is not None:

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
            "S.H.A ANC federated worker "
            "with READY handshake and "
            "early stopping."
        )
    )

    parser.add_argument(
        "--server",
        required=True,
        help="Parameter server IP.",
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
        help="Worker ID.",
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

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N samples per participant (for fast testing).",
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

        min_improvement=(
            args.min_improvement
        ),

        patience=args.patience,

        max_samples=args.max_samples,  # NEW
    )

    worker.train()


if __name__ == "__main__":

    main()