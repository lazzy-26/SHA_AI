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
    GRADIENT_ACCUMULATION_STEPS,
    HOST_IP,
    LEARNING_RATE,
    MIN_IMPROVEMENT,
    PATIENCE,
    READY_TIMEOUT,
)

from dataset import get_local_loaders
from model import create_model


# ============================================================
# FEDERATED WORKER
# ============================================================

class SHAANCWorker:

    def __init__(
        self,
        server_ip,
        port,
        worker_id,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        min_improvement=MIN_IMPROVEMENT,
        patience=PATIENCE,
        grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,
        max_samples=None,
        connect_retries=30,
    ):

        # ----------------------------------------------------
        # CONFIGURATION
        # ----------------------------------------------------

        self.server_ip = server_ip
        self.port = port
        self.worker_id = int(worker_id)

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        self.learning_rate = float(
            learning_rate
        )

        self.min_improvement = float(
            min_improvement
        )

        self.patience = max(
            int(patience),
            1,
        )

        self.grad_accum_steps = max(
            int(grad_accum_steps),
            1,
        )

        self.max_samples = max_samples

        self.connect_retries = max(
            int(connect_retries),
            1,
        )

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = create_model().to(
            self.device
        )

        self.criterion = nn.L1Loss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        self.train_loader = None
        self.validation_loader = None

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        self.best_validation_loss = float(
            "inf"
        )

        self.epochs_without_improvement = 0

        self.converged = False

        self.convergence_epoch = None

        # ----------------------------------------------------
        # NETWORK
        # ----------------------------------------------------

        self.sock = None

        self.training_complete = False

    # ========================================================
    # NETWORK
    # ========================================================

    @staticmethod
    def send_bytes(sock, data):

        header = len(data).to_bytes(
            8,
            "big",
        )

        sock.sendall(header)

        sock.sendall(data)

    @staticmethod
    def recv_exact(sock, size):

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
                    "Server closed the connection."
                )

            buffer.extend(chunk)

        return bytes(buffer)

    @classmethod
    def recv_bytes(cls, sock):

        header = cls.recv_exact(
            sock,
            8,
        )

        size = int.from_bytes(
            header,
            "big",
        )

        if size <= 0:
            raise ConnectionError(
                "Invalid payload size."
            )

        return cls.recv_exact(
            sock,
            size,
        )

    def send_message(self, message):

        payload = pickle.dumps(
            message,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        self.send_bytes(
            self.sock,
            payload,
        )

    def receive_message(self):

        payload = self.recv_bytes(
            self.sock
        )

        return pickle.loads(
            payload
        )

    # ========================================================
    # CONNECT TO SERVER
    # ========================================================

    def connect(self):

        print()
        print("=" * 70)
        print(
            f"[Worker {self.worker_id}] "
            "CONNECTING TO PARAMETER SERVER"
        )
        print("=" * 70)

        print(
            f"[Worker {self.worker_id}] "
            f"Server: {self.server_ip}:{self.port}"
        )

        last_error = None

        for attempt in range(
            1,
            self.connect_retries + 1,
        ):

            try:

                sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )

                # IMPORTANT:
                # Persistent connection.
                # No socket read timeout.
                sock.settimeout(None)

                sock.connect(
                    (
                        self.server_ip,
                        self.port,
                    )
                )

                self.sock = sock

                print(
                    f"[Worker {self.worker_id}] "
                    f"Connected successfully."
                )

                return

            except OSError as error:

                last_error = error

                print(
                    f"[Worker {self.worker_id}] "
                    f"Connection attempt "
                    f"{attempt}/{self.connect_retries} "
                    f"failed: {error}"
                )

                time.sleep(2)

        raise ConnectionError(
            f"Could not connect to server "
            f"{self.server_ip}:{self.port}. "
            f"Last error: {last_error}"
        )

    # ========================================================
    # PREPARE LOCAL DATA
    # ========================================================

    def prepare_dataset(self):

        print()
        print("=" * 70)
        print(
            f"[Worker {self.worker_id}] "
            "PREPARING LOCAL DATASET"
        )
        print("=" * 70)

        print(
            f"[Worker {self.worker_id}] "
            "Using LOCAL LibriSpeech + MUSAN data."
        )

        (
            self.train_loader,
            self.validation_loader,
        ) = get_local_loaders(
            participant_id=f"WORKER {self.worker_id}",
            batch_size=self.batch_size,
            max_samples=self.max_samples,
        )

        print()
        print(
            f"[Worker {self.worker_id}] "
            "Local dataset prepared."
        )

        print(
            f"[Worker {self.worker_id}] "
            f"Train batches: {len(self.train_loader)}"
        )

        print(
            f"[Worker {self.worker_id}] "
            f"Validation batches: "
            f"{len(self.validation_loader)}"
        )

    # ========================================================
    # READY HANDSHAKE
    # ========================================================

    def send_ready(self):

        self.send_message(
            {
                "type": "READY",
                "worker_id": self.worker_id,
            }
        )

        print()
        print(
            f"[Worker {self.worker_id}] "
            "READY sent to server."
        )

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    def load_global_model(
        self,
        weights,
    ):

        state = {
            key: value.to(
                self.device
            )
            for key, value in weights.items()
        }

        self.model.load_state_dict(
            state,
            strict=True,
        )

    # ========================================================
    # TRAIN LOCAL EPOCH
    # ========================================================

    def train_local_epoch(self):

        if self.train_loader is None:
            raise RuntimeError(
                "Training loader has not been prepared."
            )

        self.model.train()

        total_loss = 0.0
        batches = 0

        start_time = time.time()

        accumulation_steps = (
            self.grad_accum_steps
        )

        use_amp = (
            self.device.type == "cuda"
        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for batch_idx, (
            noisy,
            clean,
        ) in enumerate(
            self.train_loader
        ):

            noisy = noisy.to(
                self.device,
                non_blocking=True,
            )

            clean = clean.to(
                self.device,
                non_blocking=True,
            )

            if use_amp:

                with torch.cuda.amp.autocast():

                    enhanced = self.model(
                        noisy
                    )

                    raw_loss = self.criterion(
                        enhanced,
                        clean,
                    )

                    loss = (
                        raw_loss
                        / accumulation_steps
                    )

                scaler.scale(
                    loss
                ).backward()

            else:

                enhanced = self.model(
                    noisy
                )

                raw_loss = self.criterion(
                    enhanced,
                    clean,
                )

                loss = (
                    raw_loss
                    / accumulation_steps
                )

                loss.backward()

            # ------------------------------------------------
            # OPTIMIZER STEP
            # ------------------------------------------------

            is_accumulation_boundary = (
                (batch_idx + 1)
                % accumulation_steps
                == 0
            )

            is_last_batch = (
                batch_idx + 1
                == len(self.train_loader)
            )

            if (
                is_accumulation_boundary
                or is_last_batch
            ):

                if use_amp:

                    scaler.unscale_(
                        self.optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=5.0,
                    )

                    scaler.step(
                        self.optimizer
                    )

                    scaler.update()

                else:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=5.0,
                    )

                    self.optimizer.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            total_loss += float(
                raw_loss.item()
            )

            batches += 1

        elapsed = (
            time.time()
            - start_time
        )

        average_loss = (
            total_loss
            / max(batches, 1)
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

        if self.validation_loader is None:
            return 0.0

        self.model.eval()

        total_loss = 0.0
        batches = 0

        for noisy, clean in (
            self.validation_loader
        ):

            noisy = noisy.to(
                self.device,
                non_blocking=True,
            )

            clean = clean.to(
                self.device,
                non_blocking=True,
            )

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
            / max(batches, 1)
        )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    def check_convergence(
        self,
        validation_loss,
        epoch,
    ):

        improvement = (
            self.best_validation_loss
            - validation_loss
        )

        if validation_loss < (
            self.best_validation_loss
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

        if (
            self.epochs_without_improvement
            >= self.patience
        ):

            self.converged = True

            self.convergence_epoch = epoch

            return True

        return False

    # ========================================================
    # MODEL STATE
    # ========================================================

    def model_state_dict(self):

        return {
            key: value.detach().cpu()
            for key, value
            in self.model.state_dict().items()
        }

    # ========================================================
    # SEND ROUND UPDATE
    # ========================================================

    def send_update(
        self,
        epoch,
        loss,
        validation_loss,
        elapsed,
        batches,
    ):

        status = (
            "CONVERGED"
            if self.converged
            else "TRAINING"
        )

        message = {
            "type": "UPDATE",
            "worker_id": self.worker_id,
            "epoch": epoch,
            "round": epoch,
            "loss": float(loss),
            "validation_loss": float(
                validation_loss
            ),
            "status": status,
            "training_time": float(
                elapsed
            ),
            "batches": int(batches),
            "weights": self.model_state_dict(),
        }

        self.send_message(
            message
        )

    # ========================================================
    # MAIN TRAINING LOOP
    # ========================================================

    def run(self):

        print()
        print("=" * 70)
        print(
            f"S.H.A ANC WORKER {self.worker_id}"
        )
        print("=" * 70)

        print(
            f"Device       : {self.device}"
        )

        print(
            f"Server       : "
            f"{self.server_ip}:{self.port}"
        )

        print(
            f"Epochs       : {self.epochs}"
        )

        print(
            f"Batch size   : {self.batch_size}"
        )

        print(
            f"Learning rate: {self.learning_rate}"
        )

        print(
            f"Patience     : {self.patience}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # LOCAL DATA FIRST
        # ----------------------------------------------------

        self.prepare_dataset()

        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        self.connect()

        # ----------------------------------------------------
        # READY
        # ----------------------------------------------------

        self.send_ready()

        # ----------------------------------------------------
        # WAIT FOR GLOBAL MODELS
        # ----------------------------------------------------

        try:

            while not self.training_complete:

                message = self.receive_message()

                message_type = message.get(
                    "type"
                )

                # ============================================
                # GLOBAL MODEL
                # ============================================

                if message_type == "GLOBAL_MODEL":

                    epoch = int(
                        message.get(
                            "epoch",
                            0,
                        )
                    )

                    print()
                    print("=" * 70)
                    print(
                        f"[Worker {self.worker_id}] "
                        f"GLOBAL ROUND "
                        f"{epoch}/{self.epochs}"
                    )
                    print("=" * 70)

                    print(
                        f"[Worker {self.worker_id}] "
                        "Receiving global model..."
                    )

                    self.load_global_model(
                        message["weights"]
                    )

                    # ========================================
                    # LOCAL TRAINING
                    # ========================================

                    if self.converged:

                        print(
                            f"[Worker {self.worker_id}] "
                            "Already converged."
                        )

                        print(
                            f"[Worker {self.worker_id}] "
                            "Skipping local training."
                        )

                        validation_loss = (
                            self.validate()
                        )

                        loss = (
                            self.best_validation_loss
                        )

                        elapsed = 0.0
                        batches = 0

                    else:

                        print(
                            f"[Worker {self.worker_id}] "
                            "LOCAL TRAINING"
                        )

                        (
                            loss,
                            elapsed,
                            batches,
                        ) = self.train_local_epoch()

                        validation_loss = (
                            self.validate()
                        )

                        converged = (
                            self.check_convergence(
                                validation_loss,
                                epoch,
                            )
                        )

                        if converged:

                            print()
                            print(
                                f"[Worker {self.worker_id}] "
                                "EARLY STOPPING"
                            )

                            print(
                                f"[Worker {self.worker_id}] "
                                f"Converged at "
                                f"round {epoch}."
                            )

                    # ========================================
                    # RESULTS
                    # ========================================

                    status = (
                        "CONVERGED"
                        if self.converged
                        else "TRAINING"
                    )

                    print(
                        f"[Worker {self.worker_id}] "
                        f"train={loss:.6f} | "
                        f"validation="
                        f"{validation_loss:.6f} | "
                        f"time={elapsed:.2f}s | "
                        f"status={status}"
                    )

                    # ========================================
                    # SEND UPDATE
                    # ========================================

                    self.send_update(
                        epoch=epoch,
                        loss=loss,
                        validation_loss=validation_loss,
                        elapsed=elapsed,
                        batches=batches,
                    )

                    print(
                        f"[Worker {self.worker_id}] "
                        f"Round {epoch} update sent."
                    )

                # ============================================
                # TRAINING COMPLETE
                # ============================================

                elif message_type == "TRAINING_COMPLETE":

                    print()
                    print("=" * 70)
                    print(
                        f"[Worker {self.worker_id}] "
                        "TRAINING COMPLETE"
                    )
                    print("=" * 70)

                    self.training_complete = True

                    break

                else:

                    print(
                        f"[Worker {self.worker_id}] "
                        f"Unknown message: "
                        f"{message_type}"
                    )

        except KeyboardInterrupt:

            print()
            print(
                f"[Worker {self.worker_id}] "
                "Interrupted by user."
            )

        except Exception as error:

            print()
            print(
                f"[Worker {self.worker_id}] "
                f"ERROR: {error}"
            )

            raise

        finally:

            try:
                self.send_message(
                    {
                        "type": "FINISHED",
                        "worker_id": self.worker_id,
                    }
                )
            except Exception:
                pass

            if self.sock is not None:

                try:
                    self.sock.close()
                except Exception:
                    pass

            print(
                f"[Worker {self.worker_id}] "
                "Connection closed."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "S.H.A ANC federated training worker"
        )
    )

    parser.add_argument(
        "--server-ip",
        type=str,
        default=HOST_IP,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )

    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Worker ID, e.g. 1 or 2.",
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
        "--grad-accum-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Limit local clean files for testing."
        ),
    )

    args = parser.parse_args()

    worker = SHAANCWorker(
        server_ip=args.server_ip,
        port=args.port,
        worker_id=args.worker_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_improvement=args.min_improvement,
        patience=args.patience,
        grad_accum_steps=args.grad_accum_steps,
        max_samples=args.max_samples,
    )

    worker.run()


if __name__ == "__main__":
    main()