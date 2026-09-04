import argparse
import pickle
import socket
import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE,
    DEFAULT_PORT,
    DEFAULT_WORKERS,
    EPOCHS,
    GLOBAL_MIN_IMPROVEMENT,
    GLOBAL_MODEL_PATH,
    GLOBAL_PATIENCE,
    LEARNING_RATE,
    MIN_IMPROVEMENT,
    NUM_PARTICIPANTS,
    PATIENCE,
    READY_TIMEOUT,  # NEW
    ROUND_TIMEOUT,  # NEW
    WORKER_SOCKET_TIMEOUT,  # NEW
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
    """
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_seconds)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_seconds)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_probes)
    except (AttributeError, OSError):
        pass

    try:
        if hasattr(socket, "SIO_KEEPALIVE_VALS"):
            sock.ioctl(
                socket.SIO_KEEPALIVE_VALS,
                (1, idle_seconds * 1000, interval_seconds * 1000),
            )
    except (AttributeError, OSError):
        pass


# ============================================================
# S.H.A ANC FEDERATED PARAMETER SERVER
# ============================================================

class SHAANCParameterServer:

    def __init__(
        self,
        port,
        num_workers,
        epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        min_improvement=MIN_IMPROVEMENT,
        participant_patience=PATIENCE,
        global_patience=GLOBAL_PATIENCE,
        ready_timeout=READY_TIMEOUT,  # NEW
        round_timeout=ROUND_TIMEOUT,  # NEW
        max_samples=None,
    ):

        # Configuration
        self.port = port
        self.num_workers = num_workers
        self.num_participants = num_workers + 1
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_improvement = float(min_improvement)
        self.participant_patience = max(int(participant_patience), 1)
        self.global_patience = max(int(global_patience), 1)
        self.ready_timeout = max(int(ready_timeout), 1)
        self.round_timeout = max(int(round_timeout), 1)  # NEW
        self.max_samples = max_samples

        # Global model
        self.model = create_model()
        self.criterion = nn.L1Loss()

        # Host optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Host dataset
        self.host_train_loader = None
        self.host_validation_loader = None

        # Network
        self.server_socket = None
        self.workers = {}

        # Worker state
        self.worker_ready = set()
        self.active_workers = set()
        self.stopped_workers = set()
        self.worker_connections = {}

        # Current round updates
        self.updates = {}

        # Synchronization
        self.lock = threading.Lock()
        self.ready_condition = threading.Condition(self.lock)
        self.update_condition = threading.Condition(self.lock)

        # Training state
        self.training_complete = False
        self.global_best_validation_loss = float("inf")
        self.global_epochs_without_improvement = 0

        # Host early stopping
        self.host_best_validation_loss = float("inf")
        self.host_epochs_without_improvement = 0
        self.host_converged = False
        self.host_convergence_epoch = None

    # ========================================================
    # NETWORK HELPERS
    # ========================================================

    @staticmethod
    def send_bytes(sock, data):
        header = len(data).to_bytes(8, "big")
        sock.sendall(header)
        sock.sendall(data)

    @staticmethod
    def recv_exact(sock, size):
        buffer = bytearray()
        while len(buffer) < size:
            chunk = sock.recv(min(65536, size - len(buffer)))
            if not chunk:
                raise ConnectionError("Connection closed.")
            buffer.extend(chunk)
        return bytes(buffer)

    @classmethod
    def recv_bytes(cls, sock):
        header = cls.recv_exact(sock, 8)
        size = int.from_bytes(header, "big")
        if size <= 0:
            raise ConnectionError("Invalid payload size.")
        return cls.recv_exact(sock, size)

    # ========================================================
    # SERIALIZATION
    # ========================================================

    @staticmethod
    def model_state_dict(model):
        return {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

    # ========================================================
    # PREPARE HOST DATASET
    # ========================================================

    def prepare_host_dataset(self):
        print()
        print("=" * 70)
        print("PREPARING HOST LOCAL DATASET")
        print("=" * 70)

        if self.max_samples is not None:
            print(
                f"[HOST] LIMITING to {self.max_samples} samples "
                "(for fast testing)"
            )
            # Auto-reduce epochs for testing
            if self.epochs > 5:
                print(
                    f"[HOST] ⚠️  TESTING MODE: Epochs reduced from {self.epochs} to 5"
                )
                self.epochs = min(self.epochs, 5)

        (self.host_train_loader, self.host_validation_loader) = get_local_loaders(
            participant_id="HOST",
            batch_size=self.batch_size,
            max_samples=self.max_samples,
        )

        print()
        print("[HOST] Local dataset ready.")
        print(f"[HOST] Train batches: {len(self.host_train_loader)}")
        print(f"[HOST] Val batches: {len(self.host_validation_loader)}")

    # ========================================================
    # HOST TRAINING (Optimized)
    # ========================================================

    def train_host_epoch(self):
        if self.host_train_loader is None:
            raise RuntimeError("Host dataset has not been prepared.")

        self.model.train()
        total_loss = 0.0
        batches = 0
        start_time = time.time()

        # Use mixed precision if CUDA available
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for noisy, clean in self.host_train_loader:
            if torch.cuda.is_available():
                noisy = noisy.cuda()
                clean = clean.cuda()

            self.optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    enhanced = self.model(noisy)
                    loss = self.criterion(enhanced, clean)
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                enhanced = self.model(noisy)
                loss = self.criterion(enhanced, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        elapsed = time.time() - start_time
        average_loss = total_loss / max(batches, 1)

        return average_loss, elapsed, batches

    # ========================================================
    # HOST VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate_host(self):
        if self.host_validation_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        batches = 0

        for noisy, clean in self.host_validation_loader:
            if torch.cuda.is_available():
                noisy = noisy.cuda()
                clean = clean.cuda()

            enhanced = self.model(noisy)
            loss = self.criterion(enhanced, clean)
            total_loss += float(loss.item())
            batches += 1

        return total_loss / max(batches, 1)

    # ========================================================
    # HOST EARLY STOPPING
    # ========================================================

    def check_host_convergence(self, validation_loss, epoch):
        improvement = self.host_best_validation_loss - validation_loss

        if validation_loss < self.host_best_validation_loss:
            self.host_best_validation_loss = validation_loss

        if improvement >= self.min_improvement:
            self.host_epochs_without_improvement = 0
        else:
            self.host_epochs_without_improvement += 1

        if self.host_epochs_without_improvement >= self.participant_patience:
            self.host_converged = True
            self.host_convergence_epoch = epoch
            return True

        return False

    # ========================================================
    # SEND MESSAGE TO WORKER
    # ========================================================

    def send_message(self, sock, message):
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        self.send_bytes(sock, payload)

    # ========================================================
    # BROADCAST GLOBAL MODEL
    # ========================================================

    def broadcast_model(self, epoch):
        state = self.model_state_dict(self.model)

        message = {
            "type": "GLOBAL_MODEL",
            "epoch": epoch,
            "weights": state,
        }

        with self.lock:
            workers_snapshot = {
                worker_id: sock
                for worker_id, sock in self.workers.items()
                if worker_id in self.active_workers
            }

        print(f"Sending global model to {len(workers_snapshot)} workers...")

        for worker_id, sock in workers_snapshot.items():
            try:
                self.send_message(sock, message)
            except Exception as error:
                print(f"[Server] Could not send global model to Worker {worker_id}: {error}")
                with self.lock:
                    self.active_workers.discard(worker_id)

    # ========================================================
    # WORKER CONNECTION HANDLER
    # ========================================================

    def handle_worker(self, worker_id, sock):
        try:
            while True:
                payload = self.recv_bytes(sock)
                message = pickle.loads(payload)
                message_type = message.get("type")

                if message_type == "READY":
                    with self.ready_condition:
                        self.worker_ready.add(worker_id)
                        self.active_workers.add(worker_id)
                        print()
                        print(f"[Server] Worker {worker_id} is READY.")
                        print(
                            f"[Server] Ready workers: "
                            f"{len(self.worker_ready)}/{self.num_workers}"
                        )
                        self.ready_condition.notify_all()

                elif message_type == "UPDATE":
                    with self.update_condition:
                        self.updates[worker_id] = message
                        status = message.get("status", "TRAINING")

                        if status == "CONVERGED":
                            self.stopped_workers.add(worker_id)
                            self.active_workers.discard(worker_id)
                        else:
                            self.active_workers.add(worker_id)

                        print()
                        print(f"[Server] Received Worker {worker_id} update.")
                        self.update_condition.notify_all()

                elif message_type == "FINISHED":
                    print(f"[Server] Worker {worker_id} reports training finished.")
                    with self.lock:
                        self.active_workers.discard(worker_id)

                else:
                    print(f"[Server] Unknown message from Worker {worker_id}: {message_type}")

        except Exception as error:
            print(f"[Server] Worker {worker_id} connection ended: {error}")
            with self.lock:
                self.active_workers.discard(worker_id)

    # ========================================================
    # WAIT FOR READY WORKERS
    # ========================================================

    def wait_for_workers_ready(self, timeout=None):
        if timeout is None:
            timeout = self.ready_timeout

        deadline = time.time() + timeout

        with self.ready_condition:
            while len(self.worker_ready) < self.num_workers:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for workers to become READY after {timeout}s."
                    )
                self.ready_condition.wait(timeout=min(remaining, 1.0))
                print(
                    f"\rReady workers: {len(self.worker_ready)}/{self.num_workers}",
                    end="",
                    flush=True,
                )

        print()

    # ========================================================
    # WAIT FOR ROUND UPDATES (FIXED)
    # ========================================================

    def wait_for_updates(self, epoch, timeout=None):
        if timeout is None:
            timeout = self.round_timeout  # FIXED: use configurable timeout

        deadline = time.time() + timeout

        while True:
            with self.update_condition:
                required_workers = set(self.worker_ready)
                received_workers = set(self.updates.keys())

                if received_workers >= required_workers:
                    return [
                        self.updates[worker_id]
                        for worker_id in sorted(required_workers)
                    ]

                remaining = deadline - time.time()
                if remaining <= 0:
                    missing = required_workers - received_workers
                    raise TimeoutError(
                        f"Timed out waiting for round {epoch} updates after {timeout}s. "
                        f"Missing workers: {sorted(missing)}"
                    )

                self.update_condition.wait(timeout=min(remaining, 1.0))

    # ========================================================
    # AGGREGATE MODELS
    # ========================================================

    def average_models(self, participant_states):
        if not participant_states:
            raise RuntimeError("No participant models available.")

        averaged = {}
        keys = participant_states[0].keys()

        for key in keys:
            tensors = []
            for state in participant_states:
                tensors.append(state[key].float())
            stacked = torch.stack(tensors, dim=0)
            averaged[key] = stacked.mean(dim=0)

        self.model.load_state_dict(averaged)

    # ========================================================
    # GLOBAL EARLY STOPPING
    # ========================================================

    def check_global_convergence(self, validation_loss):
        improvement = self.global_best_validation_loss - validation_loss

        if validation_loss < self.global_best_validation_loss:
            self.global_best_validation_loss = validation_loss

        if improvement >= GLOBAL_MIN_IMPROVEMENT:
            self.global_epochs_without_improvement = 0
        else:
            self.global_epochs_without_improvement += 1

        return self.global_epochs_without_improvement >= self.global_patience

    # ========================================================
    # PRINT UPDATE
    # ========================================================

    def print_worker_update(self, update):
        worker_id = update.get("worker_id", "?")
        loss = float(update.get("loss", 0.0))
        validation_loss = float(update.get("validation_loss", loss))
        status = update.get("status", "TRAINING")
        elapsed = float(update.get("training_time", 0.0))

        print(
            f"  Worker {worker_id}: "
            f"train={loss:.6f} | "
            f"validation={validation_loss:.6f} | "
            f"time={elapsed:.2f}s | "
            f"status={status}"
        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    def save_checkpoint(self, epoch, average_training_loss, average_validation_loss):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "average_training_loss": average_training_loss,
                "average_validation_loss": average_validation_loss,
                "num_workers": self.num_workers,
                "num_participants": self.num_participants,
                "stopped_workers": sorted(self.stopped_workers),
                "host_converged": self.host_converged,
                "host_convergence_epoch": self.host_convergence_epoch,
                "global_best_validation_loss": self.global_best_validation_loss,
            },
            GLOBAL_MODEL_PATH,
        )

    # ========================================================
    # START
    # ========================================================

    def start(self):
        # HOST DATASET
        self.prepare_host_dataset()

        print()
        print("=" * 70)
        print("S.H.A ANC FEDERATED PARAMETER SERVER")
        print("=" * 70)
        print(f"Host participant : YES")
        print(f"Remote workers   : {self.num_workers}")
        print(f"Total participants: {self.num_participants}")
        print(f"Port             : {self.port}")
        print(f"Maximum epochs   : {self.epochs}")
        print(f"Learning rate    : {self.learning_rate}")
        print(f"Participant patience: {self.participant_patience}")
        print(f"Global patience  : {self.global_patience}")
        print(f"Ready timeout    : {self.ready_timeout}s")
        print(f"Round timeout    : {self.round_timeout}s")  # NEW
        print(f"Socket timeout   : {WORKER_SOCKET_TIMEOUT}s")  # NEW
        if self.max_samples is not None:
            print(f"Max samples      : {self.max_samples} (TESTING MODE)")
        print("=" * 70)

        # SERVER SOCKET
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("0.0.0.0", self.port))
        self.server_socket.listen(self.num_workers)

        print()
        print(f"Waiting for {self.num_workers} workers...")

        # ACCEPT WORKERS
        for worker_id in range(1, self.num_workers + 1):
            connection, address = self.server_socket.accept()

            enable_keepalive(
                connection,
                idle_seconds=20,
                interval_seconds=10,
                max_probes=5,
            )

            connection.settimeout(WORKER_SOCKET_TIMEOUT)  # FIXED: use configurable timeout

            self.workers[worker_id] = connection
            self.worker_connections[worker_id] = connection

            print(f"Worker {worker_id} connected from {address}")
            print(f"[Server] TCP keepalive enabled for Worker {worker_id}")
            print(f"[Server] Socket timeout: {WORKER_SOCKET_TIMEOUT}s")

            thread = threading.Thread(
                target=self.handle_worker,
                args=(worker_id, connection),
                daemon=True,
            )
            thread.start()

        # READY HANDSHAKE
        print()
        print("=" * 70)
        print("WAITING FOR ALL WORKERS TO BE READY")
        print("=" * 70)
        print("Workers are preparing their local VoiceBank-DEMAND datasets.")

        try:
            self.wait_for_workers_ready()
        except TimeoutError as error:
            print()
            print(f"[Server] {error}")
            return

        print()
        print("=" * 70)
        print("ALL REMOTE WORKERS READY")
        print("HOST IS ALSO A TRAINING PARTICIPANT")
        print(f"TOTAL PARTICIPANTS = {self.num_participants}")
        print("=" * 70)

        # FEDERATED TRAINING
        for epoch in range(1, self.epochs + 1):
            print()
            print("=" * 70)
            print(f"FEDERATED ROUND {epoch}/{self.epochs}")
            print("=" * 70)

            # RESET ROUND
            with self.lock:
                self.updates.clear()

            # BROADCAST
            print()
            print("Broadcasting global model...")
            self.broadcast_model(epoch)

            # START HOST TRAINING
            host_result = {}

            def host_training():
                if self.host_converged:
                    print()
                    print("[HOST] Already converged.")
                    print("[HOST] Skipping local training this round.")
                    validation_loss = self.validate_host()
                    host_result.update({
                        "loss": self.host_best_validation_loss,
                        "validation_loss": validation_loss,
                        "training_time": 0.0,
                        "batches": 0,
                        "status": "CONVERGED",
                        "weights": self.model_state_dict(self.model),
                    })
                    return

                print()
                print("[HOST] LOCAL TRAINING")
                loss, elapsed, batches = self.train_host_epoch()
                validation_loss = self.validate_host()
                converged = self.check_host_convergence(validation_loss, epoch)

                status = "CONVERGED" if converged else "TRAINING"

                if converged:
                    print()
                    print("[HOST] EARLY STOPPING")
                    print(f"[HOST] Validation loss stopped improving at round {epoch}.")

                print(
                    f"[HOST] train={loss:.6f} | "
                    f"validation={validation_loss:.6f} | "
                    f"time={elapsed:.2f}s"
                )

                host_result.update({
                    "loss": loss,
                    "validation_loss": validation_loss,
                    "training_time": elapsed,
                    "batches": batches,
                    "status": status,
                    "weights": self.model_state_dict(self.model),
                })

            host_thread = threading.Thread(target=host_training, daemon=True)
            host_thread.start()

            # WAIT FOR REMOTE WORKERS
            print()
            print("HOST, WORKER 1 AND WORKER 2 ARE TRAINING...")

            try:
                worker_updates = self.wait_for_updates(epoch)
            except TimeoutError as error:
                print()
                print(f"[Server] {error}")
                break

            # WAIT FOR HOST
            host_thread.join()

            # VERIFY HOST
            if not host_result:
                print("[Server] Host training did not produce an update.")
                break

            # DISPLAY RESULTS
            print()
            print("-" * 70)
            print("PARTICIPANT PERFORMANCE")
            print("-" * 70)

            print(
                f"  HOST: "
                f"train={host_result['loss']:.6f} | "
                f"validation={host_result['validation_loss']:.6f} | "
                f"status={host_result['status']}"
            )

            for update in worker_updates:
                self.print_worker_update(update)

            # BUILD 3-PARTICIPANT UPDATE
            participant_updates = [host_result]
            participant_updates.extend(worker_updates)

            # CHECK PARTICIPANT COUNT
            valid_updates = [
                update for update in participant_updates
                if "weights" in update
            ]

            if len(valid_updates) != self.num_participants:
                print()
                print("[Server] WARNING:")
                print(f"Expected {self.num_participants} participant models but received {len(valid_updates)}.")
                break

            # METRICS
            losses = [
                float(update.get("loss", 0.0))
                for update in valid_updates
            ]

            validation_losses = [
                float(update.get("validation_loss", update.get("loss", 0.0)))
                for update in valid_updates
            ]

            average_loss = sum(losses) / len(losses)
            average_validation_loss = sum(validation_losses) / len(validation_losses)

            print()
            print("-" * 70)
            print(f"Average participant training loss     = {average_loss:.6f}")
            print(f"Average participant validation loss   = {average_validation_loss:.6f}")
            print("-" * 70)

            # FEDERATED AVERAGING
            print()
            print("FEDERATED AVERAGING")
            print(f"Aggregating {len(valid_updates)} participant models:")
            print("  1. HOST")
            for worker_id in sorted(self.worker_ready):
                print(f"  {worker_id + 1}. WORKER {worker_id}")

            participant_states = [
                update["weights"] for update in valid_updates
            ]

            self.average_models(participant_states)
            print("Global model updated.")

            # GLOBAL EARLY STOPPING
            global_converged = self.check_global_convergence(average_validation_loss)

            print()
            print("GLOBAL EARLY STOPPING")
            print(f"Best global validation loss: {self.global_best_validation_loss:.6f}")
            print(
                f"Rounds without sufficient improvement: "
                f"{self.global_epochs_without_improvement}/{self.global_patience}"
            )

            # CHECKPOINT
            self.save_checkpoint(
                epoch=epoch,
                average_training_loss=average_loss,
                average_validation_loss=average_validation_loss,
            )

            print()
            print(f"Global checkpoint saved:\n{GLOBAL_MODEL_PATH}")

            # GLOBAL STOP
            if global_converged:
                print()
                print("=" * 70)
                print("GLOBAL EARLY STOPPING TRIGGERED")
                print(f"No sufficient global improvement for {self.global_patience} rounds.")
                print("=" * 70)
                break

            # PARTICIPANT STATUS
            with self.lock:
                stopped_count = len(self.stopped_workers)

            print()
            print(f"Remote workers converged: {stopped_count}/{self.num_workers}")
            print(f"Host converged: {self.host_converged}")

        # COMPLETE
        self.training_complete = True

        print()
        print("=" * 70)
        print("S.H.A ANC DISTRIBUTED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final model:")
        print(f"{GLOBAL_MODEL_PATH}")
        print(f"Best global validation loss: {self.global_best_validation_loss:.6f}")
        print(f"Remote workers converged: {sorted(self.stopped_workers)}")
        print(f"Host converged: {self.host_converged}")
        print("=" * 70)

        # CLOSE WORKERS
        for sock in self.workers.values():
            try:
                self.send_message(sock, {"type": "TRAINING_COMPLETE"})
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

        try:
            self.server_socket.close()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="S.H.A ANC federated parameter server with efficient training."
    )

    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT,
        help="Loss delta required to count as host improvement.",
    )
    parser.add_argument(
        "--participant-patience",
        type=int,
        default=PATIENCE,
        help="Rounds without sufficient improvement before HOST stops.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=GLOBAL_PATIENCE,
        help="Rounds without sufficient improvement before GLOBAL training stops.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=READY_TIMEOUT,
        help=f"Seconds to wait for workers to become READY (default: {READY_TIMEOUT})",
    )
    parser.add_argument(
        "--round-timeout",
        type=int,
        default=ROUND_TIMEOUT,
        help=f"Seconds to wait for round updates (default: {ROUND_TIMEOUT})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N samples per participant (for fast testing).",
    )

    args = parser.parse_args()

    server = SHAANCParameterServer(
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_improvement=args.min_improvement,
        participant_patience=args.participant_patience,
        global_patience=args.patience,
        ready_timeout=args.ready_timeout,
        round_timeout=args.round_timeout,
        max_samples=args.max_samples,
    )

    server.start()


if __name__ == "__main__":
    main()