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
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MIN_IMPROVEMENT,
    PATIENCE,
    READY_TIMEOUT,
    ROUND_TIMEOUT,
)

from dataset import get_local_loaders
from model import create_model


# ============================================================
# TCP KEEPALIVE
# ============================================================

def enable_keepalive(
    sock,
    idle_seconds=20,
    interval_seconds=10,
    max_probes=5,
):

    try:
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1,
        )
    except OSError:
        return

    # Linux / Unix
    try:
        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_KEEPIDLE,
            idle_seconds,
        )

        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_KEEPINTVL,
            interval_seconds,
        )

        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_KEEPCNT,
            max_probes,
        )

    except (
        AttributeError,
        OSError,
    ):
        pass

    # Windows
    try:

        if hasattr(
            socket,
            "SIO_KEEPALIVE_VALS",
        ):

            sock.ioctl(
                socket.SIO_KEEPALIVE_VALS,
                (
                    1,
                    idle_seconds * 1000,
                    interval_seconds * 1000,
                ),
            )

    except (
        AttributeError,
        OSError,
    ):
        pass


# ============================================================
# PARAMETER SERVER
# ============================================================

class SHAANCParameterServer:

    def __init__(
        self,
        port=DEFAULT_PORT,
        num_workers=DEFAULT_WORKERS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        min_improvement=MIN_IMPROVEMENT,
        participant_patience=PATIENCE,
        global_patience=GLOBAL_PATIENCE,
        ready_timeout=READY_TIMEOUT,
        round_timeout=ROUND_TIMEOUT,
        max_samples=None,
        grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,
    ):

        # ----------------------------------------------------
        # CONFIG
        # ----------------------------------------------------

        self.port = int(port)

        self.num_workers = int(
            num_workers
        )

        # HOST + REMOTE WORKERS
        self.num_participants = (
            self.num_workers + 1
        )

        self.epochs = int(epochs)

        self.batch_size = int(
            batch_size
        )

        self.learning_rate = float(
            learning_rate
        )

        self.min_improvement = float(
            min_improvement
        )

        self.participant_patience = max(
            int(participant_patience),
            1,
        )

        self.global_patience = max(
            int(global_patience),
            1,
        )

        self.ready_timeout = max(
            int(ready_timeout),
            1,
        )

        self.round_timeout = max(
            int(round_timeout),
            1,
        )

        self.max_samples = max_samples

        self.grad_accum_steps = max(
            int(grad_accum_steps),
            1,
        )

        # ----------------------------------------------------
        # GLOBAL MODEL
        # ----------------------------------------------------

        self.model = create_model()

        self.criterion = nn.L1Loss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # ----------------------------------------------------
        # HOST DATA
        # ----------------------------------------------------

        self.host_train_loader = None

        self.host_validation_loader = None

        # ----------------------------------------------------
        # NETWORK
        # ----------------------------------------------------

        self.server_socket = None

        self.workers = {}

        self.worker_connections = {}

        # ----------------------------------------------------
        # WORKER STATE
        # ----------------------------------------------------

        self.worker_ready = set()

        self.active_workers = set()

        self.stopped_workers = set()

        # ----------------------------------------------------
        # ROUND UPDATES
        # ----------------------------------------------------

        self.updates = {}

        # ----------------------------------------------------
        # THREAD SYNCHRONIZATION
        # ----------------------------------------------------

        self.lock = threading.Lock()

        self.ready_condition = (
            threading.Condition(
                self.lock
            )
        )

        self.update_condition = (
            threading.Condition(
                self.lock
            )
        )

        # ----------------------------------------------------
        # GLOBAL EARLY STOPPING
        # ----------------------------------------------------

        self.global_best_validation_loss = (
            float("inf")
        )

        self.global_epochs_without_improvement = 0

        # ----------------------------------------------------
        # HOST EARLY STOPPING
        # ----------------------------------------------------

        self.host_best_validation_loss = (
            float("inf")
        )

        self.host_epochs_without_improvement = 0

        self.host_converged = False

        self.host_convergence_epoch = None

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.training_complete = False

    # ========================================================
    # NETWORK SERIALIZATION
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

        sock.sendall(header)

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
                    "Connection closed."
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

        if size <= 0:
            raise ConnectionError(
                "Invalid payload size."
            )

        return cls.recv_exact(
            sock,
            size,
        )

    @staticmethod
    def send_message(
        sock,
        message,
    ):

        payload = pickle.dumps(
            message,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        SHAANCParameterServer.send_bytes(
            sock,
            payload,
        )

    # ========================================================
    # MODEL STATE
    # ========================================================

    @staticmethod
    def model_state_dict(
        model,
    ):

        return {
            key: value.detach().cpu()
            for key, value
            in model.state_dict().items()
        }

    # ========================================================
    # HOST DATASET
    # ========================================================

    def prepare_host_dataset(self):

        print()
        print("=" * 70)
        print("PREPARING HOST LOCAL DATASET")
        print("=" * 70)

        if self.max_samples is not None:

            print(
                f"[HOST] Limiting dataset to "
                f"{self.max_samples} clean files."
            )

        (
            self.host_train_loader,
            self.host_validation_loader,
        ) = get_local_loaders(
            participant_id="HOST",
            batch_size=self.batch_size,
            max_samples=self.max_samples,
        )

        print()
        print("[HOST] Local dataset READY.")

        print(
            f"[HOST] Train batches: "
            f"{len(self.host_train_loader)}"
        )

        print(
            f"[HOST] Validation batches: "
            f"{len(self.host_validation_loader)}"
        )

    # ========================================================
    # HOST LOCAL TRAINING
    # ========================================================

    def train_host_epoch(self):

        if self.host_train_loader is None:
            raise RuntimeError(
                "Host dataset has not been prepared."
            )

        self.model.train()

        total_loss = 0.0

        batches = 0

        start_time = time.time()

        accumulation_steps = (
            self.grad_accum_steps
        )

        use_amp = torch.cuda.is_available()

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
            self.host_train_loader
        ):

            if use_amp:

                noisy = noisy.cuda(
                    non_blocking=True
                )

                clean = clean.cuda(
                    non_blocking=True
                )

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

            is_boundary = (
                (batch_idx + 1)
                % accumulation_steps
                == 0
            )

            is_last = (
                batch_idx + 1
                == len(self.host_train_loader)
            )

            if is_boundary or is_last:

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
    # HOST VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate_host(self):

        if self.host_validation_loader is None:
            return 0.0

        self.model.eval()

        total_loss = 0.0

        batches = 0

        for noisy, clean in (
            self.host_validation_loader
        ):

            if torch.cuda.is_available():

                noisy = noisy.cuda(
                    non_blocking=True
                )

                clean = clean.cuda(
                    non_blocking=True
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
    # HOST EARLY STOPPING
    # ========================================================

    def check_host_convergence(
        self,
        validation_loss,
        epoch,
    ):

        improvement = (
            self.host_best_validation_loss
            - validation_loss
        )

        if validation_loss < (
            self.host_best_validation_loss
        ):

            self.host_best_validation_loss = (
                validation_loss
            )

        if improvement >= (
            self.min_improvement
        ):

            self.host_epochs_without_improvement = 0

        else:

            self.host_epochs_without_improvement += 1

        if (
            self.host_epochs_without_improvement
            >= self.participant_patience
        ):

            self.host_converged = True

            self.host_convergence_epoch = epoch

            return True

        return False

    # ========================================================
    # BROADCAST GLOBAL MODEL
    # ========================================================

    def broadcast_model(self, epoch):

        state = self.model_state_dict(
            self.model
        )

        message = {
            "type": "GLOBAL_MODEL",
            "epoch": epoch,
            "round": epoch,
            "weights": state,
        }

        # IMPORTANT:
        # Send to ALL READY workers, including workers that
        # have locally converged. They must remain synchronized
        # until global training finishes.

        with self.lock:

            workers_snapshot = {
                worker_id: sock
                for worker_id, sock
                in self.workers.items()
                if worker_id
                in self.worker_ready
            }

        print(
            f"Sending global model to "
            f"{len(workers_snapshot)} workers..."
        )

        for worker_id, sock in (
            workers_snapshot.items()
        ):

            try:

                self.send_message(
                    sock,
                    message,
                )

            except Exception as error:

                print(
                    f"[Server] Could not send "
                    f"global model to Worker "
                    f"{worker_id}: {error}"
                )

                with self.lock:
                    self.active_workers.discard(
                        worker_id
                    )

    # ========================================================
    # WORKER CONNECTION HANDLER
    # ========================================================

    def handle_worker(
        self,
        worker_id,
        sock,
    ):

        try:

            while not self.training_complete:

                payload = self.recv_bytes(
                    sock
                )

                message = pickle.loads(
                    payload
                )

                message_type = message.get(
                    "type"
                )

                # ============================================
                # READY
                # ============================================

                if message_type == "READY":

                    with self.ready_condition:

                        self.worker_ready.add(
                            worker_id
                        )

                        self.active_workers.add(
                            worker_id
                        )

                        print()
                        print(
                            f"[Server] Worker "
                            f"{worker_id} is READY."
                        )

                        print(
                            f"[Server] Ready workers: "
                            f"{len(self.worker_ready)}/"
                            f"{self.num_workers}"
                        )

                        self.ready_condition.notify_all()

                # ============================================
                # ROUND UPDATE
                # ============================================

                elif message_type == "UPDATE":

                    update_epoch = message.get(
                        "epoch"
                    )

                    with self.update_condition:

                        self.updates[
                            worker_id
                        ] = message

                        status = message.get(
                            "status",
                            "TRAINING",
                        )

                        if status == "CONVERGED":

                            self.stopped_workers.add(
                                worker_id
                            )

                        self.active_workers.add(
                            worker_id
                        )

                        print()
                        print(
                            f"[Server] Received "
                            f"Worker {worker_id} "
                            f"update for round "
                            f"{update_epoch}."
                        )

                        self.update_condition.notify_all()

                # ============================================
                # FINISHED
                # ============================================

                elif message_type == "FINISHED":

                    print(
                        f"[Server] Worker "
                        f"{worker_id} reports "
                        "training finished."
                    )

                    with self.lock:

                        self.active_workers.discard(
                            worker_id
                        )

                else:

                    print(
                        f"[Server] Unknown message "
                        f"from Worker {worker_id}: "
                        f"{message_type}"
                    )

        except Exception as error:

            if not self.training_complete:

                print()
                print(
                    f"[Server] Worker "
                    f"{worker_id} connection ended: "
                    f"{error}"
                )

            with self.lock:

                self.active_workers.discard(
                    worker_id
                )

                self.worker_connections.pop(
                    worker_id,
                    None,
                )

    # ========================================================
    # WAIT FOR READY
    # ========================================================

    def wait_for_workers_ready(
        self,
        timeout=None,
    ):

        if timeout is None:
            timeout = self.ready_timeout

        deadline = (
            time.time()
            + timeout
        )

        with self.ready_condition:

            while len(
                self.worker_ready
            ) < self.num_workers:

                remaining = (
                    deadline
                    - time.time()
                )

                if remaining <= 0:

                    missing = (
                        set(
                            range(
                                1,
                                self.num_workers + 1,
                            )
                        )
                        - self.worker_ready
                    )

                    raise TimeoutError(
                        f"Timed out waiting for "
                        f"workers to become READY "
                        f"after {timeout}s. "
                        f"Missing workers: "
                        f"{sorted(missing)}"
                    )

                self.ready_condition.wait(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

                print(
                    f"\rReady workers: "
                    f"{len(self.worker_ready)}/"
                    f"{self.num_workers}",
                    end="",
                    flush=True,
                )

        print()

    # ========================================================
    # WAIT FOR CURRENT ROUND UPDATES
    # ========================================================

    def wait_for_updates(
        self,
        epoch,
        timeout=None,
    ):

        if timeout is None:
            timeout = self.round_timeout

        deadline = (
            time.time()
            + timeout
        )

        required_workers = set(
            self.worker_ready
        )

        while True:

            with self.update_condition:

                received_workers = set()

                for worker_id, update in (
                    self.updates.items()
                ):

                    update_epoch = update.get(
                        "epoch"
                    )

                    if update_epoch == epoch:
                        received_workers.add(
                            worker_id
                        )

                if (
                    received_workers
                    >= required_workers
                ):

                    return [
                        self.updates[
                            worker_id
                        ]
                        for worker_id
                        in sorted(
                            required_workers
                        )
                    ]

                remaining = (
                    deadline
                    - time.time()
                )

                if remaining <= 0:

                    missing = (
                        required_workers
                        - received_workers
                    )

                    raise TimeoutError(
                        f"Timed out waiting for "
                        f"round {epoch} updates "
                        f"after {timeout}s. "
                        f"Missing workers: "
                        f"{sorted(missing)}"
                    )

                self.update_condition.wait(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

    # ========================================================
    # FEDERATED AVERAGING
    # ========================================================

    def average_models(
        self,
        participant_states,
    ):

        if len(
            participant_states
        ) != self.num_participants:

            raise RuntimeError(
                f"Expected "
                f"{self.num_participants} "
                f"participant models, "
                f"received "
                f"{len(participant_states)}."
            )

        averaged = {}

        keys = participant_states[
            0
        ].keys()

        for key in keys:

            tensors = []

            for state in (
                participant_states
            ):

                tensors.append(
                    state[key].float()
                )

            stacked = torch.stack(
                tensors,
                dim=0,
            )

            averaged[key] = (
                stacked.mean(dim=0)
            )

        self.model.load_state_dict(
            averaged,
            strict=True,
        )

    # ========================================================
    # GLOBAL EARLY STOPPING
    # ========================================================

    def check_global_convergence(
        self,
        validation_loss,
    ):

        improvement = (
            self.global_best_validation_loss
            - validation_loss
        )

        if validation_loss < (
            self.global_best_validation_loss
        ):

            self.global_best_validation_loss = (
                validation_loss
            )

        if improvement >= (
            GLOBAL_MIN_IMPROVEMENT
        ):

            self.global_epochs_without_improvement = 0

        else:

            self.global_epochs_without_improvement += 1

        return (
            self.global_epochs_without_improvement
            >= self.global_patience
        )

    # ========================================================
    # PRINT WORKER RESULT
    # ========================================================

    def print_worker_update(
        self,
        update,
    ):

        worker_id = update.get(
            "worker_id",
            "?",
        )

        loss = float(
            update.get(
                "loss",
                0.0,
            )
        )

        validation_loss = float(
            update.get(
                "validation_loss",
                loss,
            )
        )

        status = update.get(
            "status",
            "TRAINING",
        )

        elapsed = float(
            update.get(
                "training_time",
                0.0,
            )
        )

        print(
            f"  WORKER {worker_id}: "
            f"train={loss:.6f} | "
            f"validation="
            f"{validation_loss:.6f} | "
            f"time={elapsed:.2f}s | "
            f"status={status}"
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        epoch,
        average_training_loss,
        average_validation_loss,
    ):

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    self.model.state_dict(),

                "average_training_loss":
                    average_training_loss,

                "average_validation_loss":
                    average_validation_loss,

                "num_workers":
                    self.num_workers,

                "num_participants":
                    self.num_participants,

                "stopped_workers":
                    sorted(
                        self.stopped_workers
                    ),

                "host_converged":
                    self.host_converged,

                "host_convergence_epoch":
                    self.host_convergence_epoch,

                "global_best_validation_loss":
                    self.global_best_validation_loss,
            },
            GLOBAL_MODEL_PATH,
        )

    # ========================================================
    # CLOSE CONNECTIONS
    # ========================================================

    def shutdown_workers(self):

        print()
        print(
            "Sending TRAINING_COMPLETE "
            "to workers..."
        )

        with self.lock:

            workers_snapshot = dict(
                self.workers
            )

        for worker_id, sock in (
            workers_snapshot.items()
        ):

            try:

                self.send_message(
                    sock,
                    {
                        "type":
                            "TRAINING_COMPLETE"
                    },
                )

            except Exception:
                pass

        # Give workers a moment to receive
        # the final message before closing.
        time.sleep(1)

        for sock in (
            workers_snapshot.values()
        ):

            try:
                sock.close()
            except Exception:
                pass

    # ========================================================
    # START SERVER
    # ========================================================

    def start(self):

        # ----------------------------------------------------
        # HOST DATA
        # ----------------------------------------------------

        self.prepare_host_dataset()

        # ----------------------------------------------------
        # CONFIG DISPLAY
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "S.H.A ANC FEDERATED PARAMETER SERVER"
        )
        print("=" * 70)

        print(
            "Host participant : YES"
        )

        print(
            f"Remote workers   : "
            f"{self.num_workers}"
        )

        print(
            f"Total participants: "
            f"{self.num_participants}"
        )

        print(
            f"Port             : "
            f"{self.port}"
        )

        print(
            f"Maximum epochs   : "
            f"{self.epochs}"
        )

        print(
            f"Batch size       : "
            f"{self.batch_size}"
        )

        print(
            f"Learning rate    : "
            f"{self.learning_rate}"
        )

        print(
            f"Participant patience: "
            f"{self.participant_patience}"
        )

        print(
            f"Global patience  : "
            f"{self.global_patience}"
        )

        print(
            f"Ready timeout    : "
            f"{self.ready_timeout}s"
        )

        print(
            f"Round timeout    : "
            f"{self.round_timeout}s"
        )

        print(
            "Worker socket    : "
            "persistent"
        )

        if self.max_samples is not None:

            print(
                f"Max samples      : "
                f"{self.max_samples}"
            )

        print("=" * 70)

        # ----------------------------------------------------
        # SERVER SOCKET
        # ----------------------------------------------------

        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.server_socket.bind(
            (
                "0.0.0.0",
                self.port,
            )
        )

        self.server_socket.listen(
            self.num_workers
        )

        print()
        print(
            f"Waiting for "
            f"{self.num_workers} workers..."
        )

        # ----------------------------------------------------
        # ACCEPT WORKERS
        # ----------------------------------------------------

        for worker_id in range(
            1,
            self.num_workers + 1,
        ):

            connection, address = (
                self.server_socket.accept()
            )

            enable_keepalive(
                connection
            )

            # =================================================
            # CRITICAL FIX
            #
            # DO NOT use:
            #
            # connection.settimeout(7200)
            #
            # A worker may spend a long time preparing/loading
            # local audio before sending READY.
            #
            # Persistent socket stays open indefinitely.
            # =================================================

            connection.settimeout(
                None
            )

            self.workers[
                worker_id
            ] = connection

            self.worker_connections[
                worker_id
            ] = connection

            print(
                f"Worker {worker_id} "
                f"connected from {address}"
            )

            print(
                f"[Server] TCP keepalive "
                f"enabled for Worker "
                f"{worker_id}"
            )

            print(
                f"[Server] Worker {worker_id} "
                f"socket is persistent."
            )

            thread = threading.Thread(
                target=self.handle_worker,
                args=(
                    worker_id,
                    connection,
                ),
                daemon=True,
            )

            thread.start()

        # ----------------------------------------------------
        # READY HANDSHAKE
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "WAITING FOR ALL WORKERS TO BE READY"
        )
        print("=" * 70)

        print(
            "Workers are preparing their "
            "LOCAL LibriSpeech/MUSAN datasets."
        )

        try:

            self.wait_for_workers_ready()

        except TimeoutError as error:

            print()
            print(
                f"[Server] {error}"
            )

            self.training_complete = True

            self.shutdown_workers()

            return

        # ----------------------------------------------------
        # ALL READY
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "ALL REMOTE WORKERS READY"
        )

        print(
            "HOST IS ALSO A TRAINING PARTICIPANT"
        )

        print(
            f"TOTAL PARTICIPANTS = "
            f"{self.num_participants}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # FEDERATED TRAINING
        # ----------------------------------------------------

        try:

            for epoch in range(
                1,
                self.epochs + 1,
            ):

                print()
                print("=" * 70)
                print(
                    f"FEDERATED ROUND "
                    f"{epoch}/{self.epochs}"
                )
                print("=" * 70)

                # ============================================
                # CLEAR ONLY PREVIOUS ROUND UPDATES
                # ============================================

                with self.lock:
                    self.updates.clear()

                # ============================================
                # BROADCAST GLOBAL MODEL
                # ============================================

                print()
                print(
                    "Broadcasting global model..."
                )

                self.broadcast_model(
                    epoch
                )

                # ============================================
                # HOST TRAINING
                # ============================================

                host_result = {}

                def host_training():

                    if self.host_converged:

                        print()
                        print(
                            "[HOST] Already converged."
                        )

                        print(
                            "[HOST] Skipping "
                            "local training."
                        )

                        validation_loss = (
                            self.validate_host()
                        )

                        host_result.update(
                            {
                                "worker_id":
                                    "HOST",

                                "epoch":
                                    epoch,

                                "loss":
                                    self.host_best_validation_loss,

                                "validation_loss":
                                    validation_loss,

                                "training_time":
                                    0.0,

                                "batches":
                                    0,

                                "status":
                                    "CONVERGED",

                                "weights":
                                    self.model_state_dict(
                                        self.model
                                    ),
                            }
                        )

                        return

                    print()
                    print(
                        "[HOST] LOCAL TRAINING"
                    )

                    (
                        loss,
                        elapsed,
                        batches,
                    ) = self.train_host_epoch()

                    validation_loss = (
                        self.validate_host()
                    )

                    converged = (
                        self.check_host_convergence(
                            validation_loss,
                            epoch,
                        )
                    )

                    status = (
                        "CONVERGED"
                        if converged
                        else "TRAINING"
                    )

                    if converged:

                        print()
                        print(
                            "[HOST] EARLY STOPPING"
                        )

                        print(
                            f"[HOST] Converged "
                            f"at round {epoch}."
                        )

                    print(
                        f"[HOST] "
                        f"train={loss:.6f} | "
                        f"validation="
                        f"{validation_loss:.6f} | "
                        f"time={elapsed:.2f}s | "
                        f"status={status}"
                    )

                    host_result.update(
                        {
                            "worker_id":
                                "HOST",

                            "epoch":
                                epoch,

                            "loss":
                                loss,

                            "validation_loss":
                                validation_loss,

                            "training_time":
                                elapsed,

                            "batches":
                                batches,

                            "status":
                                status,

                            "weights":
                                self.model_state_dict(
                                    self.model
                                ),
                        }
                    )

                # Start host at same time workers
                host_thread = threading.Thread(
                    target=host_training,
                    daemon=True,
                )

                host_thread.start()

                # ============================================
                # WAIT FOR WORKERS
                # ============================================

                print()
                print(
                    "HOST + WORKER 1 + WORKER 2 "
                    "ARE TRAINING IN PARALLEL..."
                )

                try:

                    worker_updates = (
                        self.wait_for_updates(
                            epoch
                        )
                    )

                except TimeoutError as error:

                    print()
                    print(
                        f"[Server] {error}"
                    )

                    break

                # ============================================
                # WAIT FOR HOST
                # ============================================

                host_thread.join()

                if not host_result:

                    print(
                        "[Server] Host training "
                        "did not produce "
                        "an update."
                    )

                    break

                # ============================================
                # DISPLAY RESULTS
                # ============================================

                print()
                print("-" * 70)
                print(
                    "PARTICIPANT PERFORMANCE"
                )
                print("-" * 70)

                print(
                    f"  HOST: "
                    f"train="
                    f"{host_result['loss']:.6f} | "
                    f"validation="
                    f"{host_result['validation_loss']:.6f} | "
                    f"status="
                    f"{host_result['status']}"
                )

                for update in (
                    worker_updates
                ):

                    self.print_worker_update(
                        update
                    )

                # ============================================
                # THREE PARTICIPANTS
                # ============================================

                participant_updates = [
                    host_result
                ]

                participant_updates.extend(
                    worker_updates
                )

                valid_updates = [
                    update
                    for update
                    in participant_updates
                    if "weights" in update
                ]

                if (
                    len(valid_updates)
                    != self.num_participants
                ):

                    print()
                    print(
                        "[Server] ERROR:"
                    )

                    print(
                        f"Expected "
                        f"{self.num_participants} "
                        f"participant models."
                    )

                    print(
                        f"Received "
                        f"{len(valid_updates)}."
                    )

                    break

                # ============================================
                # METRICS
                # ============================================

                losses = [
                    float(
                        update.get(
                            "loss",
                            0.0,
                        )
                    )
                    for update
                    in valid_updates
                ]

                validation_losses = [
                    float(
                        update.get(
                            "validation_loss",
                            update.get(
                                "loss",
                                0.0,
                            ),
                        )
                    )
                    for update
                    in valid_updates
                ]

                average_loss = (
                    sum(losses)
                    / len(losses)
                )

                average_validation_loss = (
                    sum(validation_losses)
                    / len(validation_losses)
                )

                print()
                print("-" * 70)

                print(
                    f"Average participant "
                    f"training loss   = "
                    f"{average_loss:.6f}"
                )

                print(
                    f"Average participant "
                    f"validation loss = "
                    f"{average_validation_loss:.6f}"
                )

                print("-" * 70)

                # ============================================
                # FEDERATED AVERAGING
                # ============================================

                print()
                print(
                    "FEDERATED AVERAGING"
                )

                print(
                    f"Aggregating "
                    f"{len(valid_updates)} "
                    f"participants:"
                )

                print(
                    "  1. HOST"
                )

                for worker_id in sorted(
                    self.worker_ready
                ):

                    print(
                        f"  {worker_id + 1}. "
                        f"WORKER {worker_id}"
                    )

                participant_states = [
                    update["weights"]
                    for update
                    in valid_updates
                ]

                self.average_models(
                    participant_states
                )

                print(
                    "Global model updated."
                )

                # ============================================
                # GLOBAL EARLY STOPPING
                # ============================================

                global_converged = (
                    self.check_global_convergence(
                        average_validation_loss
                    )
                )

                print()
                print(
                    "GLOBAL EARLY STOPPING"
                )

                print(
                    f"Best global "
                    f"validation loss: "
                    f"{self.global_best_validation_loss:.6f}"
                )

                print(
                    f"Rounds without "
                    f"sufficient improvement: "
                    f"{self.global_epochs_without_improvement}/"
                    f"{self.global_patience}"
                )

                # ============================================
                # CHECKPOINT
                # ============================================

                self.save_checkpoint(
                    epoch=epoch,
                    average_training_loss=average_loss,
                    average_validation_loss=(
                        average_validation_loss
                    ),
                )

                print()
                print(
                    "Global checkpoint saved:"
                )

                print(
                    GLOBAL_MODEL_PATH
                )

                # ============================================
                # GLOBAL STOP
                # ============================================

                if global_converged:

                    print()
                    print("=" * 70)
                    print(
                        "GLOBAL EARLY STOPPING "
                        "TRIGGERED"
                    )

                    print(
                        f"No sufficient global "
                        f"improvement for "
                        f"{self.global_patience} "
                        f"rounds."
                    )

                    print("=" * 70)

                    break

                # ============================================
                # PARTICIPANT STATUS
                # ============================================

                with self.lock:

                    stopped_count = len(
                        self.stopped_workers
                    )

                print()
                print(
                    f"Remote workers converged: "
                    f"{stopped_count}/"
                    f"{self.num_workers}"
                )

                print(
                    f"Host converged: "
                    f"{self.host_converged}"
                )

        finally:

            # ------------------------------------------------
            # TRAINING COMPLETE
            # ------------------------------------------------

            self.training_complete = True

            print()
            print("=" * 70)
            print(
                "S.H.A ANC DISTRIBUTED "
                "TRAINING COMPLETE"
            )
            print("=" * 70)

            print(
                "Final model:"
            )

            print(
                GLOBAL_MODEL_PATH
            )

            print(
                f"Best global validation loss: "
                f"{self.global_best_validation_loss:.6f}"
            )

            print(
                f"Remote workers converged: "
                f"{sorted(self.stopped_workers)}"
            )

            print(
                f"Host converged: "
                f"{self.host_converged}"
            )

            print("=" * 70)

            # ------------------------------------------------
            # CLOSE WORKERS
            # ------------------------------------------------

            self.shutdown_workers()

            # ------------------------------------------------
            # CLOSE SERVER
            # ------------------------------------------------

            if self.server_socket is not None:

                try:
                    self.server_socket.close()
                except Exception:
                    pass


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "S.H.A ANC federated parameter "
            "server."
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
        "--participant-patience",
        type=int,
        default=PATIENCE,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=GLOBAL_PATIENCE,
    )

    parser.add_argument(
        "--ready-timeout",
        type=int,
        default=READY_TIMEOUT,
    )

    parser.add_argument(
        "--round-timeout",
        type=int,
        default=ROUND_TIMEOUT,
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
            "Limit local clean files "
            "for fast testing."
        ),
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
        grad_accum_steps=args.grad_accum_steps,
    )

    server.start()


if __name__ == "__main__":
    main()