import argparse
import pickle
import random
import socket
import threading
import time

import numpy as np
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
    PATIENCE,
)

from dataset import (
    get_local_loaders,
)

from model import create_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# PARAMETER SERVER + HOST PARTICIPANT
# ============================================================

class SHAANCParameterServer:

    def __init__(
        self,
        port,
        num_workers,
        epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        participant_patience=PATIENCE,
        participant_min_improvement=MIN_IMPROVEMENT,
        global_patience=GLOBAL_PATIENCE,
        global_min_improvement=GLOBAL_MIN_IMPROVEMENT,
    ):

        self.port = port

        self.num_workers = num_workers

        self.num_participants = (
            num_workers + 1
        )

        self.epochs = epochs

        self.batch_size = batch_size

        self.learning_rate = (
            learning_rate
        )

        # ----------------------------------------------------
        # Global model
        # ----------------------------------------------------

        self.model = create_model()

        # ----------------------------------------------------
        # Host local training
        # ----------------------------------------------------

        self.host_criterion = nn.L1Loss()

        self.host_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        self.host_train_loader = None

        self.host_validation_loader = None

        self.host_best_validation_loss = (
            float("inf")
        )

        self.host_previous_validation_loss = (
            None
        )

        self.host_epochs_without_improvement = 0

        self.host_converged = False

        self.host_convergence_epoch = None

        self.host_last_loss = float("inf")

        self.host_last_validation_loss = (
            float("inf")
        )

        # ----------------------------------------------------
        # Participant early stopping
        # ----------------------------------------------------

        self.participant_patience = max(
            int(participant_patience),
            1,
        )

        self.participant_min_improvement = (
            float(
                participant_min_improvement
            )
        )

        # ----------------------------------------------------
        # Global early stopping
        # ----------------------------------------------------

        self.global_patience = max(
            int(global_patience),
            1,
        )

        self.global_min_improvement = (
            float(
                global_min_improvement
            )
        )

        self.global_epochs_without_improvement = (
            0
        )

        self.best_global_validation_loss = (
            float("inf")
        )

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        self.server_socket = None

        self.workers = {}

        # ----------------------------------------------------
        # Worker state
        # ----------------------------------------------------

        self.ready_workers = set()

        self.active_workers = set()

        self.stopped_workers = set()

        self.worker_updates = {}

        # ----------------------------------------------------
        # Synchronization
        # ----------------------------------------------------

        self.lock = threading.Lock()

        self.training_complete = False

    # ========================================================
    # NETWORK HELPERS
    # ========================================================

    @staticmethod
    def send_bytes(
        sock,
        data,
    ):

        header = len(
            data
        ).to_bytes(
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
                    "Worker connection closed."
                )

            buffer.extend(
                chunk
            )

        return bytes(
            buffer
        )

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
                "Invalid message size."
            )

        return cls.recv_exact(
            sock,
            size,
        )

    # ========================================================
    # SEND MESSAGE
    # ========================================================

    def send_message(
        self,
        sock,
        message,
    ):

        payload = pickle.dumps(
            message,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        self.send_bytes(
            sock,
            payload,
        )

    # ========================================================
    # WORKER HANDLER
    # ========================================================

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

                message = pickle.loads(
                    payload
                )

                message_type = message.get(
                    "type"
                )

                # --------------------------------------------
                # READY
                # --------------------------------------------

                if message_type == "READY":

                    with self.lock:

                        self.ready_workers.add(
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

                    continue

                # --------------------------------------------
                # MODEL UPDATE
                # --------------------------------------------

                if (
                    message_type
                    == "MODEL_UPDATE"
                ):

                    round_number = int(
                        message.get(
                            "round",
                            0,
                        )
                    )

                    with self.lock:

                        self.worker_updates[
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

                        # IMPORTANT:
                        #
                        # A converged worker remains an
                        # active federated participant.
                        #
                        # It continues receiving the global
                        # model and contributing its latest
                        # model.

                        self.active_workers.add(
                            worker_id
                        )

                    print()

                    print(
                        f"[Server] Worker "
                        f"{worker_id} submitted "
                        f"round {round_number} update."
                    )

                    continue

                # --------------------------------------------
                # UNKNOWN
                # --------------------------------------------

                print(
                    f"[Server] Worker "
                    f"{worker_id} sent unknown "
                    f"message type: "
                    f"{message_type}"
                )

        except Exception as error:

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

    # ========================================================
    # PREPARE HOST DATASET
    # ========================================================

    def prepare_host_dataset(self):

        print()

        print(
            "=" * 70
        )

        print(
            "PREPARING HOST LOCAL DATASET"
        )

        print(
            "=" * 70
        )

        (
            self.host_train_loader,
            self.host_validation_loader,
        ) = get_local_loaders(
            participant_id="HOST",
            batch_size=self.batch_size,
        )

    # ========================================================
    # HOST LOCAL TRAINING
    # ========================================================

    def train_host_local_epoch(self):

        self.model.train()

        total_loss = 0.0

        batches = 0

        start_time = time.time()

        for noisy, clean in (
            self.host_train_loader
        ):

            self.host_optimizer.zero_grad(
                set_to_none=True
            )

            enhanced = self.model(
                noisy
            )

            loss = self.host_criterion(
                enhanced,
                clean,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=5.0,
            )

            self.host_optimizer.step()

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
    # HOST VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate_host(self):

        self.model.eval()

        total_loss = 0.0

        batches = 0

        for noisy, clean in (
            self.host_validation_loader
        ):

            enhanced = self.model(
                noisy
            )

            loss = self.host_criterion(
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
    # HOST EARLY STOPPING
    # ========================================================

    def check_host_early_stopping(
        self,
        validation_loss,
        epoch,
    ):

        if (
            validation_loss
            <
            self.host_best_validation_loss
            - self.participant_min_improvement
        ):

            self.host_best_validation_loss = (
                validation_loss
            )

            self.host_epochs_without_improvement = 0

            improved = True

        else:

            self.host_epochs_without_improvement += 1

            improved = False

        self.host_previous_validation_loss = (
            validation_loss
        )

        if (
            self.host_epochs_without_improvement
            >= self.participant_patience
        ):

            self.host_converged = True

            self.host_convergence_epoch = (
                epoch
            )

        return improved

    # ========================================================
    # HOST UPDATE
    # ========================================================

    def create_host_update(
        self,
        epoch,
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

        return {

            "type":
                "MODEL_UPDATE",

            "worker_id":
                0,

            "participant":
                "HOST",

            "round":
                epoch,

            "loss":
                float(loss),

            "validation_loss":
                float(validation_loss),

            "status":
                (
                    "CONVERGED"
                    if self.host_converged
                    else "TRAINING"
                ),

            "convergence_epoch":
                self.host_convergence_epoch,

            "training_time":
                float(elapsed),

            "batches":
                int(batches),

            "weights":
                weights,
        }

    # ========================================================
    # BROADCAST GLOBAL MODEL
    # ========================================================

    def broadcast_global_model(
        self,
        epoch,
    ):

        state = {
            key:
                value.detach().cpu()
            for key, value
            in self.model.state_dict().items()
        }

        message = {

            "type":
                "GLOBAL_MODEL",

            "round":
                epoch,

            "weights":
                state,
        }

        with self.lock:

            workers_snapshot = dict(
                self.workers
            )

        print()

        print(
            "Broadcasting global model..."
        )

        for worker_id, sock in (
            workers_snapshot.items()
        ):

            try:

                self.send_message(
                    sock,
                    message,
                )

                print(
                    f"  Global model sent "
                    f"to Worker {worker_id}"
                )

            except Exception as error:

                print(
                    f"[Server] Failed to send "
                    f"global model to Worker "
                    f"{worker_id}: "
                    f"{error}"
                )

                with self.lock:

                    self.active_workers.discard(
                        worker_id
                    )

    # ========================================================
    # WAIT FOR READY
    # ========================================================

    def wait_for_all_workers_ready(
        self,
        timeout=1800,
    ):

        print()

        print(
            "=" * 70
        )

        print(
            "WAITING FOR ALL WORKERS TO BE READY"
        )

        print(
            "=" * 70
        )

        deadline = (
            time.time()
            + timeout
        )

        while True:

            with self.lock:

                ready = set(
                    self.ready_workers
                )

            if len(ready) >= self.num_workers:

                print()

                print(
                    "ALL REMOTE WORKERS ARE READY"
                )

                print(
                    f"Ready workers: "
                    f"{sorted(ready)}"
                )

                return

            if time.time() >= deadline:

                missing = (
                    set(
                        range(
                            1,
                            self.num_workers + 1,
                        )
                    )
                    - ready
                )

                raise TimeoutError(
                    "Timed out waiting for "
                    f"workers: "
                    f"{sorted(missing)}"
                )

            print(
                f"\rReady workers: "
                f"{len(ready)}/"
                f"{self.num_workers}",
                end="",
                flush=True,
            )

            time.sleep(
                1
            )

        print()

    # ========================================================
    # WAIT FOR ROUND UPDATES
    # ========================================================

    def wait_for_round_updates(
        self,
        epoch,
        timeout=3600,
    ):

        deadline = (
            time.time()
            + timeout
        )

        required_workers = set(
            range(
                1,
                self.num_workers + 1,
            )
        )

        while True:

            with self.lock:

                received = {
                    worker_id
                    for worker_id, update
                    in self.worker_updates.items()
                    if int(
                        update.get(
                            "round",
                            -1,
                        )
                    ) == epoch
                }

            if received >= required_workers:

                with self.lock:

                    updates = [
                        self.worker_updates[
                            worker_id
                        ]
                        for worker_id in sorted(
                            required_workers
                        )
                    ]

                return updates

            if time.time() >= deadline:

                missing = (
                    required_workers
                    - received
                )

                raise TimeoutError(
                    f"Timed out waiting for "
                    f"round {epoch}. "
                    f"Missing workers: "
                    f"{sorted(missing)}"
                )

            print(
                f"\rWaiting for workers: "
                f"{len(received)}/"
                f"{self.num_workers}",
                end="",
                flush=True,
            )

            time.sleep(
                1
            )

        print()

    # ========================================================
    # FEDERATED AVERAGING
    # ========================================================

    def average_models(
        self,
        updates,
    ):

        if not updates:

            raise RuntimeError(
                "No participant updates."
            )

        states = []

        for update in updates:

            if "weights" not in update:

                continue

            states.append(
                update["weights"]
            )

        if not states:

            raise RuntimeError(
                "No model weights found."
            )

        averaged = {}

        keys = states[0].keys()

        for key in keys:

            tensors = [
                state[key].float()
                for state in states
            ]

            stacked = torch.stack(
                tensors,
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

    # ========================================================
    # GLOBAL EARLY STOPPING
    # ========================================================

    def check_global_early_stopping(
        self,
        validation_loss,
    ):

        if (
            validation_loss
            <
            self.best_global_validation_loss
            - self.global_min_improvement
        ):

            self.best_global_validation_loss = (
                validation_loss
            )

            self.global_epochs_without_improvement = 0

            return True

        self.global_epochs_without_improvement += 1

        return False

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        epoch,
        average_loss,
        average_validation_loss,
    ):

        torch.save(
            {

                "epoch":
                    epoch,

                "model_state_dict":
                    self.model.state_dict(),

                "average_participant_loss":
                    average_loss,

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

                "best_global_validation_loss":
                    self.best_global_validation_loss,

            },
            GLOBAL_MODEL_PATH,
        )

    # ========================================================
    # PRINT UPDATES
    # ========================================================

    def print_participant_status(
        self,
        host_update,
        worker_updates,
    ):

        print()

        print(
            "=" * 70
        )

        print(
            "PARTICIPANT PERFORMANCE"
        )

        print(
            "=" * 70
        )

        all_updates = [
            host_update
        ] + list(
            worker_updates
        )

        for update in all_updates:

            participant = update.get(
                "participant",
                (
                    "Worker "
                    + str(
                        update.get(
                            "worker_id",
                            "?",
                        )
                    )
                ),
            )

            if (
                update.get(
                    "worker_id"
                )
                == 0
            ):

                participant = "HOST"

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

            print(
                f"{participant:<12} "
                f"train={loss:.6f} | "
                f"validation="
                f"{validation_loss:.6f} | "
                f"status={status}"
            )

        print(
            "=" * 70
        )

    # ========================================================
    # CREATE SERVER
    # ========================================================

    def create_server_socket(self):

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

    # ========================================================
    # ACCEPT WORKERS
    # ========================================================

    def accept_workers(self):

        print()

        print(
            f"Waiting for "
            f"{self.num_workers} workers..."
        )

        for worker_id in range(
            1,
            self.num_workers + 1,
        ):

            connection, address = (
                self.server_socket.accept()
            )

            connection.settimeout(
                1800
            )

            with self.lock:

                self.workers[
                    worker_id
                ] = connection

            print()

            print(
                f"Worker {worker_id} "
                f"connected from "
                f"{address}"
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

    # ========================================================
    # SHUTDOWN WORKERS
    # ========================================================

    def shutdown_workers(self):

        print()

        print(
            "Sending shutdown signal..."
        )

        message = {
            "type": "SHUTDOWN"
        }

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
                    message,
                )

            except Exception:

                pass

    # ========================================================
    # START
    # ========================================================

    def start(self):

        print()

        print(
            "=" * 70
        )

        print(
            "S.H.A ANC FEDERATED PARAMETER SERVER"
        )

        print(
            "=" * 70
        )

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
            "=" * 70
        )

        # ----------------------------------------------------
        # Prepare host dataset BEFORE starting rounds
        # ----------------------------------------------------

        self.prepare_host_dataset()

        # ----------------------------------------------------
        # Server
        # ----------------------------------------------------

        self.create_server_socket()

        # ----------------------------------------------------
        # Workers
        # ----------------------------------------------------

        self.accept_workers()

        # ----------------------------------------------------
        # READY SYNCHRONIZATION
        # ----------------------------------------------------

        try:

            self.wait_for_all_workers_ready()

        except TimeoutError as error:

            print()

            print(
                f"[Server] {error}"
            )

            self.shutdown_workers()

            return

        print()

        print(
            "=" * 70
        )

        print(
            "ALL PARTICIPANTS READY"
        )

        print(
            "HOST IS ALSO A TRAINING PARTICIPANT"
        )

        print(
            f"TOTAL PARTICIPANTS = "
            f"{self.num_participants}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # TRAINING
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
                f"FEDERATED ROUND "
                f"{epoch}/{self.epochs}"
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # Reset worker updates
            # ------------------------------------------------

            with self.lock:

                self.worker_updates.clear()

            # ------------------------------------------------
            # Broadcast global model
            # ------------------------------------------------

            self.broadcast_global_model(
                epoch
            )

            # ------------------------------------------------
            # HOST TRAINING
            # ------------------------------------------------

            print()

            print(
                "HOST LOCAL TRAINING"
            )

            if self.host_converged:

                print(
                    "Host participant has converged."
                )

                print(
                    "Skipping host local training."
                )

                host_loss = (
                    self.host_last_loss
                )

                host_validation_loss = (
                    self.host_last_validation_loss
                )

                host_elapsed = 0.0

                host_batches = 0

            else:

                (
                    host_loss,
                    host_elapsed,
                    host_batches,
                ) = (
                    self.train_host_local_epoch()
                )

                self.host_last_loss = (
                    host_loss
                )

                print(
                    f"Host training loss: "
                    f"{host_loss:.6f}"
                )

                print(
                    "HOST VALIDATION"
                )

                host_validation_loss = (
                    self.validate_host()
                )

                self.host_last_validation_loss = (
                    host_validation_loss
                )

                print(
                    f"Host validation loss: "
                    f"{host_validation_loss:.6f}"
                )

                self.check_host_early_stopping(
                    validation_loss=(
                        host_validation_loss
                    ),
                    epoch=epoch,
                )

            # ------------------------------------------------
            # HOST UPDATE
            # ------------------------------------------------

            host_update = (
                self.create_host_update(
                    epoch=epoch,
                    loss=host_loss,
                    validation_loss=(
                        host_validation_loss
                    ),
                    elapsed=host_elapsed,
                    batches=host_batches,
                )
            )

            # ------------------------------------------------
            # WAIT FOR REMOTE WORKERS
            # ------------------------------------------------

            print()

            print(
                "WAITING FOR REMOTE PARTICIPANTS"
            )

            try:

                worker_updates = (
                    self.wait_for_round_updates(
                        epoch=epoch
                    )
                )

            except TimeoutError as error:

                print()

                print(
                    f"[Server] {error}"
                )

                print(
                    "Training stopped safely."
                )

                break

            # ------------------------------------------------
            # Performance
            # ------------------------------------------------

            self.print_participant_status(
                host_update=host_update,
                worker_updates=worker_updates,
            )

            # ------------------------------------------------
            # Combine all 3 participants
            # ------------------------------------------------

            all_updates = [
                host_update
            ] + list(
                worker_updates
            )

            print()

            print(
                "PARTICIPANTS CONTRIBUTING:"
            )

            print(
                "  1. HOST"
            )

            print(
                "  2. WORKER 1"
            )

            print(
                "  3. WORKER 2"
            )

            print()

            print(
                "FEDERATED AVERAGING"
            )

            # ------------------------------------------------
            # Aggregate
            # ------------------------------------------------

            self.average_models(
                all_updates
            )

            # ------------------------------------------------
            # Average metrics
            # ------------------------------------------------

            losses = [
                float(
                    update.get(
                        "loss",
                        0.0,
                    )
                )
                for update in all_updates
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
                for update in all_updates
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

            print(
                f"Average participant loss: "
                f"{average_loss:.6f}"
            )

            print(
                f"Average participant validation "
                f"loss: "
                f"{average_validation_loss:.6f}"
            )

            # ------------------------------------------------
            # Global early stopping
            # ------------------------------------------------

            global_improved = (
                self.check_global_early_stopping(
                    average_validation_loss
                )
            )

            if global_improved:

                print()

                print(
                    "GLOBAL MODEL IMPROVED"
                )

                print(
                    f"Best global validation loss: "
                    f"{self.best_global_validation_loss:.6f}"
                )

            else:

                print()

                print(
                    "Global model did not improve "
                    "sufficiently."
                )

                print(
                    f"Global patience: "
                    f"{self.global_epochs_without_improvement}/"
                    f"{self.global_patience}"
                )

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            self.save_checkpoint(
                epoch=epoch,
                average_loss=average_loss,
                average_validation_loss=(
                    average_validation_loss
                ),
            )

            print()

            print(
                f"Global model checkpoint saved:"
            )

            print(
                f"  {GLOBAL_MODEL_PATH}"
            )

            # ------------------------------------------------
            # Participant convergence
            # ------------------------------------------------

            converged_workers = [
                update.get(
                    "worker_id"
                )
                for update
                in worker_updates
                if update.get(
                    "status"
                ) == "CONVERGED"
            ]

            if converged_workers:

                with self.lock:

                    self.stopped_workers.update(
                        converged_workers
                    )

                print()

                print(
                    "Participants currently "
                    "converged:"
                )

                print(
                    f"  Workers: "
                    f"{sorted(converged_workers)}"
                )

            if self.host_converged:

                print()

                print(
                    "HOST PARTICIPANT: CONVERGED"
                )

            # ------------------------------------------------
            # Global stop
            # ------------------------------------------------

            if (
                self.global_epochs_without_improvement
                >= self.global_patience
            ):

                print()

                print(
                    "=" * 70
                )

                print(
                    "GLOBAL EARLY STOPPING TRIGGERED"
                )

                print(
                    f"No sufficient global "
                    f"validation improvement for "
                    f"{self.global_patience} rounds."
                )

                print(
                    "=" * 70
                )

                self.training_complete = True

                break

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        print()

        print(
            "=" * 70
        )

        print(
            "S.H.A ANC FEDERATED TRAINING COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Final model:"
        )

        print(
            f"{GLOBAL_MODEL_PATH}"
        )

        print()

        print(
            f"Best global validation loss: "
            f"{self.best_global_validation_loss:.6f}"
        )

        print(
            f"Host converged: "
            f"{self.host_converged}"
        )

        print(
            f"Host convergence round: "
            f"{self.host_convergence_epoch}"
        )

        print(
            f"Workers currently converged: "
            f"{sorted(self.stopped_workers)}"
        )

        print()

        print(
            "Federated participants:"
        )

        print(
            "  HOST"
        )

        print(
            "  WORKER 1"
        )

        print(
            "  WORKER 2"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Shutdown
        # ----------------------------------------------------

        self.shutdown_workers()

        time.sleep(
            1
        )

        with self.lock:

            workers_snapshot = dict(
                self.workers
            )

        for sock in (
            workers_snapshot.values()
        ):

            try:
                sock.close()

            except Exception:
                pass

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
            "S.H.A ANC synchronized federated "
            "parameter server"
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
        "--patience",
        type=int,
        default=PATIENCE,
    )

    parser.add_argument(
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT,
    )

    parser.add_argument(
        "--global-patience",
        type=int,
        default=GLOBAL_PATIENCE,
    )

    parser.add_argument(
        "--global-min-improvement",
        type=float,
        default=GLOBAL_MIN_IMPROVEMENT,
    )

    args = parser.parse_args()

    server = SHAANCParameterServer(

        port=args.port,

        num_workers=args.num_workers,

        epochs=args.epochs,

        batch_size=args.batch_size,

        learning_rate=args.lr,

        participant_patience=(
            args.patience
        ),

        participant_min_improvement=(
            args.min_improvement
        ),

        global_patience=(
            args.global_patience
        ),

        global_min_improvement=(
            args.global_min_improvement
        ),
    )

    server.start()


if __name__ == "__main__":

    main()