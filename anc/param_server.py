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
)

from dataset import get_local_loaders
from model import create_model


# ============================================================
# S.H.A ANC FEDERATED PARAMETER SERVER
#
# HOST = PARAMETER SERVER + TRAINING PARTICIPANT
#
# PARTICIPANTS:
#
#   HOST
#   WORKER 1
#   WORKER 2
#
# TOTAL = 3
# ============================================================

class SHAANCParameterServer:

    def __init__(
        self,
        port,
        num_workers,
        epochs,
        batch_size,
        learning_rate,
        min_improvement,
        patience,
        global_min_improvement,
        global_patience,
    ):

        self.port = port

        # Two remote workers.
        self.num_workers = num_workers

        # Host + workers.
        self.num_participants = (
            num_workers + 1
        )

        self.epochs = epochs

        self.batch_size = batch_size

        self.learning_rate = (
            learning_rate
        )

        # ----------------------------------------------------
        # Participant early stopping
        # ----------------------------------------------------

        self.min_improvement = (
            min_improvement
        )

        self.patience = max(
            int(patience),
            1,
        )

        # ----------------------------------------------------
        # Global early stopping
        # ----------------------------------------------------

        self.global_min_improvement = (
            global_min_improvement
        )

        self.global_patience = max(
            int(global_patience),
            1,
        )

        # ----------------------------------------------------
        # Host model
        # ----------------------------------------------------

        self.model = create_model()

        self.criterion = nn.L1Loss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # ----------------------------------------------------
        # Host dataset
        # ----------------------------------------------------

        self.train_loader = None
        self.validation_loader = None

        # ----------------------------------------------------
        # Worker connections
        # ----------------------------------------------------

        self.workers = {}

        self.active_workers = set()

        self.stopped_workers = set()

        # ----------------------------------------------------
        # Current round
        # ----------------------------------------------------

        self.updates = {}

        self.lock = threading.Lock()

        self.server_socket = None

        # ----------------------------------------------------
        # Host early stopping
        # ----------------------------------------------------

        self.host_best_validation_loss = float(
            "inf"
        )

        self.host_previous_validation_loss = None

        self.host_epochs_without_improvement = 0

        self.host_converged = False

        self.host_convergence_epoch = None

        # ----------------------------------------------------
        # Global early stopping
        # ----------------------------------------------------

        self.best_global_validation_loss = float(
            "inf"
        )

        self.global_epochs_without_improvement = 0

        self.training_complete = False

    # ========================================================
    # NETWORK HELPERS
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
                    "Connection closed."
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

        return cls.recv_exact(
            sock,
            size,
        )

    # ========================================================
    # PREPARE HOST DATA
    # ========================================================

    def prepare_host_dataset(self):

        if self.train_loader is not None:

            return

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
            self.train_loader,
            self.validation_loader,
        ) = get_local_loaders(
            participant_id="HOST",
            batch_size=self.batch_size,
        )

    # ========================================================
    # HOST LOCAL TRAINING
    # ========================================================

    def train_host_epoch(self):

        if self.host_converged:

            return (
                0.0,
                0.0,
                0,
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
    # HOST VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate_host(self):

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
    # HOST EARLY STOPPING
    # ========================================================

    def check_host_convergence(
        self,
        validation_loss,
        epoch,
    ):

        if (
            self.host_previous_validation_loss
            is None
        ):

            self.host_previous_validation_loss = (
                validation_loss
            )

            self.host_best_validation_loss = (
                validation_loss
            )

            return False

        improvement = (
            self.host_best_validation_loss
            - validation_loss
        )

        if (
            validation_loss
            < self.host_best_validation_loss
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

        self.host_previous_validation_loss = (
            validation_loss
        )

        if (
            self.host_epochs_without_improvement
            >= self.patience
        ):

            self.host_converged = True

            self.host_convergence_epoch = (
                epoch
            )

            return True

        return False

    # ========================================================
    # WORKER CONNECTION HANDLER
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

                update = pickle.loads(
                    payload
                )

                with self.lock:

                    self.updates[
                        worker_id
                    ] = update

                    status = update.get(
                        "status",
                        "TRAINING",
                    )

                    if status == "CONVERGED":

                        self.stopped_workers.add(
                            worker_id
                        )

                        self.active_workers.discard(
                            worker_id
                        )

                        print(
                            f"[Server] Worker "
                            f"{worker_id} "
                            f"CONVERGED."
                        )

                    else:

                        self.active_workers.add(
                            worker_id
                        )

        except Exception as error:

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
    # BROADCAST GLOBAL MODEL
    # ========================================================

    def broadcast_model(self):

        state = {
            key:
                value.detach().cpu()
            for key, value
            in self.model.state_dict().items()
        }

        payload = pickle.dumps(
            state,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        with self.lock:

            workers_snapshot = {
                worker_id: sock
                for worker_id, sock
                in self.workers.items()
                if worker_id
                in self.active_workers
            }

        for worker_id, sock in (
            workers_snapshot.items()
        ):

            try:

                self.send_bytes(
                    sock,
                    payload,
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
    # WAIT FOR WORKER UPDATES
    # ========================================================

    def wait_for_worker_updates(
        self,
        timeout=1800,
    ):

        deadline = (
            time.time()
            + timeout
        )

        while True:

            with self.lock:

                active_ids = set(
                    self.active_workers
                )

                received_ids = set(
                    self.updates.keys()
                )

                stopped_this_round = {
                    worker_id
                    for worker_id, update
                    in self.updates.items()
                    if update.get(
                        "status",
                        "TRAINING",
                    ) == "CONVERGED"
                }

                required_ids = (
                    active_ids
                    | stopped_this_round
                )

                if received_ids >= required_ids:

                    complete = True

                else:

                    complete = False

            if complete:

                break

            if time.time() > deadline:

                with self.lock:

                    missing = (
                        required_ids
                        - received_ids
                    )

                raise TimeoutError(
                    "Timed out waiting for workers. "
                    f"Missing: {sorted(missing)}"
                )

            time.sleep(
                0.1
            )

        with self.lock:

            updates = list(
                self.updates.values()
            )

        return sorted(
            updates,
            key=lambda update:
                update.get(
                    "worker_id",
                    0,
                ),
        )

    # ========================================================
    # FEDERATED AVERAGING
    # ========================================================

    def federated_average(
        self,
        participant_updates,
    ):
        """
        Sample-weighted FedAvg.

        Includes:

            Host
            Worker 1
            Worker 2
        """

        valid_updates = [
            update
            for update
            in participant_updates
            if "weights" in update
        ]

        if not valid_updates:

            raise RuntimeError(
                "No valid participant models."
            )

        total_samples = sum(
            max(
                int(
                    update.get(
                        "num_samples",
                        1,
                    )
                ),
                1,
            )
            for update
            in valid_updates
        )

        print()
        print(
            "FEDERATED AGGREGATION"
        )

        print(
            f"Participants contributing: "
            f"{len(valid_updates)}"
        )

        print(
            f"Total training samples: "
            f"{total_samples}"
        )

        averaged = {}

        first_state = (
            valid_updates[0]["weights"]
        )

        for key in first_state.keys():

            accumulator = None

            for update in valid_updates:

                state = update[
                    "weights"
                ]

                samples = max(
                    int(
                        update.get(
                            "num_samples",
                            1,
                        )
                    ),
                    1,
                )

                weight = (
                    samples
                    / total_samples
                )

                tensor = (
                    state[key]
                    .float()
                )

                contribution = (
                    tensor
                    * weight
                )

                if accumulator is None:

                    accumulator = (
                        contribution.clone()
                    )

                else:

                    accumulator += (
                        contribution
                    )

            averaged[key] = (
                accumulator
            )

        self.model.load_state_dict(
            averaged
        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        epoch,
        host_loss,
        host_validation_loss,
        global_validation_loss,
        participants,
    ):

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    self.model.state_dict(),

                "host_training_loss":
                    host_loss,

                "host_validation_loss":
                    host_validation_loss,

                "global_validation_loss":
                    global_validation_loss,

                "num_workers":
                    self.num_workers,

                "num_participants":
                    self.num_participants,

                "stopped_workers":
                    sorted(
                        self.stopped_workers
                    ),

                "participants":
                    participants,
            },
            GLOBAL_MODEL_PATH,
        )

    # ========================================================
    # GLOBAL EARLY STOPPING
    # ========================================================

    def check_global_early_stopping(
        self,
        global_validation_loss,
    ):

        improvement = (
            self.best_global_validation_loss
            - global_validation_loss
        )

        if (
            global_validation_loss
            < self.best_global_validation_loss
        ):

            self.best_global_validation_loss = (
                global_validation_loss
            )

        if improvement >= (
            self.global_min_improvement
        ):

            self.global_epochs_without_improvement = 0

        else:

            self.global_epochs_without_improvement += 1

        return (
            self.global_epochs_without_improvement
            >= self.global_patience
        )

    # ========================================================
    # PRINT PARTICIPANT STATUS
    # ========================================================

    def print_participant_status(
        self,
        host_loss,
        host_validation_loss,
        updates,
    ):

        print()
        print(
            "PARTICIPANT PERFORMANCE"
        )

        print(
            "-" * 70
        )

        print(
            f"HOST:"
        )

        print(
            f"  Train loss       : "
            f"{host_loss:.6f}"
        )

        print(
            f"  Validation loss  : "
            f"{host_validation_loss:.6f}"
        )

        print(
            f"  Status           : "
            f"{'CONVERGED' if self.host_converged else 'TRAINING'}"
        )

        for update in updates:

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

            samples = int(
                update.get(
                    "num_samples",
                    0,
                )
            )

            print()

            print(
                f"WORKER {worker_id}:"
            )

            print(
                f"  Train loss       : "
                f"{loss:.6f}"
            )

            print(
                f"  Validation loss  : "
                f"{validation_loss:.6f}"
            )

            print(
                f"  Samples          : "
                f"{samples}"
            )

            print(
                f"  Status           : "
                f"{status}"
            )

        print(
            "-" * 70
        )

    # ========================================================
    # START SERVER
    # ========================================================

    def start(self):

        self.prepare_host_dataset()

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
            f"Host participant : YES"
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
            f"Learning rate    : "
            f"{self.learning_rate}"
        )

        print(
            f"Participant patience: "
            f"{self.patience}"
        )

        print(
            f"Global patience  : "
            f"{self.global_patience}"
        )

        print(
            "=" * 70
        )

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

            connection.settimeout(
                1800
            )

            self.workers[
                worker_id
            ] = connection

            self.active_workers.add(
                worker_id
            )

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

        print()
        print(
            "=" * 70
        )

        print(
            "ALL REMOTE WORKERS CONNECTED"
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

        # ====================================================
        # FEDERATED TRAINING
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
                f"FEDERATED ROUND "
                f"{epoch}/{self.epochs}"
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # RESET WORKER UPDATES
            # ------------------------------------------------

            with self.lock:

                self.updates.clear()

            # ------------------------------------------------
            # BROADCAST CURRENT GLOBAL MODEL
            # ------------------------------------------------

            print()
            print(
                "Broadcasting global model..."
            )

            self.broadcast_model()

            # ------------------------------------------------
            # HOST LOCAL TRAINING
            # ------------------------------------------------

            print()
            print(
                "HOST LOCAL TRAINING"
            )

            if self.host_converged:

                print(
                    "Host has converged."
                )

                print(
                    "Skipping host model update."
                )

                host_loss = 0.0

                host_training_time = 0.0

                host_batches = 0

            else:

                (
                    host_loss,
                    host_training_time,
                    host_batches,
                ) = self.train_host_epoch()

            # ------------------------------------------------
            # HOST VALIDATION
            # ------------------------------------------------

            host_validation_loss = (
                self.validate_host()
            )

            host_reached_convergence = (
                self.check_host_convergence(
                    host_validation_loss,
                    epoch,
                )
            )

            print()
            print(
                f"Host train loss      : "
                f"{host_loss:.6f}"
            )

            print(
                f"Host validation loss : "
                f"{host_validation_loss:.6f}"
            )

            print(
                f"Host best validation : "
                f"{self.host_best_validation_loss:.6f}"
            )

            print(
                f"Host patience        : "
                f"{self.host_epochs_without_improvement}/"
                f"{self.patience}"
            )

            if host_reached_convergence:

                print()
                print(
                    "HOST EARLY STOPPING:"
                )

                print(
                    f"Host converged at "
                    f"round {epoch}."
                )

            # ------------------------------------------------
            # HOST UPDATE
            # ------------------------------------------------

            host_weights = {
                key:
                    value.detach().cpu()
                for key, value
                in self.model.state_dict().items()
            }

            host_update = {

                "participant_type":
                    "host",

                "participant_id":
                    "HOST",

                "worker_id":
                    0,

                "loss":
                    float(host_loss),

                "validation_loss":
                    float(host_validation_loss),

                "status":
                    (
                        "CONVERGED"
                        if self.host_converged
                        else "TRAINING"
                    ),

                "training_time":
                    float(host_training_time),

                "batches":
                    int(host_batches),

                "num_samples":
                    len(
                        self.train_loader.dataset
                    ),

                "weights":
                    host_weights,
            }

            # ------------------------------------------------
            # WAIT FOR WORKERS
            # ------------------------------------------------

            print()
            print(
                "Waiting for worker updates..."
            )

            try:

                worker_updates = (
                    self.wait_for_worker_updates()
                )

            except TimeoutError as error:

                print()
                print(
                    f"[Server] {error}"
                )

                print(
                    "Ending training safely."
                )

                break

            # ------------------------------------------------
            # COMBINE HOST + WORKERS
            # ------------------------------------------------

            participant_updates = (
                [host_update]
                + worker_updates
            )

            self.print_participant_status(
                host_loss,
                host_validation_loss,
                worker_updates,
            )

            # ------------------------------------------------
            # FEDERATED AVERAGE
            # ------------------------------------------------

            print()
            print(
                "Performing weighted FedAvg..."
            )

            self.federated_average(
                participant_updates
            )

            # ------------------------------------------------
            # GLOBAL VALIDATION
            #
            # The new aggregated model is evaluated using
            # the HOST validation set.
            #
            # This validation data remains local to the host.
            # ------------------------------------------------

            global_validation_loss = (
                self.validate_host()
            )

            # ------------------------------------------------
            # GLOBAL TRAINING LOSS
            # ------------------------------------------------

            participant_losses = [
                float(
                    update.get(
                        "loss",
                        0.0,
                    )
                )
                for update
                in participant_updates
            ]

            average_training_loss = (
                sum(
                    participant_losses
                )
                / max(
                    len(
                        participant_losses
                    ),
                    1,
                )
            )

            print()
            print(
                "=" * 70
            )

            print(
                "GLOBAL MODEL PERFORMANCE"
            )

            print(
                "=" * 70
            )

            print(
                f"Federated train loss : "
                f"{average_training_loss:.6f}"
            )

            print(
                f"Global validation loss: "
                f"{global_validation_loss:.6f}"
            )

            print(
                f"Best global validation: "
                f"{self.best_global_validation_loss:.6f}"
            )

            # ------------------------------------------------
            # GLOBAL EARLY STOPPING
            # ------------------------------------------------

            global_stop = (
                self.check_global_early_stopping(
                    global_validation_loss
                )
            )

            print(
                f"Global patience      : "
                f"{self.global_epochs_without_improvement}/"
                f"{self.global_patience}"
            )

            if global_stop:

                print()
                print(
                    "=" * 70
                )

                print(
                    "GLOBAL EARLY STOPPING"
                )

                print(
                    "=" * 70
                )

                print(
                    "The global validation loss "
                    "has stopped improving."
                )

            # ------------------------------------------------
            # CHECKPOINT
            # ------------------------------------------------

            self.save_checkpoint(
                epoch=epoch,
                host_loss=host_loss,
                host_validation_loss=(
                    host_validation_loss
                ),
                global_validation_loss=(
                    global_validation_loss
                ),
                participants=[
                    "HOST"
                ]
                + [
                    f"WORKER_{u.get('worker_id')}"
                    for u in worker_updates
                ],
            )

            print()
            print(
                f"Global model saved to:"
            )

            print(
                f"{GLOBAL_MODEL_PATH}"
            )

            # ------------------------------------------------
            # STOP AFTER GLOBAL EARLY STOPPING
            # ------------------------------------------------

            if global_stop:

                self.training_complete = True

                break

            # ------------------------------------------------
            # ACTIVE WORKERS
            # ------------------------------------------------

            with self.lock:

                active_count = len(
                    self.active_workers
                )

                stopped_count = len(
                    self.stopped_workers
                )

            print()
            print(
                f"Active workers  : "
                f"{active_count}"
            )

            print(
                f"Stopped workers : "
                f"{stopped_count}"
            )

            # ------------------------------------------------
            # ALL WORKERS CONVERGED
            #
            # We still allow the host/global model to continue
            # until global early stopping or max epochs.
            # ------------------------------------------------

            if active_count == 0:

                print()
                print(
                    "All remote workers have "
                    "converged."
                )

                # Host can continue participating.
                #
                # Global early stopping remains the final
                # criterion for the federation.

        # ====================================================
        # TRAINING COMPLETE
        # ====================================================

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
            f"Participants       : "
            f"{self.num_participants}"
        )

        print(
            f"Remote workers     : "
            f"{self.num_workers}"
        )

        print(
            f"Best global loss   : "
            f"{self.best_global_validation_loss:.6f}"
        )

        print(
            f"Stopped workers    : "
            f"{sorted(self.stopped_workers)}"
        )

        print(
            f"Host converged     : "
            f"{self.host_converged}"
        )

        if self.host_convergence_epoch:

            print(
                f"Host convergence round: "
                f"{self.host_convergence_epoch}"
            )

        print()
        print(
            f"Final model:"
        )

        print(
            f"{GLOBAL_MODEL_PATH}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # CLOSE CONNECTIONS
        # ----------------------------------------------------

        for sock in self.workers.values():

            try:
                sock.close()
            except Exception:
                pass

        if self.server_socket:

            try:
                self.server_socket.close()
            except Exception:
                pass


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "S.H.A ANC federated parameter server "
            "with host training and early stopping."
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
        help=(
            "Number of remote workers. "
            "Default = 2."
        ),
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
        "--global-min-improvement",
        type=float,
        default=GLOBAL_MIN_IMPROVEMENT,
    )

    parser.add_argument(
        "--global-patience",
        type=int,
        default=GLOBAL_PATIENCE,
    )

    args = parser.parse_args()

    server = SHAANCParameterServer(
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_improvement=(
            args.min_improvement
        ),
        patience=args.patience,
        global_min_improvement=(
            args.global_min_improvement
        ),
        global_patience=(
            args.global_patience
        ),
    )

    server.start()


if __name__ == "__main__":

    main()