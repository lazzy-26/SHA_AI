import argparse
import pickle
import random
import socket
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
# WORKER
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

        self.min_improvement = float(
            min_improvement
        )

        self.patience = max(
            int(patience),
            1,
        )

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
        # Networking
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

        self.best_validation_loss = (
            float("inf")
        )

        self.previous_validation_loss = (
            None
        )

        self.epochs_without_improvement = 0

        self.converged = False

        self.convergence_epoch = None

        self.last_loss = float("inf")

        self.last_validation_loss = (
            float("inf")
        )

    # ========================================================
    # NETWORK
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
                    "Server closed connection."
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
                "Received invalid message size."
            )

        return cls.recv_exact(
            sock,
            size,
        )

    # ========================================================
    # CONNECT
    # ========================================================

    def connect(
        self,
        max_retries=60,
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

            except OSError as error:

                print(
                    f"[Worker {self.worker_id}] "
                    f"Waiting for server "
                    f"({attempt}/{max_retries})..."
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
    # RECEIVE MESSAGE
    # ========================================================

    def receive_message(self):

        payload = self.recv_bytes(
            self.socket
        )

        return pickle.loads(
            payload
        )

    # ========================================================
    # PREPARE DATASET
    # ========================================================

    def prepare_dataset(self):

        print()

        print(
            f"[Worker {self.worker_id}] "
            "Preparing local "
            "LibriSpeech/MUSAN dataset..."
        )

        (
            self.train_loader,
            self.validation_loader,
        ) = get_local_loaders(
            participant_id=(
                f"WORKER {self.worker_id}"
            ),
            batch_size=self.batch_size,
        )

        print(
            f"[Worker {self.worker_id}] "
            "LOCAL DATASET READY"
        )

    # ========================================================
    # READY HANDSHAKE
    # ========================================================

    def send_ready(self):

        print()

        print(
            f"[Worker {self.worker_id}] "
            "Sending READY signal..."
        )

        self.send_message(
            {
                "type": "READY",
                "worker_id": self.worker_id,
            }
        )

        print(
            f"[Worker {self.worker_id}] "
            "READY sent."
        )

    # ========================================================
    # WAIT FOR SERVER
    # ========================================================

    def wait_for_round(
        self,
    ):

        message = self.receive_message()

        message_type = message.get(
            "type"
        )

        if message_type == "SHUTDOWN":

            return (
                "SHUTDOWN",
                message,
            )

        if message_type != "GLOBAL_MODEL":

            raise RuntimeError(
                f"Unexpected server message: "
                f"{message_type}"
            )

        return (
            "GLOBAL_MODEL",
            message,
        )

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    def load_global_model(
        self,
        message,
    ):

        weights = message.get(
            "weights"
        )

        if weights is None:

            raise RuntimeError(
                "Global model did not contain weights."
            )

        self.model.load_state_dict(
            weights
        )

    # ========================================================
    # LOCAL TRAINING
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

    def check_early_stopping(
        self,
        validation_loss,
        epoch,
    ):

        if (
            validation_loss
            <
            self.best_validation_loss
            - self.min_improvement
        ):

            self.best_validation_loss = (
                validation_loss
            )

            self.epochs_without_improvement = 0

            improved = True

        else:

            self.epochs_without_improvement += 1

            improved = False

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

            return (
                True,
                improved,
            )

        return (
            False,
            improved,
        )

    # ========================================================
    # SEND UPDATE
    # ========================================================

    def send_update(
        self,
        round_number,
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

        message = {

            "type":
                "MODEL_UPDATE",

            "worker_id":
                self.worker_id,

            "round":
                round_number,

            "loss":
                float(loss),

            "validation_loss":
                float(validation_loss),

            "status":
                (
                    "CONVERGED"
                    if self.converged
                    else "TRAINING"
                ),

            "convergence_epoch":
                self.convergence_epoch,

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
    # PRINT STATUS
    # ========================================================

    def print_status(
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
            f"Training loss       : "
            f"{loss:.6f}"
        )

        print(
            f"Validation loss     : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Best validation     : "
            f"{self.best_validation_loss:.6f}"
        )

        print(
            f"Patience            : "
            f"{self.epochs_without_improvement}/"
            f"{self.patience}"
        )

        print(
            f"Training time       : "
            f"{elapsed:.2f}s"
        )

        print(
            f"Status              : "
            f"{'CONVERGED' if self.converged else 'TRAINING'}"
        )

        print(
            "-" * 70
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

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
            f"Server       : "
            f"{self.server_ip}:{self.port}"
        )

        print(
            f"Worker ID    : "
            f"{self.worker_id}"
        )

        print(
            f"Workers      : "
            f"{self.num_workers}"
        )

        print(
            f"Epochs       : "
            f"{self.epochs}"
        )

        print(
            f"Batch size   : "
            f"{self.batch_size}"
        )

        print(
            f"Learning rate: "
            f"{self.learning_rate}"
        )

        print(
            f"Patience     : "
            f"{self.patience}"
        )

        print(
            "=" * 70
        )

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
            f"[Worker {self.worker_id}] "
            "Waiting for server to start "
            "federated training..."
        )

        # ----------------------------------------------------
        # ROUNDS
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
                f"WAITING FOR ROUND "
                f"{epoch}"
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # Receive server command
            # ------------------------------------------------

            (
                message_type,
                message,
            ) = self.wait_for_round()

            if message_type == "SHUTDOWN":

                print(
                    f"[Worker {self.worker_id}] "
                    "Server requested shutdown."
                )

                break

            round_number = int(
                message.get(
                    "round",
                    epoch,
                )
            )

            # ------------------------------------------------
            # Load global model
            # ------------------------------------------------

            print(
                f"[Worker {self.worker_id}] "
                "Receiving global model..."
            )

            self.load_global_model(
                message
            )

            print(
                f"[Worker {self.worker_id}] "
                "Global model received."
            )

            # ------------------------------------------------
            # Local training
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

                self.send_update(
                    round_number=round_number,
                    loss=self.last_loss,
                    validation_loss=(
                        self.last_validation_loss
                    ),
                    elapsed=0.0,
                    batches=0,
                )

                continue

            print()

            print(
                f"[Worker {self.worker_id}] "
                "Training local "
                "LibriSpeech/MUSAN data..."
            )

            (
                loss,
                elapsed,
                batches,
            ) = self.train_local_epoch()

            self.last_loss = loss

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            print(
                f"[Worker {self.worker_id}] "
                "Validating local model..."
            )

            validation_loss = (
                self.validate()
            )

            self.last_validation_loss = (
                validation_loss
            )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            (
                reached_convergence,
                _,
            ) = self.check_early_stopping(
                validation_loss=validation_loss,
                epoch=epoch,
            )

            self.print_status(
                epoch=epoch,
                loss=loss,
                validation_loss=validation_loss,
                elapsed=elapsed,
            )

            # ------------------------------------------------
            # Convergence announcement
            # ------------------------------------------------

            if reached_convergence:

                print()

                print(
                    f"[Worker {self.worker_id}] "
                    "EARLY STOPPING TRIGGERED"
                )

                print(
                    f"Converged at round: "
                    f"{epoch}"
                )

                print(
                    f"Best validation loss: "
                    f"{self.best_validation_loss:.6f}"
                )

                print(
                    "Participant will remain "
                    "in federated synchronization "
                    "using its latest model."
                )

            # ------------------------------------------------
            # Send update
            # ------------------------------------------------

            print()

            print(
                f"[Worker {self.worker_id}] "
                "Sending model update..."
            )

            self.send_update(
                round_number=round_number,
                loss=loss,
                validation_loss=validation_loss,
                elapsed=elapsed,
                batches=batches,
            )

            print(
                f"[Worker {self.worker_id}] "
                "Update sent."
            )

        # ----------------------------------------------------
        # Finish
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

        else:

            print(
                "Maximum training rounds completed."
            )

        print(
            "=" * 70
        )

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
            "S.H.A ANC synchronized federated worker"
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
    main()import argparse
import pickle
import random
import socket
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
# WORKER
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

        self.min_improvement = float(
            min_improvement
        )

        self.patience = max(
            int(patience),
            1,
        )

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
        # Networking
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

        self.best_validation_loss = (
            float("inf")
        )

        self.previous_validation_loss = (
            None
        )

        self.epochs_without_improvement = 0

        self.converged = False

        self.convergence_epoch = None

        self.last_loss = float("inf")

        self.last_validation_loss = (
            float("inf")
        )

    # ========================================================
    # NETWORK
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
                    "Server closed connection."
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
                "Received invalid message size."
            )

        return cls.recv_exact(
            sock,
            size,
        )

    # ========================================================
    # CONNECT
    # ========================================================

    def connect(
        self,
        max_retries=60,
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

            except OSError as error:

                print(
                    f"[Worker {self.worker_id}] "
                    f"Waiting for server "
                    f"({attempt}/{max_retries})..."
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
    # RECEIVE MESSAGE
    # ========================================================

    def receive_message(self):

        payload = self.recv_bytes(
            self.socket
        )

        return pickle.loads(
            payload
        )

    # ========================================================
    # PREPARE DATASET
    # ========================================================

    def prepare_dataset(self):

        print()

        print(
            f"[Worker {self.worker_id}] "
            "Preparing local "
            "LibriSpeech/MUSAN dataset..."
        )

        (
            self.train_loader,
            self.validation_loader,
        ) = get_local_loaders(
            participant_id=(
                f"WORKER {self.worker_id}"
            ),
            batch_size=self.batch_size,
        )

        print(
            f"[Worker {self.worker_id}] "
            "LOCAL DATASET READY"
        )

    # ========================================================
    # READY HANDSHAKE
    # ========================================================

    def send_ready(self):

        print()

        print(
            f"[Worker {self.worker_id}] "
            "Sending READY signal..."
        )

        self.send_message(
            {
                "type": "READY",
                "worker_id": self.worker_id,
            }
        )

        print(
            f"[Worker {self.worker_id}] "
            "READY sent."
        )

    # ========================================================
    # WAIT FOR SERVER
    # ========================================================

    def wait_for_round(
        self,
    ):

        message = self.receive_message()

        message_type = message.get(
            "type"
        )

        if message_type == "SHUTDOWN":

            return (
                "SHUTDOWN",
                message,
            )

        if message_type != "GLOBAL_MODEL":

            raise RuntimeError(
                f"Unexpected server message: "
                f"{message_type}"
            )

        return (
            "GLOBAL_MODEL",
            message,
        )

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    def load_global_model(
        self,
        message,
    ):

        weights = message.get(
            "weights"
        )

        if weights is None:

            raise RuntimeError(
                "Global model did not contain weights."
            )

        self.model.load_state_dict(
            weights
        )

    # ========================================================
    # LOCAL TRAINING
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

    def check_early_stopping(
        self,
        validation_loss,
        epoch,
    ):

        if (
            validation_loss
            <
            self.best_validation_loss
            - self.min_improvement
        ):

            self.best_validation_loss = (
                validation_loss
            )

            self.epochs_without_improvement = 0

            improved = True

        else:

            self.epochs_without_improvement += 1

            improved = False

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

            return (
                True,
                improved,
            )

        return (
            False,
            improved,
        )

    # ========================================================
    # SEND UPDATE
    # ========================================================

    def send_update(
        self,
        round_number,
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

        message = {

            "type":
                "MODEL_UPDATE",

            "worker_id":
                self.worker_id,

            "round":
                round_number,

            "loss":
                float(loss),

            "validation_loss":
                float(validation_loss),

            "status":
                (
                    "CONVERGED"
                    if self.converged
                    else "TRAINING"
                ),

            "convergence_epoch":
                self.convergence_epoch,

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
    # PRINT STATUS
    # ========================================================

    def print_status(
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
            f"Training loss       : "
            f"{loss:.6f}"
        )

        print(
            f"Validation loss     : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Best validation     : "
            f"{self.best_validation_loss:.6f}"
        )

        print(
            f"Patience            : "
            f"{self.epochs_without_improvement}/"
            f"{self.patience}"
        )

        print(
            f"Training time       : "
            f"{elapsed:.2f}s"
        )

        print(
            f"Status              : "
            f"{'CONVERGED' if self.converged else 'TRAINING'}"
        )

        print(
            "-" * 70
        )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

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
            f"Server       : "
            f"{self.server_ip}:{self.port}"
        )

        print(
            f"Worker ID    : "
            f"{self.worker_id}"
        )

        print(
            f"Workers      : "
            f"{self.num_workers}"
        )

        print(
            f"Epochs       : "
            f"{self.epochs}"
        )

        print(
            f"Batch size   : "
            f"{self.batch_size}"
        )

        print(
            f"Learning rate: "
            f"{self.learning_rate}"
        )

        print(
            f"Patience     : "
            f"{self.patience}"
        )

        print(
            "=" * 70
        )

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
            f"[Worker {self.worker_id}] "
            "Waiting for server to start "
            "federated training..."
        )

        # ----------------------------------------------------
        # ROUNDS
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
                f"WAITING FOR ROUND "
                f"{epoch}"
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # Receive server command
            # ------------------------------------------------

            (
                message_type,
                message,
            ) = self.wait_for_round()

            if message_type == "SHUTDOWN":

                print(
                    f"[Worker {self.worker_id}] "
                    "Server requested shutdown."
                )

                break

            round_number = int(
                message.get(
                    "round",
                    epoch,
                )
            )

            # ------------------------------------------------
            # Load global model
            # ------------------------------------------------

            print(
                f"[Worker {self.worker_id}] "
                "Receiving global model..."
            )

            self.load_global_model(
                message
            )

            print(
                f"[Worker {self.worker_id}] "
                "Global model received."
            )

            # ------------------------------------------------
            # Local training
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

                self.send_update(
                    round_number=round_number,
                    loss=self.last_loss,
                    validation_loss=(
                        self.last_validation_loss
                    ),
                    elapsed=0.0,
                    batches=0,
                )

                continue

            print()

            print(
                f"[Worker {self.worker_id}] "
                "Training local "
                "LibriSpeech/MUSAN data..."
            )

            (
                loss,
                elapsed,
                batches,
            ) = self.train_local_epoch()

            self.last_loss = loss

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            print(
                f"[Worker {self.worker_id}] "
                "Validating local model..."
            )

            validation_loss = (
                self.validate()
            )

            self.last_validation_loss = (
                validation_loss
            )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            (
                reached_convergence,
                _,
            ) = self.check_early_stopping(
                validation_loss=validation_loss,
                epoch=epoch,
            )

            self.print_status(
                epoch=epoch,
                loss=loss,
                validation_loss=validation_loss,
                elapsed=elapsed,
            )

            # ------------------------------------------------
            # Convergence announcement
            # ------------------------------------------------

            if reached_convergence:

                print()

                print(
                    f"[Worker {self.worker_id}] "
                    "EARLY STOPPING TRIGGERED"
                )

                print(
                    f"Converged at round: "
                    f"{epoch}"
                )

                print(
                    f"Best validation loss: "
                    f"{self.best_validation_loss:.6f}"
                )

                print(
                    "Participant will remain "
                    "in federated synchronization "
                    "using its latest model."
                )

            # ------------------------------------------------
            # Send update
            # ------------------------------------------------

            print()

            print(
                f"[Worker {self.worker_id}] "
                "Sending model update..."
            )

            self.send_update(
                round_number=round_number,
                loss=loss,
                validation_loss=validation_loss,
                elapsed=elapsed,
                batches=batches,
            )

            print(
                f"[Worker {self.worker_id}] "
                "Update sent."
            )

        # ----------------------------------------------------
        # Finish
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

        else:

            print(
                "Maximum training rounds completed."
            )

        print(
            "=" * 70
        )

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
            "S.H.A ANC synchronized federated worker"
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