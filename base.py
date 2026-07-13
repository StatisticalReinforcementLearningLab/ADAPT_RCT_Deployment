from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StoredValue:
    user_id: str
    type: str
    value: Any


@dataclass
class StateResult:
    state: dict
    stored_values: list[StoredValue] | None = None
    transformed_observations: list[dict] | None = None


@dataclass
class ActionResult:
    action: Any
    action_prob: float
    random_state: dict = field(default_factory=dict)


@dataclass
class ActionRecord:
    action_id: int
    user_id: str
    action_type: str
    action_idx: str
    action: Any
    action_prob: float
    context: dict
    state: dict
    retrieved_data: dict | None
    stored_values: list[StoredValue] | None
    transformed_observations: list[dict] | None
    model_params_id: int
    request_timestamp: datetime
    server_timestamp: datetime


@dataclass
class ObservationRecord:
    observation_id: int
    user_id: str
    observation_type: str
    payload: dict
    client_timestamp: datetime
    server_timestamp: datetime


@dataclass
class UpdatePrepResult:
    rewards: dict[int, float]
    stored_values: list[StoredValue] | None = None
    transformed_observations: list[dict] | None = None


@dataclass
class UpdateResult:
    model_params: dict
    stored_values: list[StoredValue] | None = None


@dataclass
class UpdateRecord:
    """Created and persisted by the framework. Not passed to handler methods."""

    update_id: str
    user_id: str
    action_type: str
    status: str
    request_timestamp: datetime
    created_at: datetime
    prep_retrieved_data: dict | None
    update_retrieved_data: dict | None
    rewards: dict[int, float] | None
    prep_stored_values: list[StoredValue] | None
    update_stored_values: list[StoredValue] | None
    transformed_observations: list[dict] | None
    completed_at: datetime | None = None
    error_message: str | None = None


class DBReader:
    def get_actions(
        self,
        action_type: str | None = None,
        limit: int | None = None,
        before: datetime | None = None,
    ) -> list[ActionRecord]:
        ...

    def get_observations(
        self,
        observation_type: str | None = None,
        since: datetime | None = None,
        before: datetime | None = None,
        limit: int | None = None,
    ) -> list[ObservationRecord]:
        ...

    def get_stored_value(self, type: str) -> StoredValue | None:
        ...

    def get_retrieved_data(self) -> dict:
        ...


class ActionHandlerBase(ABC):
    def __init__(self, config: dict = None):
        pass

    @abstractmethod
    def get_initial_model_params(self, user_id: str) -> dict:
        pass

    @abstractmethod
    def make_state(
        self,
        user_id: str,
        context: dict,
        db_reader: DBReader,
    ) -> StateResult:
        pass

    @abstractmethod
    def get_action(
        self,
        user_id: str,
        state: dict,
        model_params: dict,
    ) -> ActionResult:
        pass

    @abstractmethod
    def prepare_for_update(
        self,
        user_id: str,
        db_reader: DBReader,
    ) -> UpdatePrepResult:
        pass

    @abstractmethod
    def update(
        self,
        user_id: str,
        model_params: dict,
        actions: list[ActionRecord],
        rewards: dict[int, float],
        db_reader: DBReader,
    ) -> UpdateResult:
        pass
