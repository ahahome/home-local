"""Daily agenda reads and explicit disposition/contact-policy changes."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from app.agent import agenda
from app.store.db import get_conn

router = APIRouter(prefix='/api')


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ActionIn(Input):
    item_id: str = Field(min_length=1, max_length=200)
    fingerprint: str = Field(min_length=1, max_length=128)
    action: Literal['snooze', 'dismiss', 'restore']
    until: StrictInt | None = None


class PolicyIn(Input):
    enabled: StrictBool
    interval_days: StrictInt = Field(ge=1, le=365)
    expected_revision: StrictInt = Field(ge=0)


def _call(fn, *args):
    try:
        return fn(get_conn(), *args)
    except agenda.AgendaError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.get('/agenda')
def daily(include_hidden: bool = False):
    return _call(agenda.get_agenda, include_hidden)


@router.post('/agenda/actions')
def action(body: ActionIn):
    return _call(agenda.apply_action, body.item_id, body.fingerprint, body.action, body.until)


@router.get('/persons/{person_id}/contact-policy')
def contact_policy(person_id: str):
    return _call(agenda.get_contact_policy, person_id)


@router.patch('/persons/{person_id}/contact-policy')
def set_contact_policy(person_id: str, body: PolicyIn):
    return _call(agenda.update_contact_policy, person_id, body.enabled, body.interval_days,
                 body.expected_revision)
