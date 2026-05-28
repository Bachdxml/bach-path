from __future__ import annotations

from passlib.context import CryptContext


password_context = CryptContext(schemes=["argon2"], deprecated="auto")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_context.verify(password, password_hash)
    except Exception:
        return False


def hash_password(password: str) -> str:
    return password_context.hash(password)
