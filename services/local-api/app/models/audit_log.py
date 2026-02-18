from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, func, ForeignKey, Text
from app.db.base import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)

    action: Mapped[str] = mapped_column(String(64), index=True)     # e.g., "slide.import"
    entity_type: Mapped[str] = mapped_column(String(64), index=True) # "slide", "inference_run"
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Clinical-track audit needs context, but don’t bloat:
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    actor = relationship("User", back_populates="audit_logs")
