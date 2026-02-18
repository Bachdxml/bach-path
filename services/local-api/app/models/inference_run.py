from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, func, Integer, ForeignKey, Text
from app.db.base import Base
from app.models.enums import InferenceStatus

class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id: Mapped[int] = mapped_column(primary_key=True)

    slide_id: Mapped[int] = mapped_column(ForeignKey("slides.id", ondelete="CASCADE"), index=True)

    requested_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default=InferenceStatus.queued.value)

    started_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Where raw JSON output is stored on disk (keeps DB lean)
    output_json_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # If failed
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    slide = relationship("Slide", back_populates="inference_runs")
    regions = relationship("Region", back_populates="inference_run", cascade="all, delete-orphan")
    requested_by_user = relationship("User", back_populates="inference_runs")
