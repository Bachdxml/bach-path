from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, ForeignKey, Float, String, Text

from app.db.base import Base

class Region(Base):
    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(primary_key=True)
    inference_run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id", ondelete="CASCADE"), index=True)

    # Basic geometry. Keep it simple & renderable:
    # store bbox in slide coordinate space (OpenSlide level 0)
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)

    score: Mapped[float] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Optional JSON payload for richer shapes later (polygon, etc.)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    inference_run = relationship("InferenceRun", back_populates="regions")
