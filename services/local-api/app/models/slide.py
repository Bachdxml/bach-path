from __future__ import annotations
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, func, Integer, ForeignKey
from app.db.base import Base

class Slide(Base):
    __tablename__ = "slides"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Managed storage fields
    original_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    stored_filename: Mapped[str] = mapped_column(String(255), unique=True)  # e.g., "{id}.svs" or UUID.svs
    stored_path: Mapped[str] = mapped_column(String(1024))                 # absolute path

    file_size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    import_collection_id: Mapped[int | None] = mapped_column(
        ForeignKey("import_collections.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    inference_runs = relationship("InferenceRun", back_populates="slide", cascade="all, delete-orphan")
    import_collection = relationship("ImportCollection", back_populates="slides")
