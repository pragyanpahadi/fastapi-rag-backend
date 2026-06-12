from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    strategy_used = Column(String)
    chunk_count = Column(Integer)
    upload_time = Column(DateTime(timezone=True), server_default=func.now())
