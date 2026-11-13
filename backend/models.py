from sqlalchemy import Column, Integer, String, Text, Date, DateTime, Numeric, ForeignKey
from sqlalchemy.sql import func

from .database import Base


class Disease(Base):
    __tablename__ = "diseases"

    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String(100), nullable=False)
    disease_name = Column(String(200), nullable=False)
    model_class = Column(String(200), unique=True, nullable=False)
    scientific_name = Column(String(200))
    description = Column(Text)
    symptoms = Column(Text)
    prevention = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class ManagementGuideline(Base):
    __tablename__ = "management_guidelines"

    id = Column(Integer, primary_key=True, index=True)
    disease_id = Column(
        Integer,
        ForeignKey("diseases.id", ondelete="CASCADE"),
        nullable=False
    )
    management_type = Column(String(50), nullable=False)
    guideline = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    last_verified = Column(Date)


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    disease_id = Column(
        Integer,
        ForeignKey("diseases.id", ondelete="SET NULL"),
        nullable=True
    )
    confidence = Column(Numeric(5, 4), nullable=False)
    status = Column(String(20), nullable=False)
    image_path = Column(Text)
    predicted_at = Column(
        DateTime,
        server_default=func.now()
    )