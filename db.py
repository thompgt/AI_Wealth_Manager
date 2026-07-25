from datetime import datetime, timedelta, timezone
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from config import settings

# If the URL is for Postgres, replace postgres:// with postgresql:// for SQLAlchemy
db_url = settings.NEON_DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(db_url)
elif "sqlite" in db_url:
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(db_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow() -> datetime:
    """Naive UTC timestamp for the DateTime columns below.

    These columns are timezone-naive and the existing rows are naive UTC, so
    the convention has to stay naive. `datetime.utcnow()` is deprecated in
    Python 3.12+, so we build an aware UTC value and drop the tzinfo rather
    than keep calling the deprecated function in a dozen places.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ClientProfile(Base):
    __tablename__ = "client_profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, nullable=True)
    age = Column(Integer, default=0)
    # "Conservative" | "Moderate" | "Aggressive"
    risk_tolerance = Column(String, default="Moderate")
    time_horizon_years = Column(Integer, default=10)
    # e.g. ["retirement", "home purchase"]
    goals = Column(JSON, default=list)
    net_worth = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Holding(Base):
    __tablename__ = "holdings"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id"), index=True)
    symbol = Column(String, index=True)
    # For the CASH pseudo-asset: quantity holds the dollar amount, cost_basis is 1.0
    quantity = Column(Float, default=0.0)
    cost_basis = Column(Float, default=0.0)
    acquired_at = Column(DateTime, default=utcnow)


class TransactionLog(Base):
    __tablename__ = "transaction_logs"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id"), index=True)
    asset_symbol = Column(String, index=True)
    transaction_type = Column(String)  # "BUY" or "SELL"
    amount = Column(Float)
    price_per_unit = Column(Float)
    timestamp = Column(DateTime, default=utcnow)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id"), index=True)
    run_id = Column(String, index=True)  # groups all nodes of one graph execution
    node_name = Column(String)
    started_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)
    input_snapshot = Column(JSON, default=dict)
    output_snapshot = Column(JSON, default=dict)
    model_used = Column(String, nullable=True)
    status = Column(String, default="success")  # success | error | interrupted
    error_detail = Column(String, nullable=True)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("client_profiles.id"), index=True)
    run_id = Column(String, index=True)
    generated_at = Column(DateTime, default=utcnow)
    report_text = Column(String)
    structured_payload = Column(JSON, default=dict)
    version = Column(Integer, default=1)


class MarketDataCache(Base):
    __tablename__ = "market_data_cache"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    as_of_date = Column(DateTime, index=True)
    close_price = Column(Float)
    fetched_at = Column(DateTime, default=utcnow)


class Approval(Base):
    __tablename__ = "approvals"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True)
    requested_at = Column(DateTime, default=utcnow)
    decided_at = Column(DateTime, nullable=True)
    decided_by = Column(String, nullable=True)
    decision = Column(String, nullable=True)  # approved | rejected
    notes = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized.")


def seed_db():
    db: Session = SessionLocal()
    if db.query(ClientProfile).first():
        print("Database already seeded.")
        db.close()
        return

    # Create a dev/demo client: $500k net worth, moderate risk tolerance,
    # 20-year horizon, saving for retirement.
    client = ClientProfile(
        name="Test User",
        email="test.user@example.com",
        age=45,
        risk_tolerance="Moderate",
        time_horizon_years=20,
        goals=["retirement", "home purchase"],
        net_worth=500000.0,
    )
    db.add(client)
    db.commit()
    db.refresh(client)

    holdings = [
        Holding(client_id=client.id, symbol="CASH", quantity=400000.0, cost_basis=1.0),
        Holding(client_id=client.id, symbol="TSLA", quantity=400.0, cost_basis=250.0,
                acquired_at=utcnow() - timedelta(days=200)),
    ]
    db.add_all(holdings)

    # Trade from 15 days ago for wash-sale testing: sold TSLA at a loss.
    # Rule: if sold at a loss within 30 days, cannot buy back the same asset.
    past_date = utcnow() - timedelta(days=15)
    trade = TransactionLog(
        client_id=client.id,
        asset_symbol="TSLA",
        transaction_type="SELL",
        amount=10000.0,
        price_per_unit=150.0,
        timestamp=past_date,
    )
    db.add(trade)
    db.commit()
    print("Database seeded with mock client profile, holdings, and transaction log.")
    db.close()


if __name__ == "__main__":
    init_db()
    seed_db()
