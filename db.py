from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from config import settings

# If the URL is for Postgres, replace postgres:// with postgresql:// for SQLAlchemy
db_url = settings.NEON_DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
elif "sqlite" in db_url:
    # Adding connect_args for SQLite
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(db_url)

if "sqlite" not in db_url:
    engine = create_engine(db_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    net_worth = Column(Float, default=0.0)
    # Portfolio breakdown: e.g., {"AAPL": 100000, "CASH": 400000}
    portfolio = Column(JSON, default={})

class TransactionLog(Base):
    __tablename__ = "transaction_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"))
    asset_symbol = Column(String, index=True)
    transaction_type = Column(String)  # "BUY" or "SELL"
    amount = Column(Float)
    price_per_unit = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized.")

def seed_db():
    db: Session = SessionLocal()
    if db.query(UserProfile).first():
        print("Database already seeded.")
        db.close()
        return

    # Create dummy user with $500k net worth
    user = UserProfile(
        name="Test User",
        net_worth=500000.0,
        portfolio={"CASH": 400000.0, "TSLA": 100000.0}
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Add trades from 15 days ago for wash-sale testing (e.g. sold TSLA at a loss)
    # The rule is: if sold at loss within 30 days, cannot buy back. 
    # Let's mock a SELL transaction for TSLA 15 days ago
    past_date = datetime.utcnow() - timedelta(days=15)
    trade = TransactionLog(
        user_id=user.id,
        asset_symbol="TSLA",
        transaction_type="SELL",
        amount=10000.0,
        price_per_unit=150.0,
        timestamp=past_date
    )
    db.add(trade)
    db.commit()
    print("Database seeded with mock user profile and transaction log.")
    db.close()

if __name__ == "__main__":
    init_db()
    seed_db()
