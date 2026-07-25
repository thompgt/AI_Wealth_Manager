"""Seeds a demo client whose portfolio exercises every guardrail in the system.

`db.seed_db()` creates a minimal dev fixture. This builds a deliberately
flawed, realistic portfolio so a demo run actually has something to find
rather than producing a clean bill of health:

  * ~45% in a single technology position  -> concentration flaw (the Moderate
    tier caps a single position at 25%)
  * ~100% of the equity sleeve in Technology -> sector concentration flaw
  * ~40% sitting in cash                  -> cash drag flaw for a Moderate
    client, and the pool the agents get to deploy
  * three defensive names sold at a loss in the last 30 days -> wash-sale
    exposure, which is what forces the Tax-Awareness agent to actually block
    something the Stock Research agent wants to buy

That last one is the realistic scenario the tax guardrail exists for: a
client harvested losses last month, and the system must not hand those same
names straight back to them.

Share counts are derived from live prices at seed time so the target weights
hold regardless of when the demo is run.

Usage:
    python demo_data.py            # seed (idempotent)
    python demo_data.py --reset    # wipe the demo client and reseed
"""

import argparse
from datetime import timedelta
from typing import Dict

from db import (
    ClientProfile,
    Holding,
    SessionLocal,
    TransactionLog,
    init_db,
    utcnow,
)
from logging_setup import configure_logging, get_logger
from services.market_data import get_current_prices

logger = get_logger(__name__)

DEMO_CLIENT_NAME = "Demo Client (Priya Raman)"

# Target weights for the equity sleeve, as a fraction of total portfolio value.
TARGET_WEIGHTS: Dict[str, float] = {
    "AAPL": 0.45,   # the concentration problem
    "MSFT": 0.15,   # compounds it into a sector problem
}
CASH_WEIGHT = 0.40
TOTAL_PORTFOLIO_VALUE = 750_000.0

# Names the client sold at a loss in the last 30 days -- a plausible
# tax-loss-harvest across their energy and defensive sleeve.
#
# These are chosen to overlap the value/defensive cluster the Stock Research
# screen favours for a cash-heavy, tech-concentrated portfolio, because that
# overlap is the whole point of the demo: the system's own research agent
# wants to buy names the client cannot repurchase yet, and the Tax-Awareness
# agent has to stop it.
#
# Which of them actually gets proposed depends on live valuations on the day,
# so the notebook reports what it found rather than asserting a fixed answer.
RECENTLY_HARVESTED = {
    "XOM": (11, 18400.0, 103.20),  # (days ago, proceeds, price per share)
    "CVX": (18, 14900.0, 141.75),
    "PFE": (9, 3100.0, 25.40),
    "KO": (21, 6800.0, 62.10),
    "VZ": (14, 4200.0, 41.50),
}


def _existing_demo_client(db):
    return db.query(ClientProfile).filter(ClientProfile.name == DEMO_CLIENT_NAME).first()


def reset_demo_client() -> None:
    db = SessionLocal()
    try:
        client = _existing_demo_client(db)
        if client is None:
            return
        db.query(Holding).filter(Holding.client_id == client.id).delete()
        db.query(TransactionLog).filter(TransactionLog.client_id == client.id).delete()
        db.delete(client)
        db.commit()
        logger.info("Removed existing demo client %s", client.id)
    finally:
        db.close()


def seed_demo_client() -> int:
    """Create the demo client if absent. Returns its client_id either way."""
    init_db()

    db = SessionLocal()
    try:
        existing = _existing_demo_client(db)
        if existing is not None:
            logger.info("Demo client already present (id=%s)", existing.id)
            return existing.id

        client = ClientProfile(
            name=DEMO_CLIENT_NAME,
            email="priya.raman@example.com",
            age=45,
            risk_tolerance="Moderate",
            time_horizon_years=20,
            goals=["retirement", "college fund"],
            net_worth=TOTAL_PORTFOLIO_VALUE,
        )
        db.add(client)
        db.commit()
        db.refresh(client)

        prices = get_current_prices(list(TARGET_WEIGHTS))
        holdings = [
            Holding(
                client_id=client.id,
                symbol="CASH",
                quantity=TOTAL_PORTFOLIO_VALUE * CASH_WEIGHT,
                cost_basis=1.0,
            )
        ]
        for symbol, weight in TARGET_WEIGHTS.items():
            price = prices.get(symbol)
            if not price:
                logger.warning(
                    "No live price for %s; skipping it from the demo portfolio so the "
                    "seeded weights stay honest rather than being faked.", symbol,
                )
                continue
            target_value = TOTAL_PORTFOLIO_VALUE * weight
            quantity = round(target_value / price, 4)
            holdings.append(
                Holding(
                    client_id=client.id,
                    symbol=symbol,
                    quantity=quantity,
                    # Bought well below today's price: a large embedded gain,
                    # which makes trimming the concentrated position a taxable
                    # event the report should flag.
                    cost_basis=round(price * 0.55, 2),
                    acquired_at=utcnow() - timedelta(days=900),
                )
            )
        db.add_all(holdings)

        for symbol, (days_ago, proceeds, unit_price) in RECENTLY_HARVESTED.items():
            db.add(
                TransactionLog(
                    client_id=client.id,
                    asset_symbol=symbol,
                    transaction_type="SELL",
                    amount=proceeds,
                    price_per_unit=unit_price,
                    timestamp=utcnow() - timedelta(days=days_ago),
                )
            )

        db.commit()
        logger.info(
            "Seeded demo client id=%s: $%s portfolio, %d holdings, %d recent loss sales.",
            client.id, f"{TOTAL_PORTFOLIO_VALUE:,.0f}", len(holdings), len(RECENTLY_HARVESTED),
        )
        return client.id
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete and recreate the demo client")
    args = parser.parse_args()

    if args.reset:
        reset_demo_client()

    client_id = seed_demo_client()
    print(f"Demo client ready: client_id={client_id}")
