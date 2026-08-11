"""Seeds a demo client whose portfolio exercises every guardrail in the system.

A fresh database is empty and a run against it has nothing to say. This builds
a deliberately flawed, realistic portfolio so a demo run actually has something
to find rather than producing a clean bill of health:

  * ~45% in a single technology position  -> concentration flaw (the Moderate
    tier caps a single position at 25%), bought ~900 days ago at roughly half
    today's price, so trimming it realizes a large *long-term* capital gain
    and the report has to weigh the tax against the concentration
  * a second technology name compounding it into a sector concentration flaw:
    the entire equity sleeve is Technology, against a 35% sector limit
  * ~40% sitting in cash                  -> cash drag flaw for a Moderate
    client, and the pool the agents get to deploy
  * five defensive/energy names sold at a loss in the last 30 days, in the
    *taxable* account -> wash-sale exposure, which is what forces the
    Tax-Awareness agent to actually block something the research agent wants
    to buy

That last one is the realistic scenario the tax guardrail exists for: a client
harvested losses last month, and the system must not hand those same names
straight back to them. It is seeded the way a real one arises -- opening tax
lots, then an actual sale below basis recorded through
`services.tax_lots.record_sale`, which is what writes the `disposed_at` /
`realized_gain < 0` rows that `check_wash_sale` looks for. Fabricating the
flags any other way would test the seed rather than the guardrail.

The client holds two accounts, because the tax wrapper is what makes the tax
module mean anything: a taxable brokerage account (where the loss sales
happened, and where the wash-sale rule bites) and a Roth IRA (where
repurchasing one of those names would disallow the loss *permanently*).

Share counts are derived from prices at seed time so the target weights hold
regardless of when the demo is run.

Usage:
    python demo_data.py            # seed (idempotent)
    python demo_data.py --reset    # wipe the demo client and reseed
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from db import (
    Account,
    CashTransaction,
    ClientProfile,
    InvestmentPolicy,
    Organization,
    Position,
    SessionLocal,
    TaxLot,
    User,
    init_db,
    utcnow,
)
from logging_setup import configure_logging, get_logger
from security import bootstrap_admin
from services.market_data import get_current_prices
from services.policy import activate_policy, draft_policy
from services.tax_lots import record_sale, seed_lots_from_positions
from services.universe import seed_universe

logger = get_logger(__name__)

DEMO_CLIENT_NAME = "Demo Client (Priya Raman)"

TAXABLE_ACCOUNT_NAME = "Individual Brokerage"
RETIREMENT_ACCOUNT_NAME = "Roth IRA"

TOTAL_PORTFOLIO_VALUE = 750_000.0

# Target weights for the equity sleeve, as a fraction of *total* portfolio
# value. Both names are Technology, so the equity sleeve is 100% one sector.
TARGET_WEIGHTS: Dict[str, float] = {
    "AAPL": 0.45,   # the concentration problem
    "MSFT": 0.15,   # compounds it into a sector problem
}

# Cash, split across the two wrappers. 40% in total: the cash-drag flaw, and
# the pool the agents get to deploy.
TAXABLE_CASH_WEIGHT = 0.30
RETIREMENT_CASH_WEIGHT = 0.10
CASH_WEIGHT = TAXABLE_CASH_WEIGHT + RETIREMENT_CASH_WEIGHT

# How long the concentrated positions have been held. Past 365 days, so a trim
# is a long-term gain rather than a short-term one -- which is the whole reason
# the report has to reason about lot terms instead of a single average cost.
EQUITY_HOLDING_DAYS = 900
# Bought at 55% of today's price: a large embedded gain.
EQUITY_COST_RATIO = 0.55


@dataclass(frozen=True)
class HarvestedSale:
    """A position the client sold below cost inside the wash-sale window.

    Quantity and both prices are given explicitly rather than derived from a
    live quote: the *loss* is the point, and a loss computed against whatever
    the market happens to be doing on the day of the demo would sometimes be a
    gain, silently disarming the guardrail this data exists to trigger.
    """

    days_ago: int
    quantity: float
    cost_per_share: float   # what the client paid
    sale_price: float       # what they sold at -- always below cost

    @property
    def realized_loss(self) -> float:
        return (self.sale_price - self.cost_per_share) * self.quantity


# Names the client sold at a loss in the last 30 days -- a plausible
# tax-loss-harvest across their energy and defensive sleeve.
#
# These are chosen to overlap the value/defensive cluster the Stock Research
# screen favours for a cash-heavy, tech-concentrated portfolio, because that
# overlap is the whole point of the demo: the system's own research agent wants
# to buy names the client cannot repurchase yet, and the Tax-Awareness agent
# has to stop it.
#
# Which of them actually gets proposed depends on the screen on the day, so the
# notebook reports what it found rather than asserting a fixed answer.
RECENTLY_HARVESTED: Dict[str, HarvestedSale] = {
    "XOM": HarvestedSale(days_ago=11, quantity=180, cost_per_share=118.00, sale_price=103.20),
    "CVX": HarvestedSale(days_ago=18, quantity=105, cost_per_share=162.00, sale_price=141.75),
    "PFE": HarvestedSale(days_ago=9, quantity=900, cost_per_share=29.10, sale_price=25.40),
    "KO": HarvestedSale(days_ago=21, quantity=110, cost_per_share=68.50, sale_price=62.10),
    "VZ": HarvestedSale(days_ago=14, quantity=100, cost_per_share=46.80, sale_price=41.50),
}

# How long each harvested name was held before being sold. Under a year, so the
# realized losses are short-term -- the more valuable kind to harvest, and the
# reason a client would have done this.
HARVEST_HOLDING_DAYS = 200

# On file so the tax estimates are the client's real cost rather than the
# top-bracket assumption `services/tax_lots.py` falls back to.
MARGINAL_TAX_RATE = 0.32
CAPITAL_GAINS_TAX_RATE = 0.15

PriceLookup = Callable[[Sequence[str]], Mapping[str, float]]


# --- Lookup helpers ----------------------------------------------------------


def _existing_demo_client(db) -> Optional[ClientProfile]:
    return db.query(ClientProfile).filter(ClientProfile.name == DEMO_CLIENT_NAME).first()


def _demo_accounts(db, client_id: int) -> List[Account]:
    return db.query(Account).filter(Account.client_id == client_id).all()


# --- Reset -------------------------------------------------------------------


def reset_demo_client() -> None:
    """Remove the demo client and everything hanging off it.

    The child rows are deleted explicitly rather than left to `ON DELETE
    CASCADE`. The cascades are declared, but they only fire when the backend
    enforces foreign keys, and this has to work identically against SQLite
    (where they are off unless a pragma turns them on) and Postgres. A half
    deleted demo client is worse than none: the next seed would find no
    `client_profiles` row, create a fresh one, and leave the old orphaned lots
    behind to be counted by nothing.
    """
    init_db()
    db = SessionLocal()
    try:
        client = _existing_demo_client(db)
        if client is None:
            return
        client_id = client.id
        account_ids = [a.id for a in _demo_accounts(db, client_id)]
        if account_ids:
            for model in (TaxLot, Position, CashTransaction):
                db.query(model).filter(model.account_id.in_(account_ids)).delete(
                    synchronize_session=False
                )
            db.query(Account).filter(Account.id.in_(account_ids)).delete(
                synchronize_session=False
            )
        db.query(InvestmentPolicy).filter(InvestmentPolicy.client_id == client_id).delete(
            synchronize_session=False
        )
        db.query(ClientProfile).filter(ClientProfile.id == client_id).delete(
            synchronize_session=False
        )
        db.commit()
        logger.info("Removed existing demo client %s and its %d account(s)",
                    client_id, len(account_ids))
    finally:
        db.close()


# --- Seed --------------------------------------------------------------------


def _tenancy(db) -> tuple:
    """The org and advisor the demo client belongs to, creating them if needed.

    `bootstrap_admin` only acts on a database with no users at all, so on an
    existing deployment this attaches the demo client to whatever org is
    already there rather than inventing a second tenant.
    """
    bootstrap_admin(db)
    org = db.query(Organization).order_by(Organization.id).first()
    if org is None:
        raise RuntimeError(
            "No organization exists and bootstrapping did not create one. Check "
            "BOOTSTRAP_ORG_NAME / BOOTSTRAP_ADMIN_EMAIL in the environment."
        )
    advisor = (
        db.query(User)
        .filter(User.org_id == org.id)
        .order_by(User.id)
        .first()
    )
    return org, advisor


def _seed_equity_sleeve(db, account: Account, prices: Mapping[str, float]) -> List[str]:
    """The concentrated, deeply appreciated technology positions."""
    acquired = utcnow() - timedelta(days=EQUITY_HOLDING_DAYS)

    holdings = []
    for symbol, weight in TARGET_WEIGHTS.items():
        price = prices.get(symbol)
        if not price or price <= 0:
            logger.warning(
                "No price for %s; skipping it from the demo portfolio so the seeded "
                "weights stay honest rather than being faked.", symbol,
            )
            continue
        quantity = round(TOTAL_PORTFOLIO_VALUE * weight / price, 4)
        holdings.append((symbol, quantity, round(price * EQUITY_COST_RATIO, 2), acquired))

    if holdings:
        seed_lots_from_positions(db, account, holdings)
    return [h[0] for h in holdings]


def _seed_harvested_losses(db, account: Account) -> float:
    """Open each harvested lot and then actually sell it, below cost.

    Going through `record_sale` rather than writing `tax_lots` rows by hand is
    deliberate: it is the same code path a real disposal takes, so the rows it
    leaves behind are exactly the ones `check_wash_sale` queries -- disposed
    inside the window, in a taxable account, with a negative realized gain. A
    seed that hand-wrote those columns would prove only that the seed and the
    check agree on a schema.
    """
    total_loss = 0.0
    for symbol, sale in RECENTLY_HARVESTED.items():
        sold_at = utcnow() - timedelta(days=sale.days_ago)
        acquired_at = sold_at - timedelta(days=HARVEST_HOLDING_DAYS)
        seed_lots_from_positions(
            db, account, [(symbol, sale.quantity, sale.cost_per_share, acquired_at)]
        )
        realized = record_sale(
            db,
            account,
            symbol,
            sale.quantity,
            sale.sale_price,
            method="FIFO",
            executed_at=sold_at,
        )
        if realized.realized_gain >= 0:
            raise ValueError(
                f"{symbol} was seeded as a harvested loss but realized "
                f"${realized.realized_gain:,.2f}. The wash-sale guardrail only fires on "
                f"losses, so this row would silently disarm the demo."
            )
        total_loss += realized.realized_gain
    return total_loss


def _seed_policy(db, client: ClientProfile, advisor: Optional[User]) -> InvestmentPolicy:
    """An approved IPS, so the run cites agreed limits rather than defaults.

    Seeded from the Moderate tier defaults, which are what make the portfolio
    above flawed: a 25% position cap against a 45% position, a 35% sector cap
    against a 100% Technology sleeve, and a 20% cash ceiling against 40% cash.
    """
    policy = draft_policy(
        db,
        client,
        created_by_user_id=advisor.id if advisor else None,
        overrides={
            "marginal_tax_rate": MARGINAL_TAX_RATE,
            "capital_gains_tax_rate": CAPITAL_GAINS_TAX_RATE,
            "notes": "Demo mandate: standard Moderate limits, approved at onboarding.",
        },
    )
    if advisor is not None:
        activate_policy(db, policy, approved_by_user_id=advisor.id)
    else:  # pragma: no cover -- bootstrap always leaves an admin behind
        policy.status = "active"
        policy.effective_from = utcnow()
    return policy


def seed_demo_client(price_lookup: Optional[PriceLookup] = None) -> int:
    """Create the demo client if absent. Returns its client_id either way.

    `price_lookup` defaults to live quotes so the seeded weights reflect real
    prices. It is injectable so callers that must not touch the network -- the
    test suite -- can supply their own without the seed knowing or caring.
    """
    init_db()
    price_lookup = price_lookup or get_current_prices

    db = SessionLocal()
    try:
        existing = _existing_demo_client(db)
        if existing is not None:
            logger.info("Demo client already present (id=%s)", existing.id)
            return existing.id

        # Priced before anything is written. The default lookup goes to the
        # network, and holding an open write transaction across a slow call
        # locks the database against the quote cache's own writes -- and if the
        # call fails, nothing has been half-created.
        prices = dict(price_lookup(list(TARGET_WEIGHTS)))

        org, advisor = _tenancy(db)
        # The research agents can only recommend what is in the securities
        # master, and the portfolio's sector exposure is read from it too, so
        # an unseeded catalogue makes the demo's whole equity sleeve
        # "Unclassified".
        added = seed_universe(db)
        if added:
            logger.info("Seeded %d securities into the investable universe.", added)

        client = ClientProfile(
            org_id=org.id,
            advisor_id=advisor.id if advisor else None,
            name=DEMO_CLIENT_NAME,
            email="priya.raman@example.com",
            age=45,
            risk_tolerance="Moderate",
            time_horizon_years=20,
            goals=["retirement", "college fund"],
            net_worth=TOTAL_PORTFOLIO_VALUE,
            notes=(
                "Concentrated in employer-adjacent technology holdings from a previous "
                "role. Harvested losses across the defensive sleeve last month."
            ),
            status="active",
            kyc_status="verified",
            kyc_verified_at=utcnow(),
            onboarded_at=utcnow() - timedelta(days=EQUITY_HOLDING_DAYS),
        )
        db.add(client)
        db.flush()

        taxable = Account(
            org_id=org.id,
            client_id=client.id,
            name=TAXABLE_ACCOUNT_NAME,
            account_type="individual",
            tax_treatment="taxable",
            custodian="Demo Custodian",
            account_number_masked="****4417",
            cash_balance=round(TOTAL_PORTFOLIO_VALUE * TAXABLE_CASH_WEIGHT, 2),
        )
        retirement = Account(
            org_id=org.id,
            client_id=client.id,
            name=RETIREMENT_ACCOUNT_NAME,
            account_type="roth_ira",
            # Repurchasing a harvested name *here* disallows the loss
            # permanently, which is the most expensive version of the mistake
            # and the reason the accounts are split at all.
            tax_treatment="tax_exempt",
            custodian="Demo Custodian",
            account_number_masked="****9032",
            cash_balance=round(TOTAL_PORTFOLIO_VALUE * RETIREMENT_CASH_WEIGHT, 2),
        )
        db.add_all([taxable, retirement])
        db.flush()

        seeded_symbols = _seed_equity_sleeve(db, taxable, prices)
        realized_loss = _seed_harvested_losses(db, taxable)

        for account in (taxable, retirement):
            db.add(
                CashTransaction(
                    org_id=org.id,
                    account_id=account.id,
                    transaction_type="DEPOSIT",
                    amount=account.cash_balance,
                    description="Opening funding for the demo portfolio.",
                    occurred_at=utcnow() - timedelta(days=EQUITY_HOLDING_DAYS),
                )
            )

        _seed_policy(db, client, advisor)

        db.commit()
        logger.info(
            "Seeded demo client id=%s: $%s portfolio across %d accounts, %d equity "
            "position(s) (%s), %d loss sale(s) in the last 30 days realizing $%s.",
            client.id,
            f"{TOTAL_PORTFOLIO_VALUE:,.0f}",
            2,
            len(seeded_symbols),
            ", ".join(seeded_symbols) or "none",
            len(RECENTLY_HARVESTED),
            f"{realized_loss:,.0f}",
        )
        return client.id
    except Exception:
        db.rollback()
        raise
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
