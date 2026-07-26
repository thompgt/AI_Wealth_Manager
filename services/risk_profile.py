"""Risk tolerance questionnaire and scoring.

The risk tier gates every position limit, every beta ceiling and the entire
factor tilt of the screen — it is the single most consequential input in the
system. It used to be a dropdown the client picked themselves, which is both
poor advice and the wrong artifact: a suitability review asks to see the
*basis* for a tier, and "they chose Aggressive" is not one.

The scoring separates two things that a single score conflates:

* **Capacity** — objective ability to absorb loss. Time horizon, income
  stability, emergency reserves, what fraction of net worth is at stake.
  These are facts about the client's situation.
* **Tolerance** — behavioural willingness. What they say they would do in a
  drawdown, their experience, their stated preference.

The assigned tier is the **lower** of the two. A 62-year-old two years from
drawing income who enthusiastically selects "I'd buy more" in a 30% crash has
high tolerance and low capacity, and the honest answer is a conservative
mandate. Averaging the two scores would let stated enthusiasm buy risk the
client cannot afford — which is precisely the failure mode suitability rules
exist to prevent.

Assessments expire. Suitability information goes stale as circumstances
change, and a tier derived from answers given four years ago should not
silently keep authorising risk today.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from db import utcnow

QUESTIONNAIRE_VERSION = "v1"

# Suitability information should be refreshed periodically; two years is a
# common advisory cadence and is short enough that a material life change is
# unlikely to go unrecorded for long.
ASSESSMENT_VALID_DAYS = 730

TIERS = ("Conservative", "Moderate", "Aggressive")


@dataclass(frozen=True)
class Option:
    id: str
    label: str
    points: int


@dataclass(frozen=True)
class Question:
    id: str
    prompt: str
    # "capacity" questions ask what the client's situation can absorb;
    # "tolerance" questions ask what they are willing to sit through.
    dimension: str
    options: List[Option]
    help_text: Optional[str] = None

    def max_points(self) -> int:
        return max(o.points for o in self.options)

    def option(self, option_id: str) -> Optional[Option]:
        return next((o for o in self.options if o.id == option_id), None)


QUESTIONS: List[Question] = [
    Question(
        id="horizon",
        dimension="capacity",
        prompt="When do you expect to start drawing meaningfully on this money?",
        options=[
            Option("a", "Within 2 years", 0),
            Option("b", "2 to 5 years", 1),
            Option("c", "5 to 10 years", 3),
            Option("d", "More than 10 years", 4),
        ],
        help_text="The dominant input. A short horizon caps risk regardless of preference: "
                  "there is no time to recover from a drawdown before the money is needed.",
    ),
    Question(
        id="withdrawal_rate",
        dimension="capacity",
        prompt="Roughly what share of this portfolio will you need to withdraw each year?",
        options=[
            Option("a", "More than 8%", 0),
            Option("b", "4% to 8%", 1),
            Option("c", "1% to 4%", 3),
            Option("d", "Nothing planned", 4),
        ],
    ),
    Question(
        id="income_stability",
        dimension="capacity",
        prompt="How stable is your income outside this portfolio?",
        options=[
            Option("a", "No income outside the portfolio", 0),
            Option("b", "Variable or commission-based", 1),
            Option("c", "Stable employment", 3),
            Option("d", "Very secure, or multiple sources", 4),
        ],
    ),
    Question(
        id="emergency_reserve",
        dimension="capacity",
        prompt="If you had an unexpected large expense, could you cover it without selling investments?",
        options=[
            Option("a", "No, I would have to sell", 0),
            Option("b", "Partly", 2),
            Option("c", "Yes, I hold 6+ months of expenses in cash", 4),
        ],
        help_text="Without a reserve, a market drawdown and a personal cash need arrive together "
                  "and force selling at the worst possible time.",
    ),
    Question(
        id="share_of_wealth",
        dimension="capacity",
        prompt="What share of your total net worth does this portfolio represent?",
        options=[
            Option("a", "Almost all of it", 0),
            Option("b", "More than half", 1),
            Option("c", "A quarter to a half", 3),
            Option("d", "Less than a quarter", 4),
        ],
    ),
    Question(
        id="drawdown_reaction",
        dimension="tolerance",
        prompt="Your portfolio falls 25% over six months. What do you actually do?",
        options=[
            Option("a", "Sell everything and move to cash", 0),
            Option("b", "Sell some to reduce the risk", 1),
            Option("c", "Hold and wait for recovery", 3),
            Option("d", "Buy more at lower prices", 4),
        ],
        help_text="The single best behavioural predictor. Selling into a decline converts a "
                  "temporary loss into a permanent one.",
    ),
    Question(
        id="loss_threshold",
        dimension="tolerance",
        prompt="What is the largest one-year loss you could accept without changing plan?",
        options=[
            Option("a", "Any loss would concern me", 0),
            Option("b", "Up to 10%", 1),
            Option("c", "Up to 25%", 3),
            Option("d", "35% or more", 4),
        ],
    ),
    Question(
        id="experience",
        dimension="tolerance",
        prompt="How would you describe your investing experience?",
        options=[
            Option("a", "This is my first investment portfolio", 0),
            Option("b", "Some experience with funds", 2),
            Option("c", "Experienced across asset classes, through a bear market", 4),
        ],
    ),
    Question(
        id="objective",
        dimension="tolerance",
        prompt="Which statement best describes your objective?",
        options=[
            Option("a", "Preserve capital above all", 0),
            Option("b", "Steady income with modest growth", 1),
            Option("c", "Balanced growth and stability", 3),
            Option("d", "Maximum long-term growth, accepting volatility", 4),
        ],
    ),
]

QUESTIONS_BY_ID = {q.id: q for q in QUESTIONS}


@dataclass
class RiskAssessmentResult:
    tier: str
    raw_score: float
    capacity_score: float   # 0..1
    tolerance_score: float  # 0..1
    capacity_tier: str
    tolerance_tier: str
    answered: int
    total_questions: int
    # Populated when capacity and tolerance disagree. This is the sentence an
    # advisor needs in front of them, and the one the report has to carry.
    constraint_note: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "raw_score": round(self.raw_score, 3),
            "capacity_score": round(self.capacity_score, 3),
            "tolerance_score": round(self.tolerance_score, 3),
            "capacity_tier": self.capacity_tier,
            "tolerance_tier": self.tolerance_tier,
            "answered": self.answered,
            "total_questions": self.total_questions,
            "constraint_note": self.constraint_note,
            "warnings": self.warnings,
            "questionnaire_version": QUESTIONNAIRE_VERSION,
        }


def _tier_from_score(score: float) -> str:
    """Map a 0..1 dimension score to a tier.

    Thresholds are deliberately asymmetric. Reaching Aggressive requires
    scoring above 70%, while falling to Conservative only requires being below
    40% — the cost of wrongly assigning too much risk is a client who sells at
    the bottom, and the cost of assigning too little is some forgone return.
    Those are not symmetric harms.
    """
    if score < 0.40:
        return "Conservative"
    if score < 0.70:
        return "Moderate"
    return "Aggressive"


def score_answers(answers: Dict[str, str]) -> RiskAssessmentResult:
    """Score a completed questionnaire.

    Unanswered questions are excluded from the denominator rather than scored
    as zero, so a partial submission does not silently read as maximally
    conservative — but the result carries a warning, and a mostly-empty
    questionnaire is not a basis for a mandate.
    """
    dimensions: Dict[str, List[Tuple[int, int]]] = {"capacity": [], "tolerance": []}
    warnings: List[str] = []
    answered = 0

    for question in QUESTIONS:
        chosen_id = answers.get(question.id)
        if not chosen_id:
            continue
        option = question.option(str(chosen_id))
        if option is None:
            warnings.append(f"Ignored unrecognised answer {chosen_id!r} for {question.id!r}.")
            continue
        dimensions[question.dimension].append((option.points, question.max_points()))
        answered += 1

    def normalize(pairs: List[Tuple[int, int]]) -> float:
        if not pairs:
            return 0.0
        earned = sum(p for p, _ in pairs)
        possible = sum(m for _, m in pairs)
        return earned / possible if possible else 0.0

    capacity = normalize(dimensions["capacity"])
    tolerance = normalize(dimensions["tolerance"])

    if not dimensions["capacity"]:
        warnings.append(
            "No capacity questions were answered; capacity is unknown and has been "
            "treated as the lowest tier."
        )
    if not dimensions["tolerance"]:
        warnings.append(
            "No tolerance questions were answered; tolerance is unknown and has been "
            "treated as the lowest tier."
        )
    if answered < len(QUESTIONS):
        warnings.append(
            f"{len(QUESTIONS) - answered} of {len(QUESTIONS)} questions were left "
            "unanswered; the resulting tier is provisional."
        )

    capacity_tier = _tier_from_score(capacity)
    tolerance_tier = _tier_from_score(tolerance)

    # The lower of the two, always. See the module docstring: averaging lets
    # stated enthusiasm buy risk the client's circumstances cannot absorb.
    tier = TIERS[min(TIERS.index(capacity_tier), TIERS.index(tolerance_tier))]

    note = None
    if capacity_tier != tolerance_tier:
        if TIERS.index(capacity_tier) < TIERS.index(tolerance_tier):
            note = (
                f"Stated risk tolerance is {tolerance_tier} but financial capacity supports "
                f"only {capacity_tier}. The mandate is set to {tier}: horizon, withdrawal "
                "needs or reserves constrain how much loss this portfolio can absorb, "
                "regardless of comfort with volatility."
            )
        else:
            note = (
                f"Financial capacity would support {capacity_tier} but stated tolerance is "
                f"{tolerance_tier}. The mandate is set to {tier}, because a portfolio the "
                "client abandons in a drawdown does not deliver its expected return."
            )

    return RiskAssessmentResult(
        tier=tier,
        raw_score=round((capacity + tolerance) / 2, 4),
        capacity_score=round(capacity, 4),
        tolerance_score=round(tolerance, 4),
        capacity_tier=capacity_tier,
        tolerance_tier=tolerance_tier,
        answered=answered,
        total_questions=len(QUESTIONS),
        constraint_note=note,
        warnings=warnings,
    )


def apply_hard_overrides(
    result: RiskAssessmentResult,
    *,
    age: Optional[int] = None,
    time_horizon_years: Optional[int] = None,
) -> RiskAssessmentResult:
    """Cap the tier on facts the questionnaire may not have captured.

    A profile whose stored horizon is three years cannot carry an Aggressive
    mandate no matter what the answers said. This exists because the profile
    and the questionnaire are edited independently, and the more conservative
    of two contradictory statements is the safe one to honour.
    """
    warnings = list(result.warnings)
    tier = result.tier
    note = result.constraint_note

    if time_horizon_years is not None and time_horizon_years <= 3 and tier != "Conservative":
        warnings.append(
            f"Tier reduced from {tier} to Conservative: the stated time horizon of "
            f"{time_horizon_years} years leaves no time to recover from a drawdown."
        )
        tier = "Conservative"
        note = note or warnings[-1]
    elif time_horizon_years is not None and time_horizon_years <= 7 and tier == "Aggressive":
        warnings.append(
            f"Tier reduced from Aggressive to Moderate: a {time_horizon_years}-year horizon "
            "is short for an aggressive mandate."
        )
        tier = "Moderate"
        note = note or warnings[-1]

    if age is not None and age >= 75 and tier == "Aggressive":
        warnings.append(
            "Tier reduced from Aggressive to Moderate on age. This is a floor, not a "
            "judgement: it can be overridden by an explicit policy exception."
        )
        tier = "Moderate"
        note = note or warnings[-1]

    return RiskAssessmentResult(
        tier=tier,
        raw_score=result.raw_score,
        capacity_score=result.capacity_score,
        tolerance_score=result.tolerance_score,
        capacity_tier=result.capacity_tier,
        tolerance_tier=result.tolerance_tier,
        answered=result.answered,
        total_questions=result.total_questions,
        constraint_note=note,
        warnings=warnings,
    )


def expiry_from(taken_at=None):
    return (taken_at or utcnow()) + timedelta(days=ASSESSMENT_VALID_DAYS)


def questionnaire_schema() -> Dict[str, Any]:
    """Serializable form for the API and the dashboard."""
    return {
        "version": QUESTIONNAIRE_VERSION,
        "valid_days": ASSESSMENT_VALID_DAYS,
        "questions": [
            {
                "id": q.id,
                "prompt": q.prompt,
                "dimension": q.dimension,
                "help_text": q.help_text,
                "options": [{"id": o.id, "label": o.label} for o in q.options],
            }
            for q in QUESTIONS
        ],
    }
