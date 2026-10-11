"""56 synthetic multi-session scenarios with planted facts and rubric grading.

40 short scenarios plus 16 long-history scenarios (30+ sessions, ~9k tokens
of history) where facts planted early must survive chatter, corrections, and
distractors. Long histories are deterministic (fixed rotation, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Turn = tuple[str, str]  # (role, text)


@dataclass(frozen=True)
class Question:
    """One later-session question: rubric keywords, stale phrases, retrieval gold."""

    q: str
    keywords: tuple[str, ...]
    must_absent: tuple[str, ...] = ()
    gold: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """Multi-session conversation plus questions requiring recall or update."""

    id: str
    category: str
    sessions: tuple[tuple[Turn, ...], ...]
    questions: tuple[Question, ...] = field(default_factory=tuple)
    skip_learn: tuple[int, ...] = ()
    """Session indices excluded from memory learning (pure chatter).

    Documented rationale: extraction on pure chatter costs a model call and
    yields nothing, so the harness skips those sessions. All sessions still
    count toward history size (what full_history stuffing must fit).
    """


def _u(text: str) -> Turn:
    return ("user", text)


def _a(text: str) -> Turn:
    return ("assistant", text)


def _chatter(n: int) -> tuple[Turn, ...]:
    fillers = [
        ("user", "By the way the weather has been so strange lately."),
        ("assistant", "Tell me about it. Anything else on your mind?"),
        ("user", "I finally finished that book I was reading for weeks."),
        ("assistant", "Nice. What did you think of the ending?"),
        ("user", "Work has been busy but otherwise things are fine."),
        ("assistant", "Got it. Let me know if I can help with anything."),
    ]
    return tuple(fillers[:n])


SCENARIOS: tuple[Scenario, ...] = (
    # -- simple recall (8) ---------------------------------------------------------
    Scenario(
        "recall-01",
        "recall",
        (
            (_u("I live in Lisbon now, moved here in March."), _a("Lisbon in March, noted.")),
            (_u("Work is going well."), _a("Glad to hear it.")),
        ),
        (Question("Where do I live?", ("lisbon",), gold=("lisbon",)),),
    ),
    Scenario(
        "recall-02",
        "recall",
        (
            (_u("My dog is a golden retriever named Biscuit."), _a("Biscuit is a great name.")),
            (_u("I take him to the park every evening."), _a("He must love that.")),
        ),
        (
            Question(
                "What is my dog's name and breed?",
                ("biscuit", "golden retriever"),
                gold=("biscuit",),
            ),
        ),
    ),
    Scenario(
        "recall-03",
        "recall",
        (
            (_u("I drink oat milk latte every morning."), _a("Good choice.")),
            (_u("I am trying to cut down on sugar too."), _a("Worth trying.")),
        ),
        (Question("What do I drink in the morning?", ("oat", "latte"), gold=("latte",)),),
    ),
    Scenario(
        "recall-04",
        "recall",
        (
            (_u("I am a backend engineer and I work with Postgres daily."), _a("Solid stack.")),
            (_u("Lately I have been learning query optimization."), _a("Useful skill.")),
        ),
        (
            Question(
                "What is my job and main database?", ("backend", "postgres"), gold=("postgres",)
            ),
        ),
    ),
    Scenario(
        "recall-05",
        "recall",
        (
            (_u("My sister Ana is visiting me next month."), _a("Have fun with Ana.")),
            (_u("We plan to go hiking in Sintra."), _a("Sintra is beautiful.")),
        ),
        (Question("Who is visiting me next month?", ("ana",), gold=("ana",)),),
    ),
    Scenario(
        "recall-06",
        "recall",
        (
            (_u("I play chess on weekends, rated around 1400."), _a("Not bad at all.")),
            (_u("I prefer the Italian game as white."), _a("A classic.")),
        ),
        (
            Question(
                "What game do I play and what is my rating?", ("chess", "1400"), gold=("chess",)
            ),
        ),
    ),
    Scenario(
        "recall-07",
        "recall",
        (
            (_u("I am allergic to peanuts, it is serious."), _a("I will remember that.")),
            (_u("I always carry an epi-pen."), _a("Stay safe.")),
        ),
        (Question("What am I allergic to?", ("peanut",), gold=("peanut",)),),
    ),
    Scenario(
        "recall-08",
        "recall",
        (
            (_u("My favorite author is Ursula Le Guin."), _a("Excellent taste.")),
            (_u("I reread A Wizard of Earthsea every year."), _a("A ritual worth keeping.")),
        ),
        (Question("Who is my favorite author?", ("le guin",), gold=("le guin",)),),
    ),
    # -- update / contradiction (8): latest fact must win ---------------------------
    Scenario(
        "update-01",
        "update",
        (
            (_u("I live in Berlin."), _a("Berlin, got it.")),
            (_u("Update: I moved from Berlin to Lisbon last month."), _a("Lisbon now, updated.")),
        ),
        (Question("Where do I live?", ("lisbon",), ("berlin",), ("lisbon",)),),
    ),
    Scenario(
        "update-02",
        "update",
        (
            (_u("I drink regular cow milk in my coffee."), _a("Noted.")),
            (_u("I switched to oat milk, cow milk upsets my stomach now."), _a("Oat milk it is.")),
        ),
        (Question("What milk do I drink?", ("oat",), ("cow milk",), ("oat",)),),
    ),
    Scenario(
        "update-03",
        "update",
        (
            (_u("I work as a frontend developer."), _a("Cool.")),
            (_u("I changed roles: I am a backend engineer now."), _a("Congrats on the switch.")),
        ),
        (Question("What is my current role?", ("backend",), ("frontend",), ("backend",)),),
    ),
    Scenario(
        "update-04",
        "update",
        (
            (_u("My car is a red Honda Civic."), _a("Nice car.")),
            (_u("I sold the Honda and bought a blue Tesla Model 3."), _a("Big upgrade.")),
        ),
        (Question("What car do I drive now?", ("tesla",), ("honda",), ("tesla",)),),
    ),
    Scenario(
        "update-05",
        "update",
        (
            (_u("I run every morning before work."), _a("Impressive discipline.")),
            (_u("My knee hurts so I swim now instead of running."), _a("Swimming is gentler.")),
        ),
        (Question("What exercise do I do now?", ("swim",), (), ("swim",)),),
    ),
    Scenario(
        "update-06",
        "update",
        (
            (_u("I use VS Code for everything."), _a("Popular choice.")),
            (_u("I switched to Neovim last week and I love it."), _a("Welcome to the club.")),
        ),
        (Question("Which editor do I use?", ("neovim",), ("vs code",), ("neovim",)),),
    ),
    Scenario(
        "update-07",
        "update",
        (
            (_u("I am learning Spanish."), _a("Buena suerte.")),
            (
                _u("I dropped Spanish; I am learning Portuguese now that I live in Lisbon."),
                _a("Makes sense."),
            ),
        ),
        (
            Question(
                "Which language am I learning?", ("portuguese",), ("spanish",), ("portuguese",)
            ),
        ),
    ),
    Scenario(
        "update-08",
        "update",
        (
            (_u("I hate pineapple on pizza."), _a("A strong stance.")),
            (_u("Okay I admit it, pineapple on pizza is actually great."), _a("Converted.")),
        ),
        (
            Question(
                "Do I like pineapple on pizza now?", ("great", "pineapple"), (), ("pineapple",)
            ),
        ),
    ),
    # -- preference application (6) --------------------------------------------------
    Scenario(
        "pref-01",
        "preference",
        (
            (_u("I am vegetarian, I never eat meat."), _a("Noted, vegetarian.")),
            (_u("I need dinner ideas for this week."), _a("How about lentil curry?")),
        ),
        (
            Question(
                "Suggest a dinner for me.",
                ("lentil", "vegetarian", "curry"),
                ("meat", "chicken"),
                ("vegetarian",),
            ),
        ),
    ),
    Scenario(
        "pref-02",
        "preference",
        (
            (_u("I prefer short answers, no fluff please."), _a("Understood, short answers.")),
            (_u("What is the capital of Portugal?"), _a("Lisbon.")),
        ),
        (Question("What is 12 times 12?", ("144",), (), ()),),
    ),
    Scenario(
        "pref-03",
        "preference",
        (
            (_u("I love sci-fi movies, especially slow philosophical ones."), _a("Arrival fan?")),
            (_u("Recommend something for tonight."), _a("Maybe Stalker?")),
        ),
        (Question("Recommend a movie for me.", ("sci-fi", "stalker", "arrival"), (), ("sci-fi",)),),
    ),
    Scenario(
        "pref-04",
        "preference",
        (
            (_u("I wake up at 5am every day and love mornings."), _a("Early bird.")),
            (_u("When should I schedule deep work?"), _a("Mornings suit you.")),
        ),
        (
            Question(
                "When should I do deep work?", ("morning",), ("evening", "night"), ("morning",)
            ),
        ),
    ),
    Scenario(
        "pref-05",
        "preference",
        (
            (_u("I only read non-fiction, mostly history and science."), _a("Noted.")),
            (_u("Suggest a book."), _a("Maybe Guns, Germs, and Steel?")),
        ),
        (
            Question(
                "Suggest a book for me.",
                ("non-fiction", "history", "science"),
                ("novel", "fiction"),
                ("non-fiction",),
            ),
        ),
    ),
    Scenario(
        "pref-06",
        "preference",
        (
            (_u("I hate meetings before 10am."), _a("Fair enough.")),
            (_u("Help me plan tomorrow: I have a free day."), _a("Start after 10am?")),
        ),
        (
            Question(
                "When can meetings start for me?", ("10am", "10 am"), ("9am", "8am"), ("10am",)
            ),
        ),
    ),
    # -- multi-hop (6): combine two memories ------------------------------------------
    Scenario(
        "hop-01",
        "multihop",
        (
            (_u("I live in Lisbon."), _a("Noted.")),
            (_u("My office is a 40 minute bike ride from home."), _a("Good exercise.")),
        ),
        (
            Question(
                "How do I commute in Lisbon and how long does it take?",
                ("bike", "40 minute", "lisbon"),
                (),
                ("bike",),
            ),
        ),
    ),
    Scenario(
        "hop-02",
        "multihop",
        (
            (_u("I am vegetarian."), _a("Noted.")),
            (_u("I am allergic to peanuts."), _a("Serious, remembered.")),
        ),
        (
            Question(
                "Plan a safe dinner for me.",
                ("vegetarian", "peanut"),
                ("meat", "chicken"),
                ("vegetarian", "peanut"),
            ),
        ),
    ),
    Scenario(
        "hop-03",
        "multihop",
        (
            (_u("My dog Biscuit is a golden retriever."), _a("Lovely.")),
            (_u("Biscuit needs a haircut every 6 weeks."), _a("Schedule it.")),
        ),
        (
            Question(
                "What breed is Biscuit and how often is his haircut?",
                ("golden retriever", "6 weeks"),
                (),
                ("biscuit",),
            ),
        ),
    ),
    Scenario(
        "hop-04",
        "multihop",
        (
            (_u("I work with Postgres."), _a("Noted.")),
            (_u("My biggest table has 2 billion rows."), _a("That is large.")),
        ),
        (
            Question(
                "What database holds my 2 billion row table?",
                ("postgres", "2 billion"),
                (),
                ("postgres",),
            ),
        ),
    ),
    Scenario(
        "hop-05",
        "multihop",
        (
            (_u("I wake at 5am."), _a("Early.")),
            (_u("I swim for an hour each morning."), _a("Healthy.")),
        ),
        (Question("Describe my morning routine.", ("5am", "swim"), (), ("swim",)),),
    ),
    Scenario(
        "hop-06",
        "multihop",
        (
            (_u("My sister Ana visits next month."), _a("Fun.")),
            (_u("We will hike in Sintra together."), _a("Beautiful choice.")),
        ),
        (Question("Who am I hiking with in Sintra?", ("ana", "sintra"), (), ("ana",)),),
    ),
    # -- temporal (4) -------------------------------------------------------------------
    Scenario(
        "temp-01",
        "temporal",
        (
            (_u("In 2021 I graduated from university in Porto."), _a("Congrats.")),
            (_u("In 2023 I started my job in Berlin."), _a("Big move.")),
        ),
        (Question("Where did I graduate in 2021?", ("porto",), ("berlin",), ("porto",)),),
    ),
    Scenario(
        "temp-02",
        "temporal",
        (
            (_u("My dentist appointment is on Friday."), _a("Noted, Friday.")),
            (_u("Actually I moved it to Monday morning."), _a("Monday morning now.")),
        ),
        (Question("When is my dentist appointment?", ("monday",), ("friday",), ("monday",)),),
    ),
    Scenario(
        "temp-03",
        "temporal",
        (
            (_u("I visited Japan in April and loved Kyoto."), _a("Kyoto is special.")),
            (_u("I am going back to Japan in November, this time Osaka."), _a("Enjoy Osaka.")),
        ),
        (Question("Which city did I visit in April?", ("kyoto",), ("osaka",), ("kyoto",)),),
    ),
    Scenario(
        "temp-04",
        "temporal",
        (
            (_u("Last year I ran a marathon in 4 hours."), _a("Strong result.")),
            (_u("This year my goal is under 3 hours 30."), _a("Ambitious.")),
        ),
        (
            Question(
                "What is my marathon goal this year?",
                ("3 hours 30", "under 3"),
                ("4 hours",),
                ("marathon",),
            ),
        ),
    ),
    # -- distractor-heavy (4) --------------------------------------------------------------
    Scenario(
        "dist-01",
        "distractor",
        (
            (_u("I live in Lisbon."), _a("Noted."), *_chatter(4)),
            (
                _u("My friend lives in Berlin, my cousin in Madrid, I visited Paris."),
                _a("Well travelled circle."),
            ),
        ),
        (Question("Where do I live?", ("lisbon",), ("berlin", "madrid", "paris"), ("lisbon",)),),
    ),
    Scenario(
        "dist-02",
        "distractor",
        (
            (_u("I drink oat latte."), _a("Noted."), *_chatter(4)),
            (
                _u("My colleague drinks espresso, my boss drinks tea, I tried matcha once."),
                _a("Lots of drinks around."),
            ),
        ),
        (
            Question(
                "What do I drink?", ("oat", "latte"), ("espresso", "tea", "matcha"), ("latte",)
            ),
        ),
    ),
    Scenario(
        "dist-03",
        "distractor",
        (
            (_u("Biscuit is my golden retriever."), _a("Cute."), *_chatter(4)),
            (
                _u("The neighbor has a poodle, the park has many Labradors."),
                _a("Dog neighborhood."),
            ),
        ),
        (
            Question(
                "What is my dog?",
                ("biscuit", "golden retriever"),
                ("poodle", "labrador"),
                ("biscuit",),
            ),
        ),
    ),
    Scenario(
        "dist-04",
        "distractor",
        (
            (_u("I use Neovim."), _a("Nice."), *_chatter(4)),
            (_u("My team uses VS Code, Emacs, and Sublime across projects."), _a("Eclectic team.")),
        ),
        (
            Question(
                "Which editor do I personally use?",
                ("neovim",),
                ("vs code", "emacs", "sublime"),
                ("neovim",),
            ),
        ),
    ),
    # -- long-gap (4): filler session between fact and question ------------------------------
    Scenario(
        "gap-01",
        "longgap",
        (
            (_u("My passport expires in June 2027."), _a("Plenty of time.")),
            _chatter(6),
            (_u("I am planning a trip abroad."), _a("Exciting.")),
        ),
        (Question("When does my passport expire?", ("june 2027", "2027"), (), ("passport",)),),
    ),
    Scenario(
        "gap-02",
        "longgap",
        (
            (_u("My gym membership renews every January."), _a("Noted.")),
            _chatter(6),
            (_u("Should I keep going to the gym?"), _a("Up to you.")),
        ),
        (Question("When does my gym membership renew?", ("january",), (), ("gym",)),),
    ),
    Scenario(
        "gap-03",
        "longgap",
        (
            (_u("I take vitamin D every winter morning."), _a("Good habit.")),
            _chatter(6),
            (_u("What supplements do I take?"), _a("Let me recall.")),
        ),
        (Question("What supplement do I take in winter?", ("vitamin d",), (), ("vitamin",)),),
    ),
    Scenario(
        "gap-04",
        "longgap",
        (
            (_u("My car insurance renews in September."), _a("Noted, September.")),
            _chatter(6),
            (_u("Anything expiring soon I should handle?"), _a("Let me think.")),
        ),
        (Question("When does my car insurance renew?", ("september",), (), ("insurance",)),),
    ),
)

CATEGORIES = ("recall", "update", "preference", "multihop", "temporal", "distractor", "longgap")

__all__ = ["CATEGORIES", "LONG_SCENARIOS", "Question", "SCENARIOS", "Scenario"]


# -- long-history scenarios (16): 32 sessions, ~9k tokens each -------------------
# Facts planted in sessions 0-2, correction/second fact near session 12,
# distractor facts and chatter everywhere else. Chatter sessions are flagged
# skip_learn (pure filler, no model calls spent on them).

_FILLER_TOPICS: tuple[tuple[str, str], ...] = (
    (
        "The forecast says rain all weekend, typical for this time of year.",
        "Rain again? Maybe a good weekend for indoor plans.",
    ),
    (
        "I finally organized the bookshelf alphabetically by author.",
        "That must have taken a while. How many books?",
    ),
    (
        "Traffic on the ring road was terrible this morning.",
        "Rush hour is the worst. Did you take an alternate route?",
    ),
    (
        "I tried a new pasta recipe with roasted tomatoes and basil.",
        "Sounds delicious. Did it turn out well?",
    ),
    (
        "My phone battery barely lasts until lunch these days.",
        "Might be time for a battery replacement.",
    ),
    (
        "I started watching a documentary series about deep sea creatures.",
        "The anglerfish episode is wild, right?",
    ),
    (
        "The office coffee machine broke down again yesterday.",
        "Third time this month. Someone should fix it properly.",
    ),
    ("I need to buy birthday candles and wrapping paper soon.", "Any big celebration coming up?"),
    (
        "My houseplants are thriving since I moved them near the window.",
        "Light makes all the difference.",
    ),
    (
        "I read an article about urban beekeeping on rooftops.",
        "Fascinating hobby. Would you ever try it?",
    ),
    (
        "The new park downtown opened with a big playground.",
        "I heard the running track there is nice too.",
    ),
    ("I finally fixed the squeaky door hinge with some oil.", "Small repairs are so satisfying."),
    (
        "Gas prices went up again, I noticed at the station.",
        "Everything keeps getting more expensive.",
    ),
    (
        "I am thinking of repainting the bedroom a soft green.",
        "Sage green is calming. Good choice.",
    ),
    (
        "My neighbor is learning to play the trumpet, loudly.",
        "Earplugs might be a good investment.",
    ),
    ("I saw a great meteor shower last night around 2am.", "Lucky! The sky was cloudy here."),
)

_DISTRACTOR_FACTS: tuple[tuple[str, str], ...] = (
    ("My cousin Paulo is a dentist in Porto.", "Porto is a lovely city."),
    ("I own a blue bicycle I ride to the market.", "Market trips by bike sound fun."),
    ("My favorite season is autumn, I love the colors.", "Autumn light is the best."),
    ("I water the garden every Sunday morning.", "A solid routine."),
    ("My grandfather collected old stamps from Asia.", "What a collection that must be."),
    ("I take the bus line 12 to get downtown.", "Reliable route?"),
    ("My laptop is four years old and still fine.", "Good mileage out of it."),
    ("I bake sourdough bread almost every weekend.", "The crust must be great."),
)


def _long_chatter(session_idx: int) -> tuple[Turn, ...]:
    """8-turn filler session, deterministic rotation by session index."""
    turns: list[Turn] = []
    for k in range(6):
        topic = _FILLER_TOPICS[(session_idx * 3 + k) % len(_FILLER_TOPICS)]
        turns.append(_u(topic[0]))
        turns.append(_a(topic[1]))
    d = _DISTRACTOR_FACTS[session_idx % len(_DISTRACTOR_FACTS)]
    turns.append(_u(d[0]))
    turns.append(_a(d[1]))
    return tuple(turns)


def _build_long(
    scn_id: str,
    category: str,
    early: tuple[tuple[Turn, ...], ...],
    mid: tuple[Turn, ...],
    question: Question,
    *,
    n_sessions: int = 48,
    mid_at: int = 18,
) -> Scenario:
    """Assemble a long scenario: early facts, mid correction, chatter around."""
    sessions: list[tuple[Turn, ...]] = []
    skip: list[int] = []
    for i in range(n_sessions):
        if i < len(early):
            sessions.append(early[i])
        elif i == mid_at:
            sessions.append(mid)
        else:
            sessions.append(_long_chatter(i))
            skip.append(i)
    return Scenario(scn_id, category, tuple(sessions), (question,), tuple(skip))


LONG_SCENARIOS: tuple[Scenario, ...] = (
    _build_long(
        "long-recall-01",
        "recall",
        (
            (_u("I live in Lisbon, in the Alfama district."), _a("Alfama, noted.")),
            (_u("My apartment overlooks the Tagus river."), _a("What a view.")),
        ),
        (_u("Still in Alfama, love it here."), _a("Glad to hear it.")),
        Question("Which district do I live in?", ("alfama",), gold=("alfama",)),
    ),
    _build_long(
        "long-recall-02",
        "recall",
        (
            (_u("My daughter is called Ines and she is seven."), _a("Hi Ines.")),
            (_u("Ines plays the violin after school."), _a("Lovely instrument.")),
        ),
        (_u("Ines has a violin recital soon."), _a("Break a leg to her.")),
        Question("What is my daughter called?", ("ines",), gold=("ines",)),
    ),
    _build_long(
        "long-recall-03",
        "recall",
        (
            (_u("I am a marine biologist studying octopuses."), _a("Fascinating field.")),
            (_u("My lab is at the oceanography institute."), _a("Great workplace.")),
        ),
        (_u("The octopus project got extended."), _a("Good news.")),
        Question("What do I study?", ("octopus", "marine"), gold=("octopus",)),
    ),
    _build_long(
        "long-recall-04",
        "recall",
        (
            (_u("I am allergic to bee stings, I carry an epi-pen."), _a("Stay safe.")),
            (_u("It happened once on a hiking trip."), _a("Scary experience.")),
        ),
        (_u("Checked: epi-pen still valid."), _a("Good.")),
        Question("What am I allergic to?", ("bee",), gold=("bee",)),
    ),
    _build_long(
        "long-update-01",
        "update",
        (
            (_u("I live in Berlin."), _a("Berlin, got it.")),
            (_u("Been in Berlin for five years now."), _a("Quite a while.")),
        ),
        (
            _u("Update: I actually moved from Berlin to Lisbon last month, not Berlin anymore."),
            _a("Lisbon now, updated."),
        ),
        Question("Where do I live?", ("lisbon",), ("berlin",), ("lisbon",)),
    ),
    _build_long(
        "long-update-02",
        "update",
        (
            (_u("I drive a red Honda Civic."), _a("Nice car.")),
            (_u("The Honda just passed inspection."), _a("Good.")),
        ),
        (
            _u("Correction: I sold the Honda and bought a blue Tesla, not a Honda anymore."),
            _a("Big upgrade."),
        ),
        Question("What car do I drive now?", ("tesla",), ("honda",), ("tesla",)),
    ),
    _build_long(
        "long-update-03",
        "update",
        (
            (_u("I work as a frontend developer."), _a("Cool.")),
            (_u("Mostly React and TypeScript work."), _a("Solid stack.")),
        ),
        (
            _u("I changed roles and now work as a backend engineer instead of frontend."),
            _a("Congrats."),
        ),
        Question("What is my current role?", ("backend",), ("frontend",), ("backend",)),
    ),
    _build_long(
        "long-update-04",
        "update",
        (
            (_u("I drink regular cow milk daily."), _a("Noted.")),
            (_u("Always cow milk in coffee."), _a("Okay.")),
        ),
        (
            _u("I switched to oat milk; cow milk upsets my stomach now, no more cow milk."),
            _a("Oat milk it is."),
        ),
        Question("What milk do I drink?", ("oat",), ("cow milk",), ("oat",)),
    ),
    _build_long(
        "long-dist-01",
        "distractor",
        (
            (_u("I live in Valencia."), _a("Noted.")),
            (_u("Valencia has great paella."), _a("So I hear.")),
        ),
        (
            _u("My brother lives in Berlin, my aunt in Madrid, colleagues in Paris and Rome."),
            _a("Spread out family."),
        ),
        Question(
            "Where do I live?", ("valencia",), ("berlin", "madrid", "paris", "rome"), ("valencia",)
        ),
    ),
    _build_long(
        "long-dist-02",
        "distractor",
        (
            (_u("I drink barley tea every evening."), _a("Noted.")),
            (_u("Barley tea helps me sleep."), _a("Good habit.")),
        ),
        (
            _u("At work everyone drinks espresso, cola, wine, and smoothies all day."),
            _a("Quite the menu."),
        ),
        Question(
            "What do I drink in the evening?",
            ("barley",),
            ("espresso", "cola", "wine", "smoothies"),
            ("barley",),
        ),
    ),
    _build_long(
        "long-dist-03",
        "distractor",
        (
            (_u("My cat is called Miso."), _a("Cute name.")),
            (_u("Miso is a grey tabby cat."), _a("Adorable.")),
        ),
        (
            _u("The shelter also has dogs named Rex and Luna, plus a parrot called Kiko."),
            _a("Full house there."),
        ),
        Question("What is my cat called?", ("miso",), ("rex", "luna", "kiko"), ("miso",)),
    ),
    _build_long(
        "long-dist-04",
        "distractor",
        (
            (_u("I write Go for a living."), _a("Nice.")),
            (_u("Go concurrency suits my work."), _a("Indeed.")),
        ),
        (_u("Friends use Rust, Java, Haskell, and Elixir for their jobs."), _a("Eclectic circle.")),
        Question(
            "Which language do I write for a living?",
            ("go",),
            ("rust", "java", "haskell", "elixir"),
            ("go",),
        ),
    ),
    _build_long(
        "long-gap-01",
        "longgap",
        (
            (_u("My passport expires in June 2027."), _a("Plenty of time.")),
            (_u("I keep the passport in the desk drawer."), _a("Safe spot.")),
        ),
        (_u("Renewal paperwork takes months."), _a("Start early.")),
        Question("When does my passport expire?", ("june 2027", "2027"), (), ("passport",)),
    ),
    _build_long(
        "long-gap-02",
        "longgap",
        (
            (_u("My piano needs tuning every spring."), _a("Noted.")),
            (_u("It is an old upright piano."), _a("Charming.")),
        ),
        (_u("The tuner moved away last year."), _a("Need a new one.")),
        Question("When does my piano need tuning?", ("spring",), (), ("piano", "tuning")),
    ),
    _build_long(
        "long-gap-03",
        "longgap",
        (
            (_u("I take magnesium before bed."), _a("Good habit.")),
            (_u("It helps my sleep quality."), _a("Great.")),
        ),
        (_u("Pharmacy was out of stock once."), _a("Annoying.")),
        Question("What do I take before bed?", ("magnesium",), (), ("magnesium",)),
    ),
    _build_long(
        "long-gap-04",
        "longgap",
        (
            (_u("My ferry pass renews each April."), _a("Noted, April.")),
            (_u("I commute by ferry daily."), _a("Scenic commute.")),
        ),
        (_u("Ferry prices rose this year."), _a("Everything rises.")),
        Question("When does my ferry pass renew?", ("april",), (), ("ferry",)),
    ),
)

ALL_SCENARIOS: tuple[Scenario, ...] = SCENARIOS + LONG_SCENARIOS
