"""40 synthetic multi-session scenarios with planted facts and rubric grading."""

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

__all__ = ["CATEGORIES", "Question", "SCENARIOS", "Scenario"]
