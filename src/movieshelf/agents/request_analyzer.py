
import re

from movieshelf.llm import MovieShelfLLM
from movieshelf.schemas.movie_request import MovieRequest


class MovieRequestAnalyzer:
    """Analyzes natural-language movie requests."""

    SYSTEM_PROMPT = """
You are MovieShelf's movie request analysis engine.

Your task is to interpret the user's movie request and
represent it accurately using the provided JSON schema.

CORE PRINCIPLES:

1. UNDERSTAND INTENT
Understand the underlying narrative, thematic, emotional,
and viewing-experience characteristics of the request.
Do not reduce complex requests to generic genres or keywords.

2. NARRATIVE MECHANISMS
Distinguish between:
- Narrative clues
- Foreshadowing
- Plot twists
- Revelation of a twist
- Recontextualization
- Hidden information available to attentive viewers

A clue that foreshadows a future revelation does not
necessarily reveal the entire twist.

Populate narrative_mechanisms when relevant.

3. REQUIREMENTS AND PREFERENCES
- required_criteria: explicit mandatory conditions.
- preferred_criteria: desirable but non-mandatory qualities.
- non_constraints: qualities or themes the user explicitly says are not
  necessary. Do not turn these into preferences, requirements, or exclusions.
- For example, "It doesn't have to be romantic or holiday-themed" means
  neither romance nor holiday themes are required; it does not mean to
  prefer or exclude them.
- "Bonus points for subtitles" is an optional preference, not a requirement.
- Set requires_plot_evidence only when matching requires verified facts
  about plot, characters, or narrative events. Mood, genre, style, viewing
  situation, and general emotional effect do not require plot evidence.
- Every criterion must be supported by words or a clear synonym in the
  user's request. Do not infer viewing conditions, runtime, effort level,
  or story preferences from time of day or personal circumstances.
- Treat time, snacks, and the viewing setup as context, not movie criteria.
- A user's reason for wanting a movie is context, not a request for a movie
  about that circumstance. For example, job stress does not imply a desire
  to watch a movie about unemployment or illness.
- genres_include and genres_exclude: explicit genre requests.
- genres_match: use "any" by default for multiple included genres; use "all"
  only when the user explicitly requires every included genre.
- Keep assumptions separate from explicit requirements.
- Do not turn general viewing preferences into strict requirements.

4. VIEWING EXPERIENCE
- viewing_experience.atmosphere: explicit atmosphere qualities such as
  cozy, warm, comforting, or whimsical.
- viewing_experience.tone: explicit tone qualities such as lighthearted,
  optimistic, uplifting, or feel-good.
- viewing_experience.mental_effort: low, moderate, or high, only when
  supported by the user's wording.
- viewing_experience.emotional_intensity: low, moderate, or high, only
  when explicitly supported. Do not infer it from atmosphere or tone.
- viewing_experience.pacing: slow, moderate, or fast, only when explicitly
  supported.
- Keep atmosphere, tone, mental effort, emotional intensity, and pacing
  separate. Do not infer one from another.
- Do not infer low mental effort merely because a movie is described as
  cozy or comforting.
- Do not infer low emotional intensity merely because a movie is described
  as lighthearted or feel-good.
- Populate only qualities supported by the user's request. Leave other
  fields empty or null.

4. EXAMPLES
- Only interpret movies explicitly mentioned by the user.
- Use examples to understand the underlying intent.
- Do not automatically treat every characteristic of an
  example as a requirement.
- Populate relevant_mechanisms only when supported by
  the user's description.
- Never invent movie titles or add recommendations.
- Keep explanations concise and spoiler-free.

5. SPOILERS AND UNCERTAINTY
- Never reveal spoilers unless explicitly permitted.
- Do not invent plot details or interpretations.
- Do not invent requirements, timestamps, genres,
  or levels of subtlety.
- Ask for clarification only when essential.

6. OUTPUT
Return only data matching the provided JSON schema.
Keep every field concise and relevant.
Do not recommend movies.

IMPORTANT:
If the user gives examples such as The Prestige and The Thing
and asks for movies where early details foreshadow later
revelations, interpret the intent as narrative clues,
foreshadowing, hidden information, or recontextualization.

Do not automatically infer that the user wants horror,
science fiction, mystery, or immediately recognizable twists.

GENRE FILTERS:
- Populate genres_include only when the user explicitly asks for a genre.
- Populate genres_exclude only when the user explicitly rejects a genre.
- Do not infer genres from examples, studios, themes, plot descriptions,
  moods, or other preferences.
- If the wording is not an explicit genre request or rejection, leave the
  corresponding field empty.

CONTENT AVOIDANCES:
- Populate content_avoidances only when the user clearly asks to avoid gore
  or jump scares.
- Do not treat soft wording such as "not necessarily looking for cheap gore
  or jump scares just for the sake of it" as an absolute exclusion.
"""

    GENRE_ALIASES = {
        "Action": ("action",),
        "Adventure": ("adventure",),
        "Animation": ("animation", "animated", "cartoon", "cartoons"),
        "Children": ("children", "children's", "kids"),
        "Comedy": ("comedy", "comedies", "comedic"),
        "Crime": ("crime", "criminal"),
        "Documentary": ("documentary", "documentaries"),
        "Drama": ("drama", "dramatic"),
        "Fantasy": ("fantasy",),
        "Film-Noir": ("film noir", "noir"),
        "Horror": ("horror",),
        "Musical": ("musical", "musicals"),
        "Mystery": ("mystery", "mysteries"),
        "Romance": ("romance", "romantic"),
        "Sci-Fi": ("sci-fi", "sci fi", "science fiction"),
        "Thriller": ("thriller", "thrillers"),
        "War": ("war",),
        "Western": ("western",),
    }

    GENRE_NEGATION_PATTERN = re.compile(
        r"\b(?:no|not|avoid(?:ing)?|without|exclude|excluding|"
        r"don't want|do not want|not interested in|not looking for)\b"
    )
    NON_CONSTRAINT_PATTERNS = (
        re.compile(
            r"\b(?:doesn['’]?t|does not|do not|don['’]?t) have to be\s+"
            r"([^.!?;\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:doesn['’]?t|does not|do not|don['’]?t) need to be\s+"
            r"([^.!?;\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bnot necessarily(?: looking for| need(?:ed)? to be| be)?\s+"
            r"([^.!?;\n]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:not required to be|need not be|is not a requirement|"
            r"are not a requirement)\s+([^.!?;\n]+)",
            re.IGNORECASE,
        ),
    )
    OPTIONAL_PREFERENCE_PATTERN = re.compile(
        r"\b(?:bonus points?|extra points?|would be (?:a )?(?:plus|bonus)|"
        r"as a bonus|ideally|preferably|happy (?:to|with|watching)|open to)\b",
        re.IGNORECASE,
    )
    NON_CONSTRAINT_FILLER_WORDS = {
        "a", "an", "and", "any", "be", "genre", "genres", "kind",
        "of", "or", "particular", "specific", "theme", "themes", "type",
    }
    CONTENT_AVOIDANCE_ALIASES = {
        "gore": re.compile(
            r"\b(?:gore|graphic violence|violen(?:ce|ță) grafic(?:ă|a))\b",
            re.IGNORECASE,
        ),
        "jump scares": re.compile(
            r"\b(?:jump[\s-]?scares?|jump[\s-]?scare-?uri|"
            r"sperieturi bruște|sperieturi bruste)\b",
            re.IGNORECASE,
        ),
    }
    CONTENT_NEGATION_PATTERN = re.compile(
        r"\b(?:no|without|avoid(?:ing)?|exclude|excluding|"
        r"don't want|do not want|not interested in|not looking for|"
        r"not a fan of|nu vreau|nu doresc|fără|fara|evit(?:ă|a)?)\b",
        re.IGNORECASE,
    )
    SOFT_AVOIDANCE_PATTERN = re.compile(
        r"\b(?:not necessarily|nu neapărat|nu neaparat)\b",
        re.IGNORECASE,
    )
    CRITERION_STOP_WORDS = {
        "a", "an", "and", "for", "in", "movie", "of", "or", "the",
        "to", "vibe", "with",
    }
    CRITERION_CONTEXT_TERMS = {
        "3am", "late", "night", "popcorn", "snack", "snacks", "watch",
    }

    @classmethod
    def _criterion_is_supported(
        cls,
        criterion: str,
        user_request: str,
    ) -> bool:
        criterion_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", criterion.casefold())
            if (
                term not in cls.CRITERION_STOP_WORDS
                and term not in cls.CRITERION_CONTEXT_TERMS
            )
        }
        request_terms = set(
            re.findall(r"[a-z0-9]+", user_request.casefold())
        )
        return any(
            criterion_term in request_terms
            or any(
                len(criterion_term) >= 4
                and len(request_term) >= 4
                and (
                    criterion_term.startswith(request_term)
                    or request_term.startswith(criterion_term)
                )
                for request_term in request_terms
            )
            for criterion_term in criterion_terms
        )

    @classmethod
    def _non_constraints(cls, user_request: str) -> list[str]:
        phrases = []
        seen = set()
        for pattern in cls.NON_CONSTRAINT_PATTERNS:
            for match in pattern.finditer(user_request):
                for phrase in re.split(
                    r",|\bor\b|\band\b",
                    match.group(1),
                    flags=re.IGNORECASE,
                ):
                    cleaned = " ".join(
                        word
                        for word in re.findall(
                            r"[A-Za-z0-9'-]+",
                            phrase,
                        )
                        if word.casefold()
                        not in cls.NON_CONSTRAINT_FILLER_WORDS
                    ).strip()
                    normalized = cleaned.casefold()
                    if cleaned and normalized not in seen:
                        phrases.append(cleaned)
                        seen.add(normalized)
        return phrases

    @classmethod
    def _criterion_conflicts_with_non_constraint(
        cls,
        criterion: str,
        non_constraints: list[str],
    ) -> bool:
        weak_terms = cls.NON_CONSTRAINT_FILLER_WORDS | {
            "looking", "movie", "movies", "film", "films", "for",
        }
        criterion_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", criterion.casefold())
            if term not in weak_terms
        }
        non_constraint_terms = {
            term
            for phrase in non_constraints
            for term in re.findall(r"[a-z0-9]+", phrase.casefold())
            if term not in weak_terms
        }
        return bool(criterion_terms & non_constraint_terms)

    @classmethod
    def _criterion_is_optional(
        cls,
        criterion: str,
        user_request: str,
    ) -> bool:
        criterion_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", criterion.casefold())
            if term not in cls.CRITERION_STOP_WORDS
        }
        for term in criterion_terms:
            for match in re.finditer(
                rf"\b{re.escape(term)}\b",
                user_request,
                re.IGNORECASE,
            ):
                sentence_start = max(
                    user_request.rfind(mark, 0, match.start())
                    for mark in ".!?;\n"
                ) + 1
                clause_prefix = re.split(
                    r",|\b(?:and|but|however|although)\b",
                    user_request[sentence_start:match.start()],
                    flags=re.IGNORECASE,
                )[-1]
                if cls.OPTIONAL_PREFERENCE_PATTERN.search(clause_prefix):
                    return True
        return False

    def __init__(self, llm: MovieShelfLLM | None = None):
        self.llm = llm or MovieShelfLLM()

    @classmethod
    def _genre_mentions(
        cls,
        user_request: str,
        genre: str,
    ) -> list[tuple[int, bool, bool]]:
        """Return explicit genre mentions and whether each is negated."""

        aliases = cls.GENRE_ALIASES.get(genre)
        if aliases is None:
            aliases = (genre.casefold(),)

        mentions = []
        for alias in aliases:
            pattern = re.compile(
                rf"(?<!\w){re.escape(alias)}(?!\w)",
                re.IGNORECASE,
            )
            for match in pattern.finditer(user_request):
                sentence_start = max(
                    user_request.rfind(mark, 0, match.start())
                    for mark in ".!?;\n"
                ) + 1
                prefix = user_request[sentence_start:match.start()]
                clause_parts = re.split(
                    r"\b(?:but|however|although|except|rather)\b",
                    prefix,
                    flags=re.IGNORECASE,
                )
                clause_prefix = clause_parts[-1]
                optional = bool(
                    re.search(
                        r"\b(?:doesn['’]?t|does not|do not|don['’]?t) "
                        r"(?:have to|need to) be\b|"
                        r"\bnot necessarily\b|"
                        r"\bnot required to be\b|\bneed not be\b",
                        clause_prefix,
                        re.IGNORECASE,
                    )
                )
                mentions.append((
                    match.start(),
                    bool(cls.GENRE_NEGATION_PATTERN.search(clause_prefix)),
                    optional,
                ))

        return mentions

    @classmethod
    def _keep_explicit_genres(
        cls,
        genres: list[str],
        user_request: str,
        *,
        excluded: bool,
    ) -> list[str]:
        retained = []
        seen = set()

        for genre in genres:
            if not isinstance(genre, str) or not genre.strip():
                continue

            genre = genre.strip()
            normalized = genre.casefold()
            if normalized in seen:
                continue

            expected_negation = excluded
            if any(
                is_negated == expected_negation and not is_optional
                for _, is_negated, is_optional
                in cls._genre_mentions(user_request, genre)
            ):
                retained.append(genre)
                seen.add(normalized)

        return retained

    @classmethod
    def _explicit_content_avoidances(
        cls,
        user_request: str,
    ) -> list[str]:
        avoidances = []
        for avoidance, pattern in cls.CONTENT_AVOIDANCE_ALIASES.items():
            for match in pattern.finditer(user_request):
                sentence_start = max(
                    user_request.rfind(mark, 0, match.start())
                    for mark in ".!?;\n"
                ) + 1
                prefix = user_request[sentence_start:match.start()]
                clause_parts = re.split(
                    r"\b(?:but|however|although|except)\b|"
                    r"(?<!\w)(?:dar|însă|insa|except)\b",
                    prefix,
                    flags=re.IGNORECASE,
                )
                clause_prefix = clause_parts[-1]
                sentence = user_request[sentence_start:]
                if (
                    cls.SOFT_AVOIDANCE_PATTERN.search(sentence)
                    or not cls.CONTENT_NEGATION_PATTERN.search(clause_prefix)
                ):
                    continue
                avoidances.append(avoidance)
                break

        return avoidances

    @staticmethod
    def _explicit_release_year_range(
        user_request: str,
    ) -> tuple[int | None, int | None]:
        decade_match = re.search(
            r"\b(?:(18|19|20)?(\d{2})s)\b",
            user_request,
            re.IGNORECASE,
        )
        if decade_match:
            century, decade = decade_match.groups()
            decade_number = int(decade)
            if century:
                start = int(century) * 100 + decade_number
            else:
                start = (
                    2000 + decade_number
                    if decade_number < 30
                    else 1900 + decade_number
                )
            return start, start + 9

        range_match = re.search(
            r"\b(?:between\s+)?((?:18|19|20)\d{2})"
            r"\s*(?:-|–|—|to|through|and)\s*"
            r"((?:18|19|20)\d{2})\b",
            user_request,
            re.IGNORECASE,
        )
        if range_match:
            start, end = map(int, range_match.groups())
            if start <= end:
                return start, end

        return None, None

    def analyze(self, user_request: str) -> MovieRequest:
        """Analyze a user's request and return structured data."""

        if not user_request.strip():
            raise ValueError("The movie request cannot be empty.")

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"USER REQUEST:\n{user_request.strip()}"
        )

        result = self.llm.generate_structured(
            prompt=prompt,
            schema=MovieRequest,
        )

        movie_request = MovieRequest.model_validate(result)
        non_constraints = self._non_constraints(user_request)
        movie_request.non_constraints = non_constraints
        movie_request.required_criteria = [
            criterion
            for criterion in movie_request.required_criteria
            if self._criterion_is_supported(criterion, user_request)
            and not self._criterion_is_optional(criterion, user_request)
            and not self._criterion_conflicts_with_non_constraint(
                criterion,
                non_constraints,
            )
        ]
        movie_request.preferred_criteria = [
            criterion
            for criterion in movie_request.preferred_criteria
            if self._criterion_is_supported(criterion, user_request)
            and not self._criterion_conflicts_with_non_constraint(
                criterion,
                non_constraints,
            )
        ]
        (
            movie_request.release_year_from,
            movie_request.release_year_to,
        ) = self._explicit_release_year_range(user_request)
        movie_request.content_avoidances = (
            self._explicit_content_avoidances(user_request)
        )
        movie_request.genres_include = self._keep_explicit_genres(
            movie_request.genres_include,
            user_request,
            excluded=False,
        )
        movie_request.genres_exclude = self._keep_explicit_genres(
            movie_request.genres_exclude,
            user_request,
            excluded=True,
        )
        if (
            movie_request.genres_match == "all"
            and not re.search(
                r"\b(?:all|every|both)\b",
                user_request,
                re.IGNORECASE,
            )
        ):
            movie_request.genres_match = "any"

        mentioned_titles = {
            example.movie_title.casefold()
            for example in movie_request.examples_interpretation
        }

        if not mentioned_titles:
            movie_request.examples_interpretation = []
        else:
            movie_request.examples_interpretation = [
                example
                for example in movie_request.examples_interpretation
                if example.movie_title.casefold() in user_request.casefold()
            ]

        return movie_request
