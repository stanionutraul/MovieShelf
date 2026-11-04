
import json
import logging
from pyexpat.errors import messages

from movieshelf.llm import MovieShelfLLM
from movieshelf.schemas.recommendation_context import RecommendationContext
from movieshelf.tools import MovieCatalogTool


logger = logging.getLogger(__name__)


class RecommendationAgent:
    """AI agent that retrieves movie data through registered tools."""

    MAX_TOOL_CALLS = 5

    SYSTEM_PROMPT = """
You are MovieShelf, a movie recommendation agent.

YOUR FIRST PRIORITY:
Retrieve movie data using the available tools.
You MUST call at least one tool before producing any recommendation.

IMPORTANT:
- Do not explain your reasoning.
- Do not describe which tools you intend to use.
- Do not write a plan in natural language.
- Do not answer the user before retrieving catalog data.
- Your first response must contain one or more tool calls.
- After receiving tool results, decide whether more tools are needed.
- Use the remaining tool-call budget to investigate the request.
- Only produce the final recommendation after examining tool results.

AVAILABLE TOOLS:
- search_movies: Search for movies by title.
- filter_movies: Filter by genres and rating statistics.
- get_movie_details: Retrieve a movie and its MovieLens tags.
- get_random_movies: Retrieve random movies.
- get_movies_by_tags: Search movies using MovieLens user-generated tags.

TOOL SELECTION:
- For genre requirements, use filter_movies.
- For themes and descriptive preferences, use get_movies_by_tags.
- For a known movie title, use search_movies.
- For more information about a candidate, use get_movie_details.
- Combine tools when the request requires multiple types of evidence.

CATALOG LIMITATIONS:
- MovieLens provides titles, years, genres, ratings, rating counts and user-generated tags.
- Tags are not verified plot summaries.
- Never invent plot details, narrative structures, scenes or characters.
- Never interpret missing tags as proof that a movie lacks something.
- Distinguish catalog facts from tag evidence and your own interpretation.

RECOMMENDATIONS:
- Respect explicit requirements and exclusions.
- Treat examples as preference references, not automatic recommendations.
- Relevance matters more than popularity.
- Include exact titles, years and available ratings.
- Explain matches using retrieved evidence.
- Be transparent about uncertainty and weak matches.
- Do not invent movies or catalog information.

Respond in the same language as the user.
"""

    def __init__(
        self,
        catalog_tool: MovieCatalogTool,
        llm: MovieShelfLLM | None = None,
    ):
        self.llm = llm or MovieShelfLLM()
        self.catalog_tool = catalog_tool

        self.tool_methods = {
            "search_movies": self.catalog_tool.search_movies,
            "filter_movies": self.catalog_tool.filter_movies,
            "get_movie_details": self.catalog_tool.get_movie_details,
            "get_random_movies": self.catalog_tool.get_random_movies,
            "get_movies_by_tags": self.catalog_tool.get_movies_by_tags,
        }

        self.tools = [
            self._tool_schema(
                "search_movies",
                "Search movies by title.",
                {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                ["query"],
            ),
            self._tool_schema(
                "filter_movies",
                "Filter movies by genres and rating statistics.",
                {
                    "include_genres": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "exclude_genres": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "min_rating_count": {
                        "type": "integer",
                        "minimum": 0,
                    },
                    "min_average_rating": {"type": "number"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                [],
            ),
            self._tool_schema(
                "get_movie_details",
                "Retrieve catalog information and MovieLens user-generated tags for a movie using its MovieLens ID.",
                {
                    "movie_id": {"type": "integer"},
                },
                ["movie_id"],
            ),
            self._tool_schema(
                "get_random_movies",
                "Retrieve a random selection of movies.",
                {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                [],
            ),
            self._tool_schema(
                "get_movies_by_tags",
                "Search movies using MovieLens user-generated tags.",
                {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                ["query"],
            ),
        ]

    @staticmethod
    def _tool_schema(name, description, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _execute_tool(self, tool_name, arguments):
        aliases = {
            "get_movies_by_tags": {
                "function": "query",
                "tag": "query",
            },
            "get_movie_details": {
                "movieId": "movie_id",
            },
        }

        normalized_arguments = {}

        for key, value in arguments.items():
            normalized_key = aliases.get(tool_name, {}).get(key, key)
            normalized_arguments[normalized_key] = value

        method = self.tool_methods.get(tool_name)

        if method is None:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}"
            })

        try:
            logger.info(
                "Executing tool '%s' with arguments: %s",
                tool_name,
                normalized_arguments,
            )

            results = method(**normalized_arguments)

            if isinstance(results, (list, tuple)):
                result_count = len(results)
            else:
                result_count = 1 if results is not None else 0

            logger.info(
                "Tool '%s' returned %s results",
                tool_name,
                result_count,
            )

            return json.dumps(
                results,
                ensure_ascii=False,
                default=str,
            )

        except Exception as exc:
            logger.exception(
                "Tool '%s' execution failed",
                tool_name,
            )

            return json.dumps({
                "error": str(exc)
            }, ensure_ascii=False)

    def run(self, context: RecommendationContext) -> str:
        """Run the agent's tool-calling conversation."""

        logger.info("Starting recommendation agent.")

        context_json = context.model_dump_json(indent=2)

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Analyze this recommendation context and fulfill "
                    "the user's request.\n\n"
                    f"{context_json}"
                ),
            },
        ]

        tool_calls_executed = 0
        successful_tool_calls = 0

        while tool_calls_executed < self.MAX_TOOL_CALLS:
            response = self.llm.chat_with_tools(
                messages=messages,
                tools=self.tools,
            )

            assistant_message = response.message
            messages.append(assistant_message)

            tool_calls = assistant_message.tool_calls or []

            logger.info("Requested tool calls: %s", tool_calls)

            if not tool_calls:
                if successful_tool_calls == 0:
                            logger.warning(
                    "Agent finished without successful tool calls."
                    )

                            return (
                    "Nu am putut obține informații din catalog "
                    "pentru a formula recomandări fundamentate."
                    )

                logger.info(
                "No more tool calls requested. Generating final answer."
                )

            messages.append({
                "role": "user",
                "content": (
                    "Now provide the final movie recommendations. "
                    "Use only the catalog results already retrieved. "
                    "Do not describe your reasoning or planning. "
                    "Do not invent additional tool calls or movie data. "
                    "Give a concise answer in the user's language."
                ),
            })

            final_response = self.llm.chat_with_tools(
                    messages=messages,
                    tools=[],
                )

            return final_response.message.content or (
                    "Nu am putut genera recomandări finale."
                )
        for tool_call in tool_calls:
                if tool_calls_executed >= self.MAX_TOOL_CALLS:
                    break

                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                logger.info(
                    "Executing tool '%s' with arguments: %s",
                    tool_name,
                    arguments,
                )

                result = self._execute_tool(tool_name, arguments)
                logger.info("Tool result preview: %s",result[:2000], )


                try:
                    parsed_result = json.loads(result)

                    if not (
                        isinstance(parsed_result, dict)
                        and "error" in parsed_result
                    ):
                        successful_tool_calls += 1

                except json.JSONDecodeError:
                    logger.exception("Could not parse tool result.")

                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": result,
                })

                tool_calls_executed += 1

        logger.warning(
            "Maximum tool calls reached: %s",
            self.MAX_TOOL_CALLS,
        )

        if successful_tool_calls == 0:
            return (
                "Nu am putut obține informații din catalog "
                "pentru a formula recomandări fundamentate."
            )

        final_response = self.llm.chat_with_tools(
            messages=messages,
            tools=[],
        )

        return final_response.message.content or (
            "Nu am putut genera recomandări finale."
        )
