
import json
import logging

from pathlib import Path

from movieshelf.llm import MovieShelfLLM
from movieshelf.schemas.recommendation_context import RecommendationContext
from movieshelf.tools import MovieCatalogTool


logger = logging.getLogger(__name__)


class RecommendationAgent:
    """AI agent that retrieves movie data through registered tools."""

    MAX_TOOL_CALLS = 5


    SYSTEM_PROMPT = """
You are MovieShelf, an intelligent movie recommendation agent.

Your responsibility is to understand the user's movie request,
retrieve relevant movie data using your tools, evaluate the results,
and present useful recommendations.

You MUST use tools to retrieve movie information.
Never invent movie titles, ratings, genres, tags or other facts.

AVAILABLE INFORMATION:
- Movie titles and release years.
- Genres.
- Average ratings and rating counts.
- Weighted ratings.
- MovieLens user-generated tags.

IMPORTANT LIMITATIONS:
- MovieLens tags are user-generated descriptions, not verified plot summaries.
- Tags are evidence of how users describe or associate a movie,
  not proof of its actual plot or content.
- You do not have reliable plot summaries or detailed narrative information.
- Never claim that a movie contains a specific plot twist, narrative
  structure, scene or character unless retrieved information supports it.
- Do not interpret the absence of a tag as proof that a movie lacks
  a particular theme or element.
- Never claim that a movie is the only one, the best one, or the
  closest possible match unless the retrieved results justify that claim.
- Be transparent when available information is insufficient.

TOOL USAGE:
- Use search_movies for title searches.
- Use filter_movies for genre and rating requirements.
- Use get_movie_details to retrieve a movie's catalog information
  and associated tags.
- Use get_random_movies for random recommendations.
- Use get_movies_by_tags for searches based on MovieLens tags.
- Combine multiple tools when necessary to investigate different
  parts of the user's request.
- Respect all explicit exclusions.
- Do not claim to have applied a filter or performed a search
  unless you actually used the corresponding tool.
- When a tool returns no results, do not pretend it found matches.

UNDERSTANDING USER REQUESTS:
Separate the request into:
1. Explicit requirements: things the user directly asks for.
2. Preferences: things the user would like, but which are flexible.
3. Exclusions: things the user does not want.
4. Examples: movies, directors, genres or styles mentioned as
   references for the desired experience.
5. Uncertainties: details that cannot be established from the request
   or available catalog data.

Treat examples as meaningful preference signals.
Do not automatically recommend the examples themselves.
Identify the characteristics they may represent and look for other
movies with supporting catalog evidence.

When several examples are provided, do not focus on only one.
Consider the shared characteristics and the different aspects
each example may represent.

RECOMMENDATION PRINCIPLES:
- Relevance is more important than popularity.
- Consider ratings and rating counts as supporting information.
- Do not recommend movies merely because they have high ratings.
- Evaluate each candidate against the user's actual requirements.
- Do not assume that matching one keyword means matching the
  entire request.
- Prefer a small number of genuinely relevant recommendations.
- If no strong matches exist, say so instead of filling the list
  with weak recommendations.
- Explain compromises and limitations honestly.

EVIDENCE AND EXPLANATIONS:
For each recommendation, distinguish between:

1. Catalog facts:
   Information directly retrieved from the catalog, such as title,
   year, genres and ratings.

2. Tag evidence:
   Tags retrieved from MovieLens that suggest an association
   with a theme, topic or style.
   Describe these as user-generated tags, not verified facts.

3. Interpretation:
   Your reasoning about why the available evidence may make
   the movie relevant to the request.
   Make clear when this is an interpretation rather than a fact.

Never turn an interpretation into a confirmed plot detail.
Never invent alternatives when tools return no results.

When presenting recommendations:
- Include the exact movie title and year when available.
- Include actual ratings and rating counts when available.
- Explain specifically why each movie matches the request.
- Mention relevant tags when they support the recommendation.
- Identify important mismatches or missing evidence.
- Do not make unsupported claims about what a movie contains.

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
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
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
                    "min_rating_count": {"type": "integer", "minimum": 0},
                    "min_average_rating": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                [],
            ),
            self._tool_schema(
                "get_movie_details",
                "Retrieve catalog information for a movie using its MovieLens ID.",
                {
                    "movie_id": {"type": "integer"},
                },
                ["movie_id"],
            ),
            self._tool_schema(
                "get_random_movies",
                "Retrieve a random selection of movies.",
                {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                [],
            ),
            self._tool_schema(
                "get_movies_by_tags",
                "Search movies using MovieLens user-generated tags.",
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
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

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a registered catalog tool."""

        method = self.tool_methods.get(tool_name)

        if method is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            logger.info(
                "Executing tool '%s' with arguments: %s",
                tool_name,
                arguments,
            )
            logger.info(
    "Tool arguments received: %s",
    arguments,
)


            results = method(**arguments)

            logger.info(
                "Tool execution completed. Result count: %s",
                len(results) if isinstance(results, list) else (
                    1 if results is not None else 0
                ),
            )

            return json.dumps(
                results,
                ensure_ascii=False,
                default=str,
            )

        except Exception:
            logger.exception("Tool '%s' execution failed.", tool_name)

            return json.dumps(
                {"error": "Tool execution failed."},
                ensure_ascii=False,
            )

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
                    "the user's request:\n\n"
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

            logger.info(
                "Assistant response: %s",
                assistant_message,
            )

            logger.info(
                "Requested tool calls: %s",
                assistant_message.tool_calls,
            )

            messages.append(assistant_message)

            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                if successful_tool_calls == 0:
                    return (
                        "Nu am putut obține informații din catalog "
                        "pentru a formula recomandări fundamentate."
                    )

                return assistant_message.content or (
                    "Nu am putut genera recomandări."
                )

            for tool_call in tool_calls:
                if tool_calls_executed >= self.MAX_TOOL_CALLS:
                    break

                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                result = self._execute_tool(tool_name, arguments)

                try:
                    parsed_result = json.loads(result)
                    if not (
                        isinstance(parsed_result, dict)
                        and "error" in parsed_result
                    ):
                        successful_tool_calls += 1
                except json.JSONDecodeError:
                    pass

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": result,
                    }
                )

                tool_calls_executed += 1

        final_response = self.llm.chat_with_tools(
            messages=messages,
            tools=[],
        )

        if successful_tool_calls == 0:
            return (
                "Nu am putut obține informații din catalog "
                "pentru a formula recomandări fundamentate."
            )

        return final_response.message.content or (
            "Nu am putut genera recomandări finale."
        )
