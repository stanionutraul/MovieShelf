
from movieshelf.llm import MovieShelfLLM


def main():
    llm = MovieShelfLLM()

    prompt = """
    You are MovieShelf, an intelligent AI-powered movie
    discovery assistant.

    Your task is to deeply understand the user's request,
    identify its underlying narrative concepts, distinguish
    essential requirements from examples, and explain what
    kind of movies the user is looking for.

    USER REQUEST:

    Films that tell you the twist right at the start if
    you're paying attention?

    I'm looking for films like The Prestige, where the
    explanation for the dove trick foreshadows what happens
    to Hugh Jackman's character, or The Thing, where the
    Norwegian scientists blatantly say that the dog is a
    creature in disguise, but it's in Norwegian, so most
    viewers won't understand what they're saying.

    IMPORTANT REQUIREMENTS:

    1. Do not recommend movies yet.
    2. Do not reveal or invent movie spoilers.
    3. Understand the narrative mechanism behind the examples.
    4. Identify the similarities between the examples.
    5. Distinguish the actual request from superficial
       similarities such as genre, director, or actors.
    6. Identify possible keywords and concepts that could
       help a movie search engine find matching films.
    7. Identify ambiguities or aspects that may require
       clarification.

    Think carefully about the request.

    Return your analysis using these sections:

    1. CORE INTENT
       What is the user really looking for?

    2. NARRATIVE MECHANISMS
       What storytelling techniques are relevant?

    3. ESSENTIAL CRITERIA
       What must a matching movie contain?

    4. SECONDARY CRITERIA
       What characteristics are desirable but not mandatory?

    5. SEARCH CONCEPTS
       Provide useful search terms and related concepts.

    6. EXCLUSIONS
       What kinds of movies would technically resemble
       the examples but fail to satisfy the request?

    7. CLARIFICATION
       What, if anything, would be useful to ask the user?
    """

    response = llm.generate_response(prompt)

    print("\\n" + "=" * 60)
    print("MOVIESHELF - AI REQUEST ANALYSIS")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
