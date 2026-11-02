
from movieshelf.agents.request_analyzer import MovieRequestAnalyzer


def main():
    analyzer = MovieRequestAnalyzer()

    test_requests = [
        {
            "name": "Test 1 - Early Plot Twists",
            "request": """
            Films that tell you the twist right at the start
            if you're paying attention?

            I'm looking for films like The Prestige, where the
            explanation for the dove trick foreshadows what happens
            to Hugh Jackman's character, or The Thing, where the
            Norwegian scientists blatantly say that the dog is a
            creature in disguise, but it's in Norwegian, so most
            viewers won't understand what they're saying.

            Please avoid spoilers in the recommendations.
            """
        },
        {
            "name": "Test 2 - True Detective",
            "request": """
            I've watched a lot of crime shows and movies, but
            nothing has come close to True Detective season 1.

            The atmosphere, the writing, Rust Cohle, the whole
            story... just incredible.

            I've been trying to find something that gives me the
            same feeling, but nothing has worked so far.

            Any recommendations? Doesn't have to be exactly the
            same, just something equally dark, gripping and well
            written. Preferably on Netflix or HBO.
            """
        },
        {
            "name": "Test 3 - Jake Gyllenhaal",
            "request": """
            I've recently been watching Jake Gyllenhaal's films.
            I watched Nightcrawler, Prisoners, Enemy, and Donnie Darko.

            I fucking loved Nightcrawler. The character driven
            storytelling, the plot, Jake's performance, the direction.
            Everything just clicked for me.

            I liked Prisoners too, especially the performances and
            some of Villeneuve's directorial choices. However, I felt
            like the movie had the potential to be something truly
            exceptional but fell a little short.

            I watched Donnie Darko recently as well. I didn't
            understand everything on my first watch, but reading
            about the lore and interpretations afterwards made the
            experience worthwhile.

            Enemy, on the other hand, really didn't work for me.
            I understand the general interpretation, but I felt
            the way the story was presented didn't adequately
            support it.

            I don't mind ambiguity or movies that make you think,
            but I want the storytelling to give me something to
            work with rather than having to rely on explanations
            afterwards.

            What Jake Gyllenhaal movie should I watch next?
            I'm also open to recommendations outside his
            filmography.
            """
        },
        {
            "name": "Test 4 - Thanksgiving",
            "request": """
            I'm looking for any Thanksgiving movies.

            The only one I can think of is Planes, Trains and
            Automobiles, which was ok, but too much Steve Martin.

            What does everyone have?
            """
        }
    ]

    for test in test_requests:
        print("\n" + "=" * 70)
        print(f"MOVIESHELF - {test['name'].upper()}")
        print("=" * 70)

        try:
            result = analyzer.analyze(test["request"])
            print(result.model_dump_json(indent=2))

        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
