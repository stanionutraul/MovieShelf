from movieshelf.agents.request_analyzer import MovieRequestAnalyzer


def main():
    analyzer = MovieRequestAnalyzer()

    user_request = """
    I want science fiction films about space exploration.
    Avoid horror movies.
    """

    result = analyzer.analyze(user_request)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()