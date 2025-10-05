from pettingzoo.test import parallel_api_test
import CustomEnvironment


def main():
    with CustomEnvironment() as env:
        parallel_api_test(env)


if __name__ == "__main__":
    main()
