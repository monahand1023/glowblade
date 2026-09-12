import click


@click.group()
@click.version_option()
def main():
    """lightsaber-fx: turn a home video into a lightsaber VFX clip."""


if __name__ == "__main__":
    main()
